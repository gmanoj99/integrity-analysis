from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..adapters.ai_usage_logger import (
    STEP_CAMERA_PERCEPTION,
    STEP_SCREEN_CAMERA_PERCEPTION,
    STEP_SCREEN_PERCEPTION,
)
from ..contracts.perception import (
    CachedPerceptionChunk,
    PerceptionAttention,
    PerceptionAudio,
    PerceptionBody,
    PerceptionChunkJobPayload,
    PerceptionChunkResult,
    PerceptionEnvironment,
    PerceptionEvent,
    PerceptionEventKind,
    PerceptionHands,
    PerceptionIdentity,
    PerceptionInteraction,
    PerceptionObservation,
    PerceptionObjects,
    PerceptionPeople,
    PerceptionTernary,
)
from ..contracts.screen_perception import (
    SCREEN_PERCEPTION_VERSION,
    ForegroundAppClass,
    ScreenObservation,
    ScreenTernary,
)
from ..deps import PipelineDeps
from ..gemini_call import generate_and_log
from ..media.perception_media import resolve_media_parts
from ..prompts import (
    PERCEPTION_MAX_OUTPUT_TOKENS,
    PERCEPTION_PROMPT_VERSION,
    PERCEPTION_RESPONSE_SCHEMA,
    PERCEPTION_SYSTEM_PROMPT,
    SCREEN_CAMERA_PERCEPTION_SYSTEM_PROMPT,
    SCREEN_CAMERA_RESPONSE_SCHEMA,
    SCREEN_PERCEPTION_SYSTEM_PROMPT,
    build_perception_chunk_user_prompt,
    build_screen_camera_user_prompt,
    build_screen_perception_user_prompt,
)
from ..prompts.perception import PerceptionChunkUserPromptInput
from ..prompts.shared import (
    DEFAULT_GEMINI_FLASH_MODEL,
    GEMINI_MIN_VIDEO_DURATION_MS,
    SCREEN_CAMERA_PERCEPTION_PROMPT_VERSION,
    SCREEN_PERCEPTION_PROMPT_VERSION,
)

SPEECH_SUMMARY_STUB = "Speech present without parseable content."

SCREEN_CAMERA_MAX_OUTPUT_TOKENS = PERCEPTION_MAX_OUTPUT_TOKENS + 2048

EVENT_KINDS: tuple[PerceptionEventKind, ...] = (
    "sustained_gaze",
    "phone_visible",
    "phone_in_hand",
    "second_person_present",
    "second_person_interaction",
    "face_absent",
    "leave_seat",
    "speech",
    "reaching_outside_frame",
    "notes_visible",
    "headphones_visible",
    "earphone_visible",
    "second_screen_visible",
    "other",
)

SIDECAR_SUFFIXES = ("__events", "__metadata")


def is_sidecar_chunk(chunk_id: str) -> bool:
    return chunk_id.endswith(SIDECAR_SUFFIXES)


def perception_chunk_cache_key(chunk_id: str) -> str:
    return f"{PERCEPTION_PROMPT_VERSION}|chunk:{chunk_id}"


def screen_chunk_cache_key(chunk_id: str) -> str:
    return f"{SCREEN_PERCEPTION_VERSION}|chunk:{chunk_id}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_json_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE).strip()


def ensure_speech_event_completeness(event: PerceptionEvent) -> PerceptionEvent:
    speech_present_yes = event.attrs.get("speechPresent") == "yes" or event.kind == "speech"
    if not speech_present_yes:
        return event

    attrs = dict(event.attrs)
    quality_caveat = event.quality_caveat
    changed = False

    if attrs.get("speechPresent") != "yes":
        attrs["speechPresent"] = "yes"
        changed = True

    existing_summary = attrs.get("conversationSummaryEn")
    if not (isinstance(existing_summary, str) and existing_summary.strip()):
        from_event = (event.summary or "").strip()
        attrs["conversationSummaryEn"] = from_event or SPEECH_SUMMARY_STUB
        quality_caveat = (
            f"{quality_caveat}; speech_summary_backfilled"
            if quality_caveat
            else "speech_summary_backfilled"
        )
        changed = True

    speech_class = attrs.get("speechContentClass")
    if speech_class in (None, "UNKNOWN", "silence"):
        attrs["speechContentClass"] = "unclear"
        quality_caveat = (
            f"{quality_caveat}; speech_class_backfilled"
            if quality_caveat
            else "speech_class_backfilled"
        )
        changed = True

    if not changed:
        return event
    return event.model_copy(update={"attrs": attrs, "quality_caveat": quality_caveat})


def parse_events_response(
    text: str | None,
    payload: PerceptionChunkJobPayload,
) -> PerceptionChunkResult | None:
    if not text:
        return None
    try:
        raw = json.loads(_clean_json_text(text))
        if not isinstance(raw, dict):
            return None
        chunk_duration_ms = int(raw.get("chunkDurationMs") or payload.duration_ms or 60_000)
        fully = raw.get("chunkFullyReviewed") is True
        tier_raw = str(raw.get("captureQualityTier") or "MEDIUM")
        capture_quality_tier = tier_raw if tier_raw in {"HIGH", "MEDIUM", "LOW", "NONE"} else "MEDIUM"

        events: list[PerceptionEvent] = []
        raw_events = raw.get("events") if isinstance(raw.get("events"), list) else []
        for index, item in enumerate(raw_events):
            if not isinstance(item, dict):
                continue
            start_local = float(item.get("startMsLocal", float("nan")))
            end_local = float(item.get("endMsLocal", float("nan")))
            if start_local != start_local or end_local != end_local:
                continue
            start_local = _clamp(start_local, 0, chunk_duration_ms)
            end_local = _clamp(end_local, 0, chunk_duration_ms)
            if end_local <= start_local:
                continue
            kind = item.get("kind") if item.get("kind") in EVENT_KINDS else "other"
            duration_ms = max(1, round(end_local - start_local))
            event_id = item.get("eventId") if isinstance(item.get("eventId"), str) and item["eventId"] else f"e_{payload.chunk_id}_{index}"
            attrs = dict(item.get("attrs") or {}) if isinstance(item.get("attrs"), dict) else {}
            summary = item.get("summary") if isinstance(item.get("summary"), str) else None
            quality_caveat = (
                item.get("qualityCaveat") if isinstance(item.get("qualityCaveat"), str) else None
            )
            linked = [
                value
                for value in (item.get("linkedPriorEventIds") or [])
                if isinstance(value, str)
            ]
            confidence_raw = item.get("confidence")
            confidence = (
                max(0.0, min(1.0, float(confidence_raw)))
                if isinstance(confidence_raw, (int, float))
                else 0.5
            )
            normalized = ensure_speech_event_completeness(
                PerceptionEvent(
                    event_id=event_id,
                    kind=kind,  # type: ignore[arg-type]
                    start_ms_local=round(start_local),
                    end_ms_local=round(end_local),
                    duration_ms=duration_ms,
                    start_ms_session=payload.start_offset_ms + round(start_local),
                    end_ms_session=payload.start_offset_ms + round(end_local),
                    chunk_id=payload.chunk_id,
                    attrs=attrs,
                    summary=summary,
                    linked_prior_event_ids=linked,
                    confidence=confidence,
                    quality_caveat=quality_caveat,
                )
            )
            events.append(normalized)

        chunk_baseline_raw = raw.get("chunkBaseline")
        chunk_baseline = None
        if isinstance(chunk_baseline_raw, dict):
            from ..contracts.perception import PerceptionChunkBaseline

            chunk_baseline = PerceptionChunkBaseline.model_validate(
                {
                    "facePresentDominant": chunk_baseline_raw.get("facePresentDominant"),
                    "settingType": chunk_baseline_raw.get("settingType"),
                    "audioNotes": chunk_baseline_raw.get("audioNotes"),
                }
            )
        return PerceptionChunkResult(
            chunk_id=payload.chunk_id,
            chunk_duration_ms=chunk_duration_ms,
            chunk_fully_reviewed=fully,
            capture_quality_tier=capture_quality_tier,  # type: ignore[arg-type]
            chunk_baseline=chunk_baseline,
            events=events,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _unknown_observation_groups() -> dict[str, Any]:
    unknown: PerceptionTernary = "UNKNOWN"
    return {
        "identity": PerceptionIdentity(),
        "attention": PerceptionAttention(),
        "hands": PerceptionHands(),
        "objects": PerceptionObjects(),
        "people": PerceptionPeople(),
        "environment": PerceptionEnvironment(),
        "body": PerceptionBody(),
        "interaction": PerceptionInteraction(),
        "audio": PerceptionAudio(),
        "confidence": 0.5,
        "quality_caveat": None,
    }


def project_events_to_observations(
    result: PerceptionChunkResult,
    section_id: str | None,
) -> list[PerceptionObservation]:
    unknown = _unknown_observation_groups()

    def base_obs(start_ms: int, end_ms: int) -> PerceptionObservation:
        return PerceptionObservation(
            window_id=f"w_{start_ms}_{end_ms}",
            section_id=section_id,
            start_ms=start_ms,
            end_ms=end_ms,
            identity=unknown["identity"].model_copy(),
            attention=unknown["attention"].model_copy(),
            hands=unknown["hands"].model_copy(),
            objects=unknown["objects"].model_copy(),
            people=unknown["people"].model_copy(),
            environment=unknown["environment"].model_copy(),
            body=unknown["body"].model_copy(),
            interaction=unknown["interaction"].model_copy(),
            audio=unknown["audio"].model_copy(),
            confidence=0.5,
            quality_caveat=None,
            video_available=result.chunk_fully_reviewed,
            capture_quality_tier=result.capture_quality_tier,
        )

    if not result.events:
        if result.chunk_fully_reviewed:
            obs = base_obs(0, result.chunk_duration_ms)
            obs = obs.model_copy(update={"video_available": True, "confidence": 0.7})
            return [obs]
        return []

    observations: list[PerceptionObservation] = []
    for raw_event in result.events:
        event = ensure_speech_event_completeness(raw_event)
        obs = base_obs(event.start_ms_session, event.end_ms_session)
        obs = obs.model_copy(
            update={
                "confidence": event.confidence,
                "quality_caveat": event.quality_caveat,
                "video_available": True,
            }
        )
        attrs = event.attrs
        identity = obs.identity.model_copy(deep=True)
        attention = obs.attention.model_copy(deep=True)
        hands = obs.hands.model_copy(deep=True)
        objects = obs.objects.model_copy(deep=True)
        people = obs.people.model_copy(deep=True)
        environment = obs.environment.model_copy(deep=True)
        body = obs.body.model_copy(deep=True)
        interaction = obs.interaction.model_copy(deep=True)
        audio = obs.audio.model_copy(deep=True)

        def set_if(value: Any, setter: Any) -> None:
            if value is not None and value != "UNKNOWN":
                setter(value)

        set_if(attrs.get("gazeDirection"), lambda v: setattr(attention, "gaze_direction", v))
        set_if(attrs.get("gazeTarget"), lambda v: setattr(attention, "gaze_target", v))
        set_if(attrs.get("gazeStable"), lambda v: setattr(attention, "gaze_stable", v))
        set_if(attrs.get("repeatedGazePattern"), lambda v: setattr(attention, "repeated_gaze_pattern", v))
        set_if(attrs.get("attentionState"), lambda v: setattr(attention, "attention_state", v))
        set_if(attrs.get("facePresent"), lambda v: setattr(identity, "face_present", v))
        set_if(attrs.get("identityConsistent"), lambda v: setattr(identity, "identity_consistent", v))
        set_if(attrs.get("faceOccluded"), lambda v: setattr(identity, "face_occluded", v))
        set_if(attrs.get("faceOrientation"), lambda v: setattr(identity, "face_orientation", v))
        set_if(attrs.get("phoneVisible"), lambda v: setattr(objects, "phone_visible", v))
        set_if(attrs.get("notebookVisible"), lambda v: setattr(objects, "notebook_visible", v))
        set_if(attrs.get("paperVisible"), lambda v: setattr(objects, "paper_visible", v))
        set_if(attrs.get("calculatorVisible"), lambda v: setattr(objects, "calculator_visible", v))
        set_if(attrs.get("headphonesVisible"), lambda v: setattr(objects, "headphones_visible", v))
        set_if(attrs.get("headphonesLink"), lambda v: setattr(objects, "headphones_link", v))
        set_if(attrs.get("earphoneVisible"), lambda v: setattr(objects, "earphone_visible", v))
        set_if(attrs.get("earphoneLink"), lambda v: setattr(objects, "earphone_link", v))
        set_if(attrs.get("secondScreenVisible"), lambda v: setattr(objects, "second_screen_visible", v))
        set_if(attrs.get("unknownObjectVisible"), lambda v: setattr(objects, "unknown_object_visible", v))
        set_if(attrs.get("handsVisible"), lambda v: setattr(hands, "hands_visible", v))
        if attrs.get("handCount") is not None:
            hands.hand_count = attrs["handCount"]  # type: ignore[assignment]
        set_if(attrs.get("handLocation"), lambda v: setattr(hands, "hand_location", v))
        set_if(attrs.get("handActivity"), lambda v: setattr(hands, "hand_activity", v))
        set_if(attrs.get("objectInHand"), lambda v: setattr(hands, "object_in_hand", v))
        if attrs.get("objectInHand") == "phone":
            if hands.hands_visible == "UNKNOWN":
                hands.hands_visible = "yes"
            if objects.phone_visible == "UNKNOWN":
                objects.phone_visible = "yes"
        set_if(attrs.get("secondPersonVisible"), lambda v: setattr(people, "second_person_visible", v))
        set_if(attrs.get("secondPersonInteracting"), lambda v: setattr(people, "second_person_interacting", v))
        set_if(attrs.get("candidateRespondingToSecondPerson"), lambda v: setattr(people, "candidate_responding_to_second_person", v))
        set_if(attrs.get("secondPersonLooksLike"), lambda v: setattr(people, "second_person_looks_like", v))
        set_if(attrs.get("secondPersonRoleCue"), lambda v: setattr(people, "second_person_role_cue", v))
        set_if(attrs.get("secondPersonActivity"), lambda v: setattr(people, "second_person_activity", v))
        set_if(attrs.get("secondPersonPosition"), lambda v: setattr(people, "second_person_position", v))
        set_if(attrs.get("secondPersonObjectInHand"), lambda v: setattr(people, "second_person_object_in_hand", v))
        set_if(attrs.get("leftSeat"), lambda v: setattr(body, "left_seat", v))
        set_if(attrs.get("reachingOutsideFrame"), lambda v: setattr(interaction, "reaching_outside_frame", v))
        set_if(attrs.get("reachingDirection"), lambda v: setattr(interaction, "reaching_direction", v))
        set_if(attrs.get("settingType"), lambda v: setattr(environment, "setting_type", v))
        set_if(attrs.get("framing"), lambda v: setattr(environment, "framing", v))
        set_if(attrs.get("lightingCondition"), lambda v: setattr(environment, "lighting_condition", v))
        set_if(attrs.get("speechPresent"), lambda v: setattr(audio, "speech_present", v))
        set_if(attrs.get("speechContentClass"), lambda v: setattr(audio, "speech_content_class", v))
        if "conversationSummaryEn" in attrs:
            audio.conversation_summary_en = attrs.get("conversationSummaryEn")
        set_if(attrs.get("speechSource"), lambda v: setattr(audio, "speech_source", v))
        set_if(attrs.get("speechOverlapWithLips"), lambda v: setattr(audio, "speech_overlap_with_lips", v))
        set_if(attrs.get("speechLanguage"), lambda v: setattr(audio, "speech_language", v))
        set_if(attrs.get("speechStyle"), lambda v: setattr(audio, "speech_style", v))
        set_if(attrs.get("codeMixing"), lambda v: setattr(audio, "code_mixing", v))
        set_if(attrs.get("backgroundVoices"), lambda v: setattr(audio, "background_voices", v))
        set_if(attrs.get("deviceSounds"), lambda v: setattr(audio, "device_sounds", v))
        if "notablePhrasesOriginal" in attrs:
            audio.notable_phrases_original = attrs.get("notablePhrasesOriginal")

        if event.kind == "face_absent" and identity.face_present == "UNKNOWN":
            identity.face_present = "no"
        if event.kind == "leave_seat" and body.left_seat == "UNKNOWN":
            body.left_seat = "yes"
        if event.kind in {"second_person_present", "second_person_interaction"}:
            if people.second_person_visible == "UNKNOWN":
                people.second_person_visible = "yes"
            if event.kind == "second_person_interaction" and people.second_person_interacting == "UNKNOWN":
                people.second_person_interacting = "yes"
        if event.kind == "sustained_gaze" and attention.gaze_direction == "UNKNOWN":
            attention.gaze_direction = "away"
        if event.kind == "speech" and audio.speech_present == "UNKNOWN":
            audio.speech_present = "yes"
        if event.kind == "phone_visible" and objects.phone_visible == "UNKNOWN":
            objects.phone_visible = "yes"
        if event.kind == "phone_in_hand":
            if objects.phone_visible == "UNKNOWN":
                objects.phone_visible = "yes"
            if hands.hands_visible == "UNKNOWN":
                hands.hands_visible = "yes"
            if hands.object_in_hand == "UNKNOWN":
                hands.object_in_hand = "phone"
            if hands.hand_location == "UNKNOWN":
                hands.hand_location = "phone"
        if event.kind == "notes_visible":
            if objects.notebook_visible == "UNKNOWN" and objects.paper_visible == "UNKNOWN":
                objects.notebook_visible = "yes"
        if event.kind == "headphones_visible" and objects.headphones_visible == "UNKNOWN":
            objects.headphones_visible = "yes"
        if event.kind == "earphone_visible" and objects.earphone_visible == "UNKNOWN":
            objects.earphone_visible = "yes"
        if event.kind == "second_screen_visible" and objects.second_screen_visible == "UNKNOWN":
            objects.second_screen_visible = "yes"
        if event.kind == "reaching_outside_frame" and interaction.reaching_outside_frame == "UNKNOWN":
            interaction.reaching_outside_frame = "yes"

        observations.append(
            obs.model_copy(
                update={
                    "identity": identity,
                    "attention": attention,
                    "hands": hands,
                    "objects": objects,
                    "people": people,
                    "environment": environment,
                    "body": body,
                    "interaction": interaction,
                    "audio": audio,
                }
            )
        )
    return observations


def _ternary(value: Any) -> ScreenTernary:
    return value if value in {"yes", "no", "UNKNOWN"} else "UNKNOWN"


def _app_class(value: Any) -> ForegroundAppClass:
    allowed: set[ForegroundAppClass] = {
        "exam_ide",
        "browser",
        "notes",
        "ai_chat",
        "messaging",
        "os_desktop",
        "other",
        "UNKNOWN",
    }
    return value if value in allowed else "UNKNOWN"


def parse_screen_response(text: str, payload: PerceptionChunkJobPayload) -> ScreenObservation:
    try:
        raw = json.loads(_clean_json_text(text))
        if not isinstance(raw, dict):
            raise ValueError("invalid screen response")
        labels = [
            str(item)[:80]
            for item in (raw.get("externalResourceLabels") or [])
            if isinstance(item, str)
        ][:10]
        excerpt_raw = raw.get("pastedTextExcerpt")
        excerpt = excerpt_raw.strip()[:300] if isinstance(excerpt_raw, str) and excerpt_raw.strip() else None
        question_raw = raw.get("visibleQuestionRef")
        question_ref = (
            question_raw.strip()[:120]
            if isinstance(question_raw, str) and question_raw.strip()
            else None
        )
        confidence_raw = raw.get("confidence")
        confidence = (
            max(0.0, min(1.0, float(confidence_raw)))
            if isinstance(confidence_raw, (int, float))
            else 0.5
        )
        caveat_raw = raw.get("qualityCaveat")
        caveat = caveat_raw[:240] if isinstance(caveat_raw, str) else None
        return ScreenObservation(
            chunk_id=payload.chunk_id,
            sequence=payload.sequence,
            section_id=payload.section_id,
            start_ms=payload.start_offset_ms,
            end_ms=payload.end_offset_ms,
            exam_ui_visible=_ternary(raw.get("examUiVisible")),
            foreground_app_class=_app_class(raw.get("foregroundAppClass")),
            external_resource_labels=labels,
            ai_assistant_ui_visible=_ternary(raw.get("aiAssistantUiVisible")),
            secondary_workspace_visible=_ternary(raw.get("secondaryWorkspaceVisible")),
            fullscreen_exam_likely=_ternary(raw.get("fullscreenExamLikely")),
            paste_cue_visible=_ternary(raw.get("pasteCueVisible")),
            pasted_text_excerpt=excerpt,
            visible_question_ref=question_ref,
            confidence=confidence,
            quality_caveat=caveat,
            video_available=True,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return ScreenObservation(
            chunk_id=payload.chunk_id,
            sequence=payload.sequence,
            section_id=payload.section_id,
            start_ms=payload.start_offset_ms,
            end_ms=payload.end_offset_ms,
            exam_ui_visible="UNKNOWN",
            foreground_app_class="UNKNOWN",
            external_resource_labels=[],
            ai_assistant_ui_visible="UNKNOWN",
            secondary_workspace_visible="UNKNOWN",
            fullscreen_exam_likely="UNKNOWN",
            paste_cue_visible="UNKNOWN",
            pasted_text_excerpt=None,
            visible_question_ref=None,
            confidence=0.0,
            quality_caveat="Unparseable screen perception response",
            video_available=False,
        )


async def _gemini_text(
    deps: PipelineDeps,
    *,
    parts: list[dict[str, Any]],
    config: dict[str, Any],
    step: str,
    model_version: str,
    extra_meta: dict[str, Any],
) -> str | None:
    response = await generate_and_log(
        deps,
        step=step,
        model=DEFAULT_GEMINI_FLASH_MODEL,
        model_version=model_version,
        contents=[{"role": "user", "parts": parts}],
        config=config,
        extra_meta=extra_meta,
    )
    text = response.get("text")
    return text if isinstance(text, str) else None


async def analyze_camera_chunk(
    deps: PipelineDeps,
    payload: PerceptionChunkJobPayload,
) -> CachedPerceptionChunk:
    if payload.duration_ms < GEMINI_MIN_VIDEO_DURATION_MS:
        deps.logger.warning(
            "perception-chunk-job: chunk too short for Gemini — skipping",
            candidate_id=payload.candidate_id,
            chunk_id=payload.chunk_id,
            duration_ms=payload.duration_ms,
        )
        result = PerceptionChunkResult(
            chunk_id=payload.chunk_id,
            chunk_fully_reviewed=True,
            capture_quality_tier="HIGH",
            events=[],
        )
        return CachedPerceptionChunk(result=result, observations=[])

    cache_key = perception_chunk_cache_key(payload.chunk_id)
    cached = await deps.cache.get(cache_key)
    if isinstance(cached, dict):
        if cached.get("result"):
            loaded = CachedPerceptionChunk.model_validate(cached)
            if loaded.result.chunk_fully_reviewed:
                return loaded
        if isinstance(cached.get("text"), str):
            parsed = parse_events_response(cached["text"], payload)
            if parsed and parsed.chunk_fully_reviewed:
                return CachedPerceptionChunk(
                    result=parsed,
                    observations=project_events_to_observations(parsed, payload.section_id),
                )

    user = build_perception_chunk_user_prompt(
        PerceptionChunkUserPromptInput(
            chunk_id=payload.chunk_id,
            chunk_start_s=round(payload.start_offset_ms / 1000),
            chunk_duration_s=round(payload.duration_ms / 1000),
            section_type=payload.section_id,
        )
    )
    clip_label = (
        f"[VIDEO CLIP — session T+{round(payload.start_offset_ms / 1000)}s – "
        f"T+{round(payload.end_offset_ms / 1000)}s]"
    )
    parts, delivery = await resolve_media_parts(
        payload.signed_url,
        "video/webm",
        [user, clip_label],
    )
    deps.logger.info(
        "perception-chunk-job: resolved media parts (camera)",
        candidate_id=payload.candidate_id,
        chunk_id=payload.chunk_id,
        delivery=delivery,
    )

    flash_config = {
        "temperature": 0,
        "maxOutputTokens": PERCEPTION_MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "responseSchema": PERCEPTION_RESPONSE_SCHEMA,
        "systemInstruction": PERCEPTION_SYSTEM_PROMPT,
    }
    camera_meta = {"chunk_id": payload.chunk_id, "section_id": payload.section_id}
    text = await _gemini_text(
        deps,
        parts=parts,
        config=flash_config,
        step=STEP_CAMERA_PERCEPTION,
        model_version=PERCEPTION_PROMPT_VERSION,
        extra_meta=camera_meta,
    )
    parsed = parse_events_response(text, payload)
    if parsed is None:
        text = await _gemini_text(
            deps,
            parts=parts,
            config=flash_config,
            step=STEP_CAMERA_PERCEPTION,
            model_version=PERCEPTION_PROMPT_VERSION,
            extra_meta=camera_meta,
        )
        parsed = parse_events_response(text, payload)
    if parsed is None:
        deps.logger.warning(
            "perception-chunk-job: camera chunk unparseable — returning empty result",
            candidate_id=payload.candidate_id,
            chunk_id=payload.chunk_id,
        )
        result = PerceptionChunkResult(
            chunk_id=payload.chunk_id,
            chunk_fully_reviewed=False,
            capture_quality_tier="NONE",
            events=[],
        )
        return CachedPerceptionChunk(result=result, observations=[])

    if text:
        await deps.cache.set(cache_key, {"text": text})
    observations = project_events_to_observations(parsed, payload.section_id)
    observations = [
        obs.model_copy(
            update={
                "start_ms": obs.start_ms or payload.start_offset_ms,
                "end_ms": obs.end_ms or payload.end_offset_ms,
                "window_id": (
                    f"w_{payload.start_offset_ms}_{payload.end_offset_ms}"
                    if obs.window_id.startswith("w_0_") and not parsed.events
                    else obs.window_id
                ),
            }
        )
        for obs in observations
    ]
    return CachedPerceptionChunk(result=parsed, observations=observations)


async def analyze_screen_chunk(
    deps: PipelineDeps,
    payload: PerceptionChunkJobPayload,
) -> ScreenObservation:
    cache_key = screen_chunk_cache_key(payload.chunk_id)
    cached = await deps.cache.get(cache_key)
    if isinstance(cached, dict):
        if cached.get("chunkId") or cached.get("chunk_id"):
            return ScreenObservation.model_validate(cached)
        if isinstance(cached.get("text"), str):
            return parse_screen_response(cached["text"], payload)

    user = build_screen_perception_user_prompt(
        chunk_id=payload.chunk_id,
        section_id=payload.section_id,
        start_s=round(payload.start_offset_ms / 1000),
        end_s=round(payload.end_offset_ms / 1000),
        duration_s=round(payload.duration_ms / 1000),
    )
    parts, delivery = await resolve_media_parts(payload.signed_url, "video/webm", [user])
    deps.logger.info(
        "perception-chunk-job: resolved media parts (screen)",
        candidate_id=payload.candidate_id,
        chunk_id=payload.chunk_id,
        delivery=delivery,
    )
    flash_config = {
        "temperature": 0,
        "maxOutputTokens": PERCEPTION_MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "systemInstruction": SCREEN_PERCEPTION_SYSTEM_PROMPT,
    }
    text = await _gemini_text(
        deps,
        parts=parts,
        config=flash_config,
        step=STEP_SCREEN_PERCEPTION,
        model_version=SCREEN_PERCEPTION_PROMPT_VERSION,
        extra_meta={"chunk_id": payload.chunk_id, "section_id": payload.section_id},
    )
    observation = parse_screen_response(text or "", payload)
    if text:
        await deps.cache.set(cache_key, {"text": text})
    return observation


def _stamp_session_window(
    observations: list[PerceptionObservation],
    parsed: PerceptionChunkResult,
    payload: PerceptionChunkJobPayload,
) -> list[PerceptionObservation]:
    return [
        obs.model_copy(
            update={
                "start_ms": obs.start_ms or payload.start_offset_ms,
                "end_ms": obs.end_ms or payload.end_offset_ms,
                "window_id": (
                    f"w_{payload.start_offset_ms}_{payload.end_offset_ms}"
                    if obs.window_id.startswith("w_0_") and not parsed.events
                    else obs.window_id
                ),
            }
        )
        for obs in observations
    ]


@dataclass(frozen=True, slots=True)
class ScreenCameraChunkAnalysis:
    camera: CachedPerceptionChunk
    screen: ScreenObservation | None


def split_screen_camera_response(text: str | None) -> tuple[str | None, str | None]:
    cleaned = _clean_json_text(text)
    if not cleaned:
        return None, None
    try:
        raw = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, None
    if not isinstance(raw, dict):
        return None, None

    def half(key: str) -> str | None:
        value = raw.get(key)
        return json.dumps(value) if isinstance(value, dict) else None

    return half("camera"), half("screen")


async def analyze_screen_camera_chunk(
    deps: PipelineDeps,
    payload: PerceptionChunkJobPayload,
) -> ScreenCameraChunkAnalysis:
    camera_cache_key = perception_chunk_cache_key(payload.chunk_id)
    screen_cache_key = screen_chunk_cache_key(payload.chunk_id)

    if payload.duration_ms < GEMINI_MIN_VIDEO_DURATION_MS:
        deps.logger.warning(
            "perception-chunk-job: combined chunk too short for Gemini — skipping",
            candidate_id=payload.candidate_id,
            chunk_id=payload.chunk_id,
            duration_ms=payload.duration_ms,
        )
        result = PerceptionChunkResult(
            chunk_id=payload.chunk_id,
            chunk_fully_reviewed=True,
            capture_quality_tier="HIGH",
            events=[],
        )
        return ScreenCameraChunkAnalysis(
            camera=CachedPerceptionChunk(result=result, observations=[]),
            screen=None,
        )

    cached_camera = await _cached_camera_chunk(deps, camera_cache_key, payload)
    cached_screen = await _cached_screen_observation(deps, screen_cache_key, payload)
    if cached_camera is not None and cached_screen is not None:
        return ScreenCameraChunkAnalysis(camera=cached_camera, screen=cached_screen)

    camera_user_input = PerceptionChunkUserPromptInput(
        chunk_id=payload.chunk_id,
        chunk_start_s=round(payload.start_offset_ms / 1000),
        chunk_duration_s=round(payload.duration_ms / 1000),
        section_type=payload.section_id,
    )
    user = build_screen_camera_user_prompt(
        camera_user_input,
        section_id=payload.section_id,
        start_s=round(payload.start_offset_ms / 1000),
        end_s=round(payload.end_offset_ms / 1000),
        duration_s=round(payload.duration_ms / 1000),
    )
    clip_label = (
        f"[SCREEN+CAMERA CLIP — session T+{round(payload.start_offset_ms / 1000)}s – "
        f"T+{round(payload.end_offset_ms / 1000)}s]"
    )
    parts, delivery = await resolve_media_parts(
        payload.signed_url,
        "video/webm",
        [user, clip_label],
    )
    deps.logger.info(
        "perception-chunk-job: resolved media parts (screen+camera)",
        candidate_id=payload.candidate_id,
        chunk_id=payload.chunk_id,
        delivery=delivery,
    )

    flash_config = {
        "temperature": 0,
        "maxOutputTokens": SCREEN_CAMERA_MAX_OUTPUT_TOKENS,
        "responseMimeType": "application/json",
        "responseSchema": SCREEN_CAMERA_RESPONSE_SCHEMA,
        "systemInstruction": SCREEN_CAMERA_PERCEPTION_SYSTEM_PROMPT,
    }
    meta = {"chunk_id": payload.chunk_id, "section_id": payload.section_id}

    async def call() -> tuple[str | None, str | None, str | None]:
        raw_text = await _gemini_text(
            deps,
            parts=parts,
            config=flash_config,
            step=STEP_SCREEN_CAMERA_PERCEPTION,
            model_version=SCREEN_CAMERA_PERCEPTION_PROMPT_VERSION,
            extra_meta=meta,
        )
        camera_text, screen_text = split_screen_camera_response(raw_text)
        return raw_text, camera_text, screen_text

    _, camera_text, screen_text = await call()
    parsed = parse_events_response(camera_text, payload)
    if parsed is None:
        _, camera_text, retry_screen_text = await call()
        parsed = parse_events_response(camera_text, payload)
        if retry_screen_text is not None:
            screen_text = retry_screen_text

    screen_observation = parse_screen_response(screen_text or "", payload)

    if parsed is None:
        deps.logger.warning(
            "perception-chunk-job: combined chunk camera half unparseable — "
            "returning empty camera result",
            candidate_id=payload.candidate_id,
            chunk_id=payload.chunk_id,
        )
        result = PerceptionChunkResult(
            chunk_id=payload.chunk_id,
            chunk_fully_reviewed=False,
            capture_quality_tier="NONE",
            events=[],
        )
        return ScreenCameraChunkAnalysis(
            camera=CachedPerceptionChunk(result=result, observations=[]),
            screen=screen_observation,
        )

    observations = _stamp_session_window(
        project_events_to_observations(parsed, payload.section_id), parsed, payload
    )
    return ScreenCameraChunkAnalysis(
        camera=CachedPerceptionChunk(result=parsed, observations=observations),
        screen=screen_observation,
    )


async def _cached_camera_chunk(
    deps: PipelineDeps,
    cache_key: str,
    payload: PerceptionChunkJobPayload,
) -> CachedPerceptionChunk | None:
    cached = await deps.cache.get(cache_key)
    if not isinstance(cached, dict):
        return None
    if cached.get("result"):
        loaded = CachedPerceptionChunk.model_validate(cached)
        if loaded.result.chunk_fully_reviewed:
            return loaded
    if isinstance(cached.get("text"), str):
        parsed = parse_events_response(cached["text"], payload)
        if parsed and parsed.chunk_fully_reviewed:
            return CachedPerceptionChunk(
                result=parsed,
                observations=project_events_to_observations(parsed, payload.section_id),
            )
    return None


async def _cached_screen_observation(
    deps: PipelineDeps,
    cache_key: str,
    payload: PerceptionChunkJobPayload,
) -> ScreenObservation | None:
    cached = await deps.cache.get(cache_key)
    if not isinstance(cached, dict):
        return None
    if cached.get("chunkId") or cached.get("chunk_id"):
        return ScreenObservation.model_validate(cached)
    if isinstance(cached.get("text"), str):
        return parse_screen_response(cached["text"], payload)
    return None


async def process_perception_chunk_job(
    deps: PipelineDeps,
    payload: PerceptionChunkJobPayload,
) -> CachedPerceptionChunk | ScreenCameraChunkAnalysis | ScreenObservation | None:
    if payload.evidence_type == "screenCamera":
        if is_sidecar_chunk(payload.chunk_id):
            deps.logger.warning(
                "perception-chunk-job: sidecar chunk reached perception job — skipping",
                candidate_id=payload.candidate_id,
                chunk_id=payload.chunk_id,
            )
            return None
        analysis = await analyze_screen_camera_chunk(deps, payload)
        await deps.cache.set(
            perception_chunk_cache_key(payload.chunk_id),
            analysis.camera.model_dump(by_alias=True),
        )
        if analysis.screen is not None:
            await deps.cache.set(
                screen_chunk_cache_key(payload.chunk_id),
                analysis.screen.model_dump(by_alias=True),
            )
        return analysis

    if payload.evidence_type == "screenRecording":
        if is_sidecar_chunk(payload.chunk_id):
            deps.logger.warning(
                "perception-chunk-job: sidecar chunk reached perception job — skipping",
                candidate_id=payload.candidate_id,
                chunk_id=payload.chunk_id,
            )
            return None
        if payload.duration_ms < GEMINI_MIN_VIDEO_DURATION_MS:
            deps.logger.warning(
                "perception-chunk-job: screen chunk too short for Gemini — skipping",
                candidate_id=payload.candidate_id,
                chunk_id=payload.chunk_id,
                duration_ms=payload.duration_ms,
            )
            return None
        observation = await analyze_screen_chunk(deps, payload)
        await deps.cache.set(
            screen_chunk_cache_key(payload.chunk_id),
            observation.model_dump(by_alias=True),
        )
        return observation

    cached = await analyze_camera_chunk(deps, payload)
    await deps.cache.set(
        perception_chunk_cache_key(payload.chunk_id),
        cached.model_dump(by_alias=True),
    )
    return cached
