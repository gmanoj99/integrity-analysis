"""Camera and screen perception prompts (faithful TypeScript port)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .shared import PERCEPTION_PROMPT_VERSION

__all__ = [
    "PERCEPTION_PROMPT_VERSION",
    "PERCEPTION_SYSTEM_PROMPT",
    "PERCEPTION_RESPONSE_SCHEMA",
    "SCREEN_CAMERA_LAYOUT_RULE",
    "SCREEN_CAMERA_PERCEPTION_SYSTEM_PROMPT",
    "SCREEN_CAMERA_RESPONSE_SCHEMA",
    "SCREEN_PERCEPTION_SYSTEM_PROMPT",
    "SCREEN_RESPONSE_SCHEMA",
    "build_perception_chunk_user_prompt",
    "build_screen_camera_user_prompt",
    "build_screen_perception_user_prompt",
]

TERNARY = {"type": "string", "enum": ["yes", "no", "UNKNOWN"]}

PERCEPTION_SYSTEM_PROMPT = """You are a structured audiovisual observation system for an online exam integrity platform.

ROLE: Observe exactly what is visible AND audible in this video clip. Report ONLY observable facts.

STRICT RULES:
1. NEVER conclude cheating or risk — only describe what is visibly/audibly present
2. NEVER use evaluative words like "suspicious", "cheating", "violating", "copying"
3. "UNKNOWN" is always valid — return it whenever frame/audio quality prevents certainty
4. Each field is independent — do not let one observation bias another
5. Exam context: shared rooms, family nearby, fans/noise, varying lighting are all normal
6. Machine Facts in the prompt are deterministic platform events. Treat them as contextual information only. Never recreate them, never contradict them, never infer additional Machine Facts from the video.
7. Never assign risk. Never generate recommendations or verdicts of any kind.
8. When capture quality is LOW or NONE, prefer UNKNOWN for any field that cannot be clearly determined. Never invent visual detail that is not visible in the frame.
9. The hands.* group describes ONLY the exam candidate — the person whose face is most prominently facing the camera. If another person's hands are visible in the frame, ignore them entirely for all hands.* fields.
10. identity.identityConsistent: report "no" only when the face appears to be a genuinely different person than in recent prior context based on visible facial structure. Changes in lighting, angle, or expression alone are NOT enough — only report "no" for a clear person change.
11. people.secondPersonInteracting: "yes" means the second person is actively engaging with the candidate — speaking to them, handing them something, pointing at their screen, or physically approaching them. "no" means the second person is present but not engaging with the candidate.
12. people.candidateRespondingToSecondPerson: "yes" means the candidate has visibly turned toward, spoken to, or accepted something from the second person. "no" means the candidate remained focused on their own work despite the second person's presence.
13. AUDIO — Listen to the full audio track. Understand speech in English, Telugu, Hindi, Malayalam, Tamil, Marathi, Bengali, and common code-mix (Indic + English exam/code terms). Do NOT ignore or down-weight non-English speech. Do NOT treat speaking a non-English language as suspicious by itself.
14. audio.speechContentClass is a LANGUAGE-AGNOSTIC content taxonomy (what the talk was about), NOT a risk label. Use the same classes whether the utterance is in te/hi/ml/ta/mr/bn/en. Prefer a concrete class over "UNKNOWN". If speech is heard but words are not clearly intelligible, use "unclear" (not "UNKNOWN").
15. Whenever audio.speechPresent is "yes", you MUST fill audio.conversationSummaryEn with a faithful 2–4 sentence English paraphrase of what was said. Do not invent content. Do not use evaluative words. Example: "Other person told the candidate which loop condition to write." / "Candidate asked the invigilator how to fix a camera warning." If speech is present but unintelligible, still set speechContentClass="unclear" and conversationSummaryEn to an honest meta paraphrase such as "Speech present; words not clearly intelligible."
16. people.secondPersonLooksLike distinguishes live people from reflections, posters/photos, and on-screen video. people.secondPersonPosition and people.secondPersonRoleCue are soft observational cues only.
17. Murmuring or reciting MCQ option letters (A/B/C/D, "option C", "answer is B") in any language → speechContentClass "reciting_answer_choices". Whispering code logic / solution steps → "discussing_solution" or "receiving_dictation". Reading the question stem aloud without stating an answer choice → "reading_question". Invigilator/proctor logistics (camera, attendance, time left) → "invigilator_or_admin" or "technical_exam_help".
18. For kind=speech you MUST set attrs: speechPresent=yes, speechContentClass, conversationSummaryEn. Prefer speechSource, speechOverlapWithLips, speechLanguage. Quiet clip → events=[] (do not invent speech).

FIELD ENUMS — use EXACTLY these strings (case-sensitive) inside event attrs (and chunkBaseline where applicable):
identity.facePresent: "yes"|"no"|"UNKNOWN"
identity.identityConsistent: "yes"|"no"|"UNKNOWN"
identity.faceOccluded: "yes"|"no"|"UNKNOWN"
identity.faceOrientation: "toward_camera"|"turned_left"|"turned_right"|"down"|"up"|"UNKNOWN"
attention.gazeDirection: "screen"|"left"|"right"|"down"|"up"|"away"|"UNKNOWN"
attention.gazeStable: "yes"|"no"|"UNKNOWN"
attention.repeatedGazePattern: "none"|"left_recurring"|"right_recurring"|"down_recurring"|"UNKNOWN"
attention.attentionState: "focused"|"distracted"|"thinking"|"UNKNOWN"
attention.gazeTarget: "primary_screen"|"secondary_monitor"|"phone"|"other_person"|"off_screen_general"|"UNKNOWN"
hands.handsVisible: "yes"|"no"|"UNKNOWN"
hands.handCount: integer≥0 or "UNKNOWN"
hands.handLocation: "keyboard"|"desk"|"phone"|"below_frame"|"face"|"other"|"UNKNOWN"
hands.handActivity: "typing"|"writing"|"holding_object"|"idle"|"UNKNOWN"
hands.objectInHand: "phone"|"pen"|"paper"|"book"|"none"|"UNKNOWN"
objects.phoneVisible: "yes"|"no"|"UNKNOWN"
objects.notebookVisible: "yes"|"no"|"UNKNOWN"
objects.paperVisible: "yes"|"no"|"UNKNOWN"
objects.calculatorVisible: "yes"|"no"|"UNKNOWN"
objects.headphonesVisible: "yes"|"no"|"UNKNOWN"  (over-ear or on-ear wired/wireless headphones)
objects.headphonesLink: "wired"|"wireless"|"UNKNOWN"  (set when headphonesVisible=yes; else UNKNOWN)
objects.earphoneVisible: "yes"|"no"|"UNKNOWN"  (small in-ear wired or wireless earbuds/earpiece — distinct from headphones)
objects.earphoneLink: "wired"|"wireless"|"UNKNOWN"  (set when earphoneVisible=yes; else UNKNOWN)
objects.secondScreenVisible: "yes"|"no"|"UNKNOWN"  (a secondary monitor or screen in frame, separate from the primary exam screen)
objects.unknownObjectVisible: "yes"|"no"|"UNKNOWN"
people.secondPersonVisible: "yes"|"no"|"UNKNOWN"
people.secondPersonInteracting: "yes"|"no"|"UNKNOWN"
people.secondPersonObjectInHand: "phone"|"paper"|"none"|"UNKNOWN"
people.secondPersonActivity: "using_phone"|"reading"|"writing"|"speaking_to_candidate"|"passing_by"|"idle"|"UNKNOWN"
people.candidateRespondingToSecondPerson: "yes"|"no"|"UNKNOWN"
people.secondPersonPosition: "background"|"adjacent"|"behind"|"leaning_in"|"UNKNOWN"
people.secondPersonLooksLike: "live_person"|"reflection"|"poster_or_photo"|"on_screen_video"|"UNKNOWN"
people.secondPersonRoleCue: "unknown"|"peer_helper"|"household"|"invigilator_or_staff"|"passerby"|"UNKNOWN"
environment.lightingCondition: "adequate"|"dim"|"bright"|"backlit"|"UNKNOWN"
environment.framing: "full_face"|"partial"|"obscured"|"off_center"|"UNKNOWN"
environment.settingType: "private_room"|"shared_space"|"exam_hall"|"UNKNOWN"
body.leftSeat: "yes"|"no"|"UNKNOWN"
interaction.reachingOutsideFrame: "yes"|"no"|"UNKNOWN"
interaction.reachingDirection: "down"|"left"|"right"|"forward"|"none"|"UNKNOWN"
audio.speechPresent: "yes"|"no"|"UNKNOWN"
audio.speechSource: "candidate"|"other_person_in_room"|"off_camera"|"device_playback"|"mixed"|"UNKNOWN"
audio.speechOverlapWithLips: "yes"|"no"|"UNKNOWN"
audio.speechStyle: "normal"|"whisper"|"raised"|"reading_aloud"|"UNKNOWN"
audio.speechLanguage: "en"|"te"|"hi"|"ml"|"ta"|"mr"|"bn"|"mixed"|"other"|"UNKNOWN"
audio.codeMixing: "yes"|"no"|"UNKNOWN"
audio.backgroundVoices: "none"|"one"|"multiple"|"UNKNOWN"
audio.deviceSounds: "none"|"notification"|"call_ringtone"|"keyboard_other"|"UNKNOWN"
audio.speechContentClass: "silence"|"self_talk_or_thinking"|"reading_question"|"asking_for_answer"|"receiving_dictation"|"discussing_solution"|"reciting_answer_choices"|"invigilator_or_admin"|"technical_exam_help"|"casual_non_exam"|"unclear"|"UNKNOWN"
audio.conversationSummaryEn: null when no speech; otherwise a 2–4 sentence English paraphrase (required when speechPresent is "yes")
audio.notablePhrasesOriginal: null or a short array of native-script/transliterated key phrases

TIMED EVENTS:
Emit events[] with exact startMsLocal/endMsLocal/durationMs for every notable behavioural span. Do not bucket into fixed windows. Put observation fields into attrs using the FIELD ENUMS above. Quiet clip → events=[] with chunkFullyReviewed=true.

PER-KIND ATTRS (MUST set when emitting that kind):
- phone_in_hand: phoneVisible=yes, objectInHand=phone, handsVisible=yes; prefer handLocation=phone; optional handActivity=holding_object
- phone_visible: phoneVisible=yes; set objectInHand if held
- sustained_gaze: gazeDirection + gazeTarget; prefer gazeStable, repeatedGazePattern, attentionState
- speech: speechPresent=yes, speechContentClass (use unclear if unintelligible — not UNKNOWN), conversationSummaryEn (required; meta paraphrase if unintelligible); prefer speechSource, speechLanguage, speechOverlapWithLips
- second_person_present: secondPersonVisible=yes + secondPersonLooksLike/position/roleCue when known
- second_person_interaction: secondPersonInteracting=yes + candidateRespondingToSecondPerson/activity/objectInHand + looksLike when known
- notes_visible: notebookVisible and/or paperVisible=yes
- headphones_visible: headphonesVisible=yes; prefer headphonesLink=wired|wireless when known
- earphone_visible: earphoneVisible=yes; prefer earphoneLink=wired|wireless when known
- second_screen_visible: secondScreenVisible=yes
- face_absent: facePresent=no
- leave_seat: leftSeat=yes
- reaching_outside_frame: reachingOutsideFrame=yes + reachingDirection when known"""

PERCEPTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "chunkId",
        "chunkDurationMs",
        "chunkFullyReviewed",
        "captureQualityTier",
        "events",
    ],
    "properties": {
        "chunkId": {"type": "string"},
        "chunkDurationMs": {"type": "number"},
        "chunkFullyReviewed": {"type": "boolean"},
        "captureQualityTier": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW", "NONE"],
        },
        "chunkBaseline": {
            "type": "object",
            "properties": {
                "facePresentDominant": TERNARY,
                "settingType": {
                    "type": "string",
                    "enum": ["private_room", "shared_space", "exam_hall", "UNKNOWN"],
                },
                "audioNotes": {"type": "string", "nullable": True},
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "eventId",
                    "kind",
                    "startMsLocal",
                    "endMsLocal",
                    "durationMs",
                    "confidence",
                    "attrs",
                ],
                "properties": {
                    "eventId": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
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
                        ],
                    },
                    "startMsLocal": {"type": "number"},
                    "endMsLocal": {"type": "number"},
                    "durationMs": {"type": "number"},
                    "attrs": {
                        "type": "object",
                        "required": ["conversationSummaryEn", "speechContentClass"],
                        "properties": {
                            "gazeDirection": {
                                "type": "string",
                                "enum": [
                                    "screen",
                                    "left",
                                    "right",
                                    "down",
                                    "up",
                                    "away",
                                    "UNKNOWN",
                                ],
                            },
                            "gazeTarget": {
                                "type": "string",
                                "enum": [
                                    "primary_screen",
                                    "secondary_monitor",
                                    "phone",
                                    "other_person",
                                    "off_screen_general",
                                    "UNKNOWN",
                                ],
                            },
                            "gazeStable": TERNARY,
                            "repeatedGazePattern": {
                                "type": "string",
                                "enum": [
                                    "none",
                                    "left_recurring",
                                    "right_recurring",
                                    "down_recurring",
                                    "UNKNOWN",
                                ],
                            },
                            "attentionState": {
                                "type": "string",
                                "enum": ["focused", "distracted", "thinking", "UNKNOWN"],
                            },
                            "facePresent": TERNARY,
                            "identityConsistent": TERNARY,
                            "faceOccluded": TERNARY,
                            "faceOrientation": {
                                "type": "string",
                                "enum": [
                                    "toward_camera",
                                    "turned_left",
                                    "turned_right",
                                    "down",
                                    "up",
                                    "UNKNOWN",
                                ],
                            },
                            "phoneVisible": TERNARY,
                            "notebookVisible": TERNARY,
                            "paperVisible": TERNARY,
                            "calculatorVisible": TERNARY,
                            "headphonesVisible": TERNARY,
                            "headphonesLink": {
                                "type": "string",
                                "enum": ["wired", "wireless", "UNKNOWN"],
                            },
                            "earphoneVisible": TERNARY,
                            "earphoneLink": {
                                "type": "string",
                                "enum": ["wired", "wireless", "UNKNOWN"],
                            },
                            "secondScreenVisible": TERNARY,
                            "unknownObjectVisible": TERNARY,
                            "handsVisible": TERNARY,
                            "handCount": {},
                            "handLocation": {
                                "type": "string",
                                "enum": [
                                    "keyboard",
                                    "desk",
                                    "phone",
                                    "below_frame",
                                    "face",
                                    "other",
                                    "UNKNOWN",
                                ],
                            },
                            "handActivity": {
                                "type": "string",
                                "enum": [
                                    "typing",
                                    "writing",
                                    "holding_object",
                                    "idle",
                                    "UNKNOWN",
                                ],
                            },
                            "objectInHand": {
                                "type": "string",
                                "enum": [
                                    "phone",
                                    "pen",
                                    "paper",
                                    "book",
                                    "none",
                                    "UNKNOWN",
                                ],
                            },
                            "secondPersonVisible": TERNARY,
                            "secondPersonInteracting": TERNARY,
                            "candidateRespondingToSecondPerson": TERNARY,
                            "secondPersonLooksLike": {
                                "type": "string",
                                "enum": [
                                    "live_person",
                                    "reflection",
                                    "poster_or_photo",
                                    "on_screen_video",
                                    "UNKNOWN",
                                ],
                            },
                            "secondPersonRoleCue": {
                                "type": "string",
                                "enum": [
                                    "unknown",
                                    "peer_helper",
                                    "household",
                                    "invigilator_or_staff",
                                    "passerby",
                                    "UNKNOWN",
                                ],
                            },
                            "secondPersonActivity": {
                                "type": "string",
                                "enum": [
                                    "using_phone",
                                    "reading",
                                    "writing",
                                    "speaking_to_candidate",
                                    "passing_by",
                                    "idle",
                                    "UNKNOWN",
                                ],
                            },
                            "secondPersonPosition": {
                                "type": "string",
                                "enum": [
                                    "background",
                                    "adjacent",
                                    "behind",
                                    "leaning_in",
                                    "UNKNOWN",
                                ],
                            },
                            "secondPersonObjectInHand": {
                                "type": "string",
                                "enum": ["phone", "paper", "none", "UNKNOWN"],
                            },
                            "speechPresent": TERNARY,
                            "speechContentClass": {
                                "type": "string",
                                "enum": [
                                    "silence",
                                    "self_talk_or_thinking",
                                    "reading_question",
                                    "asking_for_answer",
                                    "receiving_dictation",
                                    "discussing_solution",
                                    "reciting_answer_choices",
                                    "invigilator_or_admin",
                                    "technical_exam_help",
                                    "casual_non_exam",
                                    "unclear",
                                    "UNKNOWN",
                                ],
                            },
                            "conversationSummaryEn": {"type": "string", "nullable": True},
                            "speechSource": {
                                "type": "string",
                                "enum": [
                                    "candidate",
                                    "other_person_in_room",
                                    "off_camera",
                                    "device_playback",
                                    "mixed",
                                    "UNKNOWN",
                                ],
                            },
                            "speechOverlapWithLips": TERNARY,
                            "speechLanguage": {
                                "type": "string",
                                "enum": [
                                    "en",
                                    "te",
                                    "hi",
                                    "ml",
                                    "ta",
                                    "mr",
                                    "bn",
                                    "mixed",
                                    "other",
                                    "UNKNOWN",
                                ],
                            },
                            "speechStyle": {
                                "type": "string",
                                "enum": [
                                    "normal",
                                    "whisper",
                                    "raised",
                                    "reading_aloud",
                                    "UNKNOWN",
                                ],
                            },
                            "codeMixing": TERNARY,
                            "backgroundVoices": {
                                "type": "string",
                                "enum": ["none", "one", "multiple", "UNKNOWN"],
                            },
                            "deviceSounds": {
                                "type": "string",
                                "enum": [
                                    "none",
                                    "notification",
                                    "call_ringtone",
                                    "keyboard_other",
                                    "UNKNOWN",
                                ],
                            },
                            "notablePhrasesOriginal": {
                                "type": "array",
                                "items": {"type": "string"},
                                "nullable": True,
                            },
                            "leftSeat": TERNARY,
                            "reachingOutsideFrame": TERNARY,
                            "reachingDirection": {
                                "type": "string",
                                "enum": [
                                    "down",
                                    "left",
                                    "right",
                                    "forward",
                                    "none",
                                    "UNKNOWN",
                                ],
                            },
                            "settingType": {
                                "type": "string",
                                "enum": [
                                    "private_room",
                                    "shared_space",
                                    "exam_hall",
                                    "UNKNOWN",
                                ],
                            },
                            "framing": {
                                "type": "string",
                                "enum": [
                                    "full_face",
                                    "partial",
                                    "obscured",
                                    "off_center",
                                    "UNKNOWN",
                                ],
                            },
                            "lightingCondition": {
                                "type": "string",
                                "enum": [
                                    "adequate",
                                    "dim",
                                    "bright",
                                    "backlit",
                                    "UNKNOWN",
                                ],
                            },
                        },
                    },
                    "summary": {"type": "string", "nullable": True},
                    "linkedPriorEventIds": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                    "qualityCaveat": {"type": "string", "nullable": True},
                },
            },
        },
    },
}

SCREEN_PERCEPTION_SYSTEM_PROMPT = """You are a structured screen-recording observation system for an online exam integrity platform.

Observe what is VISIBLE on the candidate's screen. Report ONLY observable facts. Never conclude cheating or risk. "UNKNOWN" when blank/blurred/unreadable. Do NOT describe faces, gaze, or people.

When pasteCueVisible=yes, extract pastedTextExcerpt (≤300 chars) or null if unreadable.
When a question number/title is visible, set visibleQuestionRef; else null.
externalResourceLabels: recognizable non-exam sites/apps; empty array if none."""

SCREEN_CAMERA_LAYOUT_RULE = """LAYOUT — THIS CLIP CARRIES BOTH SIGNALS IN ONE FILE.

This is a screen recording with the candidate's webcam composited into it as a
picture-in-picture inset (a small rectangle in one corner). The inset may be
blank, black, frozen or absent — that is normal, and it means the camera half
is UNKNOWN, not that anything is wrong.

Produce BOTH halves of the response from this one clip:
- The "camera" half describes ONLY what is inside the webcam inset: the person,
  their hands, objects they hold, people and movement behind them — plus the
  clip's audio, which belongs entirely to the camera half.
- The "screen" half describes ONLY the screen content OUTSIDE the inset: the
  exam UI, applications, browser tabs, editors and any visible text.

Keep the halves strictly independent:
- Never let one half inform the other. Do not infer screen activity from what
  the person appears to be doing, and do not infer the person's attention,
  gaze or behaviour from what is on the screen.
- The inset is NOT a second screen: never report it as objects.secondScreenVisible
  or as screen.secondaryWorkspaceVisible.
- A person visible on the screen itself (a video call, a recorded lecture) is
  screen content, not a second person in the room. Report people only from the
  inset, and use people.secondPersonLooksLike="on_screen_video" if the inset
  itself shows one.
- attention.gazeDirection describes where the person in the inset is looking
  relative to their own camera, exactly as for a standalone webcam clip.
- If the inset is blank or absent, return camera events=[] with
  captureQualityTier="NONE" and still fill the screen half normally.

Return one JSON object with exactly two keys: "camera" and "screen"."""

SCREEN_CAMERA_PERCEPTION_SYSTEM_PROMPT = (
    f"{PERCEPTION_SYSTEM_PROMPT}\n\n"
    f"{SCREEN_PERCEPTION_SYSTEM_PROMPT}\n\n"
    f"{SCREEN_CAMERA_LAYOUT_RULE}"
)

# The fields ``parse_screen_response`` reads, nothing more: the screen half of
# the combined response is fed to that parser unchanged.
SCREEN_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "examUiVisible",
        "foregroundAppClass",
        "externalResourceLabels",
        "aiAssistantUiVisible",
        "secondaryWorkspaceVisible",
        "fullscreenExamLikely",
        "pasteCueVisible",
        "confidence",
    ],
    "properties": {
        "examUiVisible": TERNARY,
        "foregroundAppClass": {
            "type": "string",
            "enum": [
                "exam_ide",
                "browser",
                "notes",
                "ai_chat",
                "messaging",
                "os_desktop",
                "other",
                "UNKNOWN",
            ],
        },
        "externalResourceLabels": {"type": "array", "items": {"type": "string"}},
        "aiAssistantUiVisible": TERNARY,
        "secondaryWorkspaceVisible": TERNARY,
        "fullscreenExamLikely": TERNARY,
        "pasteCueVisible": TERNARY,
        "pastedTextExcerpt": {"type": "string", "nullable": True},
        "visibleQuestionRef": {"type": "string", "nullable": True},
        "confidence": {"type": "number"},
        "qualityCaveat": {"type": "string", "nullable": True},
    },
}

# One response, two halves — each half is exactly the schema its existing
# parser already understands, so neither parser has to change.
SCREEN_CAMERA_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["camera", "screen"],
    "properties": {
        "camera": PERCEPTION_RESPONSE_SCHEMA,
        "screen": SCREEN_RESPONSE_SCHEMA,
    },
}


PER_KIND_ATTRS_CHECKLIST = """Per-kind attrs checklist (MUST):
- phone_in_hand → phoneVisible=yes, objectInHand=phone, handsVisible=yes; prefer handLocation=phone
- phone_visible → phoneVisible=yes; objectInHand if held
- sustained_gaze → gazeDirection + gazeTarget
- speech → speechPresent=yes, speechContentClass (unclear if unintelligible), conversationSummaryEn (required)
- second_person_present → secondPersonVisible=yes + looksLike/position/roleCue
- second_person_interaction → interacting=yes + responding/activity/objectInHand
- notes_visible → notebookVisible and/or paperVisible=yes
- headphones_visible → headphonesVisible=yes + headphonesLink when known
- earphone_visible → earphoneVisible=yes + earphoneLink when known
- second_screen_visible → secondScreenVisible=yes
- face_absent → facePresent=no
- leave_seat → leftSeat=yes
- reaching_outside_frame → reachingOutsideFrame=yes + reachingDirection when known"""


@dataclass(frozen=True, slots=True)
class PerceptionChunkUserPromptInput:
    chunk_id: str
    chunk_start_s: int
    chunk_duration_s: int
    phase: str | None = None
    section_type: str | None = None
    machine_fact_hints: list[dict[str, Any]] | None = None
    open_event_tail: list[dict[str, Any]] | None = None
    prior_context: dict[str, str] | None = None


def build_perception_chunk_user_prompt(input: PerceptionChunkUserPromptInput) -> str:
    facts = (
        "\n".join(f"  - {item['kind']} at ~{item['atS']}s" for item in input.machine_fact_hints)
        if input.machine_fact_hints
        else "  (none)"
    )
    open_tail = (
        "\n".join(
            f"  - {item['eventId']} kind={item['kind']} ended_local={item['endMsLocal']}ms "
            f"summary={item.get('summary') or ''}"
            for item in input.open_event_tail
        )
        if input.open_event_tail
        else "  (none)"
    )
    prior = (
        (
            f"face={input.prior_context['facePresent']}, "
            f"framing={input.prior_context['framing']}, "
            f"object_in_hand={input.prior_context['objectInHand']}, "
            f"setting={input.prior_context['settingType']}"
        )
        if input.prior_context
        else "(none — first chunk or no prior)"
    )
    return f"""Analyze this entire webcam clip in one pass. Watch AND listen.
chunkId: "{input.chunk_id}"
Session offset: {input.chunk_start_s}s – {input.chunk_start_s + input.chunk_duration_s}s
Clip-local time: 0s – {input.chunk_duration_s}s ({input.chunk_duration_s * 1000} ms)
Phase: {input.phase or "unknown"} | Section: {input.section_type or "none"}

Machine fact hints (context only — do not recreate or contradict):
{facts}

Open events from previous chunk (continue or close with absolute local times if still ongoing):
{open_tail}

Prior visual context: {prior}

{PER_KIND_ATTRS_CHECKLIST}

Return JSON matching the schema: chunkFullyReviewed, captureQualityTier, chunkBaseline, and events[] with exact startMsLocal/endMsLocal/durationMs and attrs using FIELD ENUMS. Quiet clip → events=[]."""


def build_screen_camera_user_prompt(
    camera_input: PerceptionChunkUserPromptInput,
    *,
    section_id: str | None,
    start_s: int,
    end_s: int,
    duration_s: int,
) -> str:
    """Both existing user prompts under one instruction, for the single call.

    Each half is the prompt its own analyser already sends, so the model sees
    the same task description it does today — only the framing that the two
    views arrive in one clip is new.
    """

    return f"""This clip is a screen recording with a picture-in-picture webcam inset. \
Analyze it ONCE and return both halves.

=== CAMERA HALF (the webcam inset only) ===
{build_perception_chunk_user_prompt(camera_input)}

=== SCREEN HALF (the screen content outside the inset only) ===
{build_screen_perception_user_prompt(
    chunk_id=camera_input.chunk_id,
    section_id=section_id,
    start_s=start_s,
    end_s=end_s,
    duration_s=duration_s,
)}

Return one JSON object: {{"camera": <camera half>, "screen": <screen half>}}. \
Emit both keys even when one half is entirely UNKNOWN."""


def build_screen_perception_user_prompt(
    *,
    chunk_id: str,
    section_id: str | None,
    start_s: int,
    end_s: int,
    duration_s: int,
) -> str:
    return f"""Analyze this screen-recording chunk.
chunkId: {chunk_id}
sectionId: {section_id or "unknown"}
session window: T+{start_s}s – T+{end_s}s
duration: {duration_s}s

Return one JSON object: examUiVisible, foregroundAppClass, externalResourceLabels, aiAssistantUiVisible, secondaryWorkspaceVisible, fullscreenExamLikely, pasteCueVisible, pastedTextExcerpt, visibleQuestionRef, confidence, qualityCaveat."""
