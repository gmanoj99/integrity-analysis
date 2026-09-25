# Local testing — v3 combined screen+camera chunks

Temporary harness for running the real AI flow on one candidate's recording
from raw signed S3 URLs. No SQS, S3, ECS, IAM or presigning involved.

Everything lives in three places:

| Path                    | What it is                                              |
| ----------------------- | ------------------------------------------------------- |
| `run_review.py`         | the runner                                              |
| `input/input.json`      | `{"chunk_urls": [...]}` — the signed URLs, used verbatim |
| `input/media/`          | local copies of the chunks (survives URL expiry)        |
| `output/`               | every artifact the run produces                         |

## Why it goes through the worker contract

`run_review.py` does **not** hand-build a `ManifestSet`. It synthesizes the
payload the Django backend would have staged (`StagedReviewPayload`) and runs it
through `worker/contracts.build_review_request`, so the chunk ids, manifest
grouping, exam-mode resolution and the `screenCamera` perception path are byte
for byte the ones the ECS worker takes. A bug you find locally is a real bug.

## The three commands

```bash
# 0. cache the media while the signed URLs are still valid (they expire in 1h)
.venv/bin/python run_review.py --download --dry-run

# 1. ingest only — no Gemini call, no API key, no cost
.venv/bin/python run_review.py --dry-run

# 2. the whole flow with scripted Gemini responses — proves the wiring, free
.venv/bin/python run_review.py --fake-gemini

# 3. the real AI flow
GEMINI_API_KEY=... .venv/bin/python run_review.py

# ...naming the bundle
GEMINI_API_KEY=... .venv/bin/python run_review.py --bundle-name evidence_bundle1.json
```

`--bundle-name` only renames the final bundle; the other artifacts keep their
names and are overwritten each run.

Once `--download` has run, every later command reads from `input/media/` and
works after the URLs have expired. To test a different attempt: replace
`input/input.json`, re-run `--download`.

## How the URLs are interpreted

Each URL is classified by its **S3 prefix and filename** — the same things the
backend's own manifest builder keys off — and the `{epoch13}[__{duration}]` stem
is the contract the whole pipeline depends on (`timeline.parse_chunk_id`):

| URL shape | `mediaType` | Notes |
| --- | --- | --- |
| `user_camera_recordings/…/{epoch}__{dur}.webm` | `CAMERA_VIDEO` | pure camera |
| `user_session_recordings/…/v3/{epoch}__{dur}.webm` | `SCREEN_CAMERA_VIDEO` | combined screen+camera |
| `user_session_recordings/…/{epoch}__{dur}.webm` | `SCREEN_VIDEO` | screen only |
| `user_session_recordings/…/{epoch}.json[.gz]` | `RRWEB_EVENT` | rrweb DOM batch |
| `…__metadata.json.gz` | — | read only to cross-check the stem against the recorder's timestamps |
| `…__events.json.gz` | — | decoded and reported, but **not** fed to the pipeline (see below) |

`--media-type` forces one type for every chunk if the auto-classification is
ever wrong.

Mixing is expected and handled: put an attempt's camera **and** rrweb URLs in
one input file and the review gets both modalities.
`worker/contracts._resolve_exam_mode` then picks `rrweb` or `screen` by chunk
count — camera chunks live in their own manifest and never compete.

**rrweb durations.** An rrweb filename carries only the upload epoch, so the
runner reads the batch's own first/last DOM `timestamp` from the cached file and
supplies a real `durationMs`, exactly as the backend manifest does. Without it
the timeline falls back to a flat `KEYSTROKE_FALLBACK_DURATION_MS` per batch.
This means **`--download` before a real rrweb run**, or the spans are guesses.

## Running more than one attempt

`--run-name` prefixes every output file so attempts never overwrite each other,
and `--cache-dir` keeps their media apart:

```bash
.venv/bin/python run_review.py \
  --input input/input2.json --cache-dir input/media2 \
  --run-name attempt2 --section-id coding --section-type coding \
  --bundle-name evidence_bundle2.json
```

Identity (`attemptUserId` / `orgAssessId` / `examAttemptId`) is recovered from
the three UUIDs in the key, which is exactly the sidecar's
`uploaderPathIdentifier`. The session timeline is synthesized as a single
section spanning the chunks; override it by adding `candidateId`,
`assessmentId`, `sections` or `activityLogs` keys to `input/input.json`, or use
`--section-id` / `--section-type`.

## Media delivery — the one thing that differs from production

`media/perception_media.resolve_media_parts` probes the object for EBML magic
and, when it really is a WebM, hands Gemini the **presigned URL** as
`fileData.fileUri`. The Gemini *developer* API (`genai.Client(api_key=...)`,
which is what `GoogleGeminiClient` uses) only resolves a `fileUri` from its own
Files API or YouTube, so a raw S3 URL is not fetchable there.

So the runner defaults to `--media-delivery inline`, which base64s the clip into
the request. At ~2.8 MB per 60s chunk that is comfortably inside the request
limit.

| `--media-delivery` | Behaviour                                        |
| ------------------ | ------------------------------------------------ |
| `inline` (default) | clip bytes base64'd into the request             |
| `files`            | upload to the Gemini Files API, use that `fileUri` |
| `auto`             | the shipped code path, unmodified — use this to check whether the presigned-`fileUri` route actually works |

Worth resolving before this ships: if `auto` fails against the real API, the
production worker has the same problem, and `files` is the closest fix.

## Attempts tested so far

| Run | Input | Evidence | Mode | Bundle |
| --- | --- | --- | --- | --- |
| 1 | `input/input.json` | 5 combined | `screen` | `output/evidence_bundle1.json` |
| 2 | `input/input2.json` | 13 camera + 5 rrweb | `rrweb` | `output/attempt2_evidence_bundle2.json` |
| 3 | `input/input3.json` | 17 combined | `screen` | `output/attempt3_evidence_bundle3.json` |

## Output artifacts

| File                          | Use it to check                                    |
| ----------------------------- | -------------------------------------------------- |
| `staged_payload.json`         | the manifest the backend would need to produce     |
| `review_request.json`         | the normalized request, with the derived chunk ids |
| `ingest_summary.json`         | exam mode, chunk counts, drops, warnings, and every timeline span |
| `sidecar_metadata.json`       | `stemMatchesMetadata` per chunk — filename vs recorder truth |
| `client_reported_events.json` | the proctoring events, with their session offsets  |
| `perception_cache.json`       | per-chunk Gemini output; written even if deliberation fails |
| `evidence_bundle.json`        | the final bundle                                   |

## Two findings from the current data

1. **`__events.json.gz` never reaches the pipeline.** It carries the client's own
   proctoring events, not rrweb DOM events:

   ```json
   {"identifier": "MULTIPLE_FACES_DETECTED_POPUP_APPEARED",
    "absoluteTimestamp": 1788864818706,
    "payload": {"message": "Multiple faces detected in camera."}}
   ```

   `run_behavioral_analysis` has a `client_reported_events` parameter built for
   exactly this shape (`machine_facts.contracts.ClientReportedEvent`), but
   `run_integrity_review` never passes it, and no `mediaType` in
   `worker/contracts._MEDIA_TYPE_TO_EVIDENCE_TYPE` maps to it. The runner
   decodes it into `output/client_reported_events.json` so it is at least
   visible. That event sits at session offset **184 509 ms**, inside chunk 4
   (180 207–240 210 ms) — a free ground truth for whether the camera half of the
   combined analysis spots a second person there.

2. **The camera manifest is empty and that is correct.** With
   `SCREEN_CAMERA_VIDEO` chunks there are no separate `CAMERA_VIDEO` entries;
   the camera observations come out of the combined analysis. Each chunk
   produces two media-index entries (`video` + `screen`) off the one file, which
   is what the frontend uses to pick the right segment.
