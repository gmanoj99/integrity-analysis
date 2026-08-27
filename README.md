# Integrity Review Pipeline

Standalone Python 3.11+ port of the camera, screen, and rrweb integrity-analysis
path. It has no Django, database, queue, cohort, coding-skill, plagiarism, or
physical clip-extraction dependency.

## Local development environment

Create and install the isolated environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

`run_integrity_review(request, deps)` in `integrity_review_pipeline/pipeline.py`
is the pure entrypoint: it takes a `ReviewRequest` and injected `PipelineDeps`
(object store, Gemini client, cache, limiter, logger, media URI provider, and
`organization_id`) and returns a validated `EvidenceBundle`. See
`tests/test_pipeline.py` for a full in-memory, dependency-injected example.

## ECS worker

`integrity_review_pipeline/worker/` is the production entrypoint: an SQS-driven
Fargate worker that stages requests/results in S3, runs `run_integrity_review`,
and publishes `VIDEO_ANALYSIS_RESPONSE` messages. Configuration is entirely via
environment variables (see `worker/config.py`). Build and run it locally with:

```bash
docker build -t integrity-review-worker .
docker run --rm \
  -e AWS_STORAGE_BUCKET_NAME=... \
  -e REQUEST_QUEUE_URL=... -e RESULT_QUEUE_URL=... \
  -e AWS_REGION=... -e GEMINI_API_KEY=... \
  integrity-review-worker
```

## Isolated AWS test infrastructure

`infra/aws/` is an idempotent boto3 provisioner (no Terraform) for a
non-production ECS Fargate stack: VPC, KMS, S3, SQS, ECR, Secrets Manager,
IAM, ECS, CI/CD (CodeCommit/CodeBuild/CodePipeline/EventBridge), and
CloudWatch alarms. See `infra/aws/config/beta.json` for the config shape and
`infra/aws/cli.py` for the `bootstrap` / `plan` / `apply` / `status` /
`destroy` commands.

## Input contract

```json
{
  "candidateId": "attempt-user-id",
  "assessmentId": "org-assessment-id",
  "cameraRecordings": ["https://signed-camera-chunk.webm"],
  "sessionRecordings": ["https://signed-session-chunk.json"],
  "activityTimeline": [
    {
      "eventType": "ASSESSMENT_STARTED",
      "timestamp": 1700000000000,
      "order": 0
    }
  ],
  "sections": [
    {
      "examAttemptId": "exam-attempt-id",
      "examId": "exam-id",
      "sectionType": "mcq",
      "title": "MCQ"
    }
  ]
}
```

`sessionRecordings` must resolve to one mode per review:

- rrweb: `{epoch}.json` or `{epoch}.json.gz`
- old screen: `{epoch}__{duration}.json` containing a base64 WebM data URL
- new screen: `{epoch}__{duration}.webm`, optionally accompanied by
  `__events.json.gz` / `__metadata.json.gz` sidecars

Mixed screen and rrweb media is rejected. Camera is always analyzed.

## Pipeline

1. URL classification and evidence manifests
2. Canonical activity timeline and one master timeline
3. Camera perception and conditional screen perception (Gemini Flash)
4. Conditional rrweb machine facts and deterministic findings
5. Candidate-only baseline and behavioral correlation
6. Deliberation (one Gemini Pro call)
7. Deterministic scoring and EvidenceBundle assembly

The final bundle omits the intentionally out-of-scope `questionInsights` and
`performanceEvidence` sections. `trackBObservations` remains in JSON for target
contract compatibility, while the Python implementation uses `findings/`
internally.

## Media and clips

Camera/raw WebM is supplied to Gemini using its signed URL. Legacy JSON-wrapped
screen recordings are downloaded, gunzipped when needed, decoded, and supplied
inline. rrweb JSON is parsed locally and never sent to Gemini.

No physical clips are generated. Bundle clips contain offsets into original
recordings, and `mediaIndex` maps each chunk to its test signed URL. The future
ECS adapter can replace `sourceRef` with an S3 key without changing analysis.

## Tests

```bash
.venv/bin/pytest tests
```

The suite covers URL classification, old/new media decoding, timeline
synchronization, perception parsing, machine facts, findings, baseline,
correlation, deliberation rules, scoring, bundle assembly, and an end-to-end
`input.json` to `EvidenceBundle` run using deterministic fakes.
