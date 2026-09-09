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
Fargate worker that consumes `AI_ANALYSIS_REQUEST` messages, stages
requests/results in S3, runs `run_integrity_review`, and publishes
`AI_ANALYSIS_RESPONSE` messages. Configuration is entirely via
environment variables (see `worker/config.py`). Build and run it locally with:

```bash
docker build -t integrity-review-worker .
docker run --rm \
  -e AWS_STORAGE_BUCKET_NAME=... \
  -e REQUEST_QUEUE_URL=... -e RESPONSE_QUEUE_URL=... \
  -e AWS_REGION=... -e GEMINI_API_KEY=... \
  integrity-review-worker
```

## Isolated AWS test infrastructure

`infra/aws/` is an idempotent boto3 provisioner (no Terraform) for a
non-production ECS Fargate stack: a dedicated security group, S3, SQS (with
request/response dead-letter queues), ECR, SSM Parameter Store, IAM, ECS,
and CloudWatch alarms. The ECS worker is deployed into the enterprise
backend's existing VPC and private subnets (see `network` in
`infra/aws/config/beta.json`); this tool never creates, modifies, or
deletes any VPC, subnet, route table, Internet/NAT Gateway, or Elastic IP.
It does create and own the worker's own security group inside that VPC
rather than reusing the existing Lambda security group, since that one has
inbound rules this worker doesn't need. It does not provision its own CI/CD; an externally-owned CodeBuild
project runs `infra/aws/cicd/buildspec.yml`, which calls `bootstrap` /
`apply` / docker build+push / `status` in sequence. See
`infra/aws/config/beta.json` for the config shape and `infra/aws/cli.py`
for the `bootstrap` / `plan` / `apply` / `status` / `destroy` commands.

No customer-managed KMS key is used anywhere in this stack: SQS queues
(and their DLQs), the ECS CloudWatch log group, and the ECR repository are
all unencrypted or use their service's standard/AWS-owned encryption. Each
main queue redrives to its own DLQ after 5 failed receives
(`maxReceiveCount=5`), with a CloudWatch alarm firing if either DLQ is
non-empty. The `GEMINI_API_KEY` env var is injected into the ECS task from
an SSM Parameter Store `SecureString` parameter named
`<project>-<environment>-gemini-api-key` (e.g.
`nw-assessments-ai-analysis-beta-gemini-api-key`), never created by this
tool -- set it out-of-band, once per environment:

```bash
aws ssm put-parameter \
  --name "nw-assessments-ai-analysis-beta-gemini-api-key" \
  --type "SecureString" \
  --value "<GEMINI_API_KEY>" \
  --overwrite
```

The ECS execution role can only read SSM parameters whose name starts with
that same `<project>-<environment>-` prefix, so any future secret must use
the same prefix to be readable by the task.

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
