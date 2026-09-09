# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Only copy build metadata first so dependency layers cache independently of
# application code changes.
COPY pyproject.toml ./

COPY __init__.py pipeline.py behavioral_pipeline.py deps.py duck_helpers.py gemini_call.py ./
COPY adapters ./adapters
COPY baseline ./baseline
COPY contracts ./contracts
COPY correlation ./correlation
COPY deliberation ./deliberation
COPY evidence_bundle ./evidence_bundle
COPY findings ./findings
COPY integrity_review_pipeline ./integrity_review_pipeline
COPY lib ./lib
COPY machine_facts ./machine_facts
COPY media ./media
COPY perception ./perception
COPY prompts ./prompts
COPY scoring ./scoring
COPY timeline ./timeline
COPY worker ./worker

RUN pip install --no-cache-dir .

RUN groupadd --system worker && useradd --system --gid worker --no-create-home worker
USER worker

# The worker is a long-poll SQS consumer with no inbound port; ECS health is
# process liveness, not an HTTP endpoint.
ENTRYPOINT ["python", "-m", "nw_assessments_ai_analysis_backend.worker.main"]
