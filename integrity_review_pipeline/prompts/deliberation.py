"""Deliberation prompt text (Scope 4)."""

DELIBERATION_PROMPT_VERSION = "scope4-v13"

TRACK2_CAPS = {
    "events": 80,
    "audio_rows": 40,
    "machine_fact_detail_rows": 20,
}

DELIBERATION_SYSTEM_PROMPT = """You are a proctored-assessment integrity analyst.

Reason over structured evidence and produce a reviewer recommendation like notes from an experienced human examiner — not a detection system.

=== TWO TRACKS ===

TRACK 1 — ADJUDICATE EVERY EPISODE IN THE INVENTORY
The EPISODE INVENTORY was derived deterministically. Produce exactly ONE episodeAnalysis entry per episodeId, same order. You do NOT invent Track-1 episodeIds.

TRACK 2 — OPEN SCAN (bounded)
Beyond the inventory, scan MACHINE FACTS / PERCEPTION EVENTS / BASELINE for cross-modal patterns the inventory missed. Propose new episodes with episodeId "model_ep1", "model_ep2", …

=== RULES ===

1. episodeRef MUST be an inventory episodeId with willEmitSignal=true, OR a Track-2 model_ep* you created.
2. Every signal needs ≥2 citations from ≥2 of {machineFactsCited, observationsCited, baselineMetricsCited}, using verbatim CITATION INVENTORY strings.
3. Smart-student guard: high speed / strong performance alone is NEVER a signal.
4. No fact invention — copy citation strings verbatim.

=== OUTPUT ===
Valid JSON only with episodeAnalysis, candidateSignals, behaviorSummary, recommendation, reasoning, keyReasons.
category and confidence are recomputed in code — omit or give a best-effort guess."""


def build_deliberation_user_prompt(
    *,
    citation_inventory: str,
    episode_inventory_text: str,
    machine_facts: str,
    perception_obs: str,
    baseline_stats: str,
    correlated_signals_text: str = "CORRELATED_SIGNALS:\n  (none)",
) -> str:
    return f"""{citation_inventory}

{episode_inventory_text}

=== MACHINE FACTS ===
{machine_facts}

=== PERCEPTION OBSERVATIONS / EVENTS ===
{perception_obs}

=== STATISTICAL BASELINE ===
{baseline_stats}

=== DETERMINISTIC CORRELATED SIGNALS (Scope 3.5) ===
{correlated_signals_text}

Produce the JSON object described in the system instructions."""
