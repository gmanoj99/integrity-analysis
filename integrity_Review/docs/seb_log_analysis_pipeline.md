# SEB / TSB Log Analysis Pipeline

**Audience:** investors, product stakeholders, and engineering leadership  
**Scope:** Safe Exam Browser / Topin Secure Browser session logs (Windows and macOS)  
**Source of truth:** current implementation in `integrity_review_pipeline/seb_logs/` and `prompts/seb_log_analysis.py`  
**Not in scope:** video / screen analysis (separate pipeline)

---

## 1. Why this exists

A typical exam session produces **tens to hundreds of thousands of log lines** (often tens of megabytes). Sending that volume to a language model would be expensive, noisy, and unsafe: the model could invent events that never appeared in the logs.

This pipeline therefore splits the work:

| Layer | Who | Job |
|---|---|---|
| **Deterministic reduction** | Code (no AI) | Parse logs, drop junk, extract security events, redact secrets, produce a structured evidence pack |
| **Interpretation** | LLM (Gemini, JSON output, temperature 0) | Explain what the evidence means for a human reviewer — correlate, prioritize, write a narrative |
| **Post-LLM checks** | Code (no AI) | Confirm the model’s JSON is usable and that findings do not contradict extracted facts in a few hard cases |

The model is **not** asked to decide that a candidate cheated. It is asked to surface evidence-based findings for a human reviewer.

---

## 2. End-to-end picture

```
Raw session logs in object storage
        │
        ▼
Discover sessions (Windows 4-file set or macOS unified log)
        │
        ▼
PRE-LLM REDUCTION  (code)
  parse → collapse bulk kills → collapse noise → extract signals
  → collect processes / windows / config / coverage → redact PII
        │
        ▼
Structured evidence  +  capped text prompt
        │
        ▼
LLM  (interpret only)
        │
        ▼
POST-LLM VALIDATION  (code)
  parse JSON → citation checks → severity warning → enforcement gate
  → schema check
        │
        ▼
Stored result: reduction + AI analysis + validation notes
        │
        ▼
Human reviewer
```

On a real Windows session (~70 MB of logs, ~370k Browser.log lines):

- **~9,700** repetitive lines collapsed as noise  
- **~1,350** process-kill lines collapsed into a few application summaries  
- **11** security signal *types* retained (with counts), not thousands of duplicate lines  
- The model then sees a **short text briefing**, not the raw files  

---

## 3. Pre-LLM reduction

Reduction is **Process 2 (deterministic analysis)**. Entry points:

- Worker: `worker/message_processor.py`  
- Orchestrator: `integrity_review_pipeline/seb_logs/pipeline.py`  
- Per session: `SessionReducer` in `integrity_review_pipeline/seb_logs/reduction/session_reducer.py`

Each log line is handled in a **fixed order**. A line that is absorbed as bulk or noise is **not** also turned into process/window metadata (with one exception: Windows “belongs to blacklisted application” still becomes a signal).

### 3.1 Discover the session

**What:** Find a folder that looks like `YYYY-MM-DD_HHhMMmSSs[_id]` and map files.

**Windows:** `Runtime.log`, `Client.log`, `Browser.log`, `Service.log`, optional `manifest.json`.  
**macOS:** one unified `org.safeexambrowser.SafeExamBrowser ….log` (if several exist, the latest is used).

**Why:** Reviewers need to know which files were present. Missing `Service.log` on Windows means OS lockdowns often cannot be verified.

**Discarded here:** macOS metadata files (`._*`, `__MACOSX`). Extra macOS logs beyond the newest file.

---

### 3.2 Parse each line (grammar)

**What:** Turn free text into timestamp, severity, module, and message.

- Windows Client/Runtime/Service: `2026-08-15 19:07:14.480 [01] - INFO: [KeyboardInterceptor] Blocked 'Alt+F4'`  
- Browser: Chromium-style CEF lines  
- macOS: `YYYY/MM/DD HH:MM:SS:mmm  message` (including multi-line blocks)

**Why:** Downstream rules need structure. Header comments are skipped as events but still used to read OS/machine identity.

**Fate:** Unparseable empty/header lines are not treated as events.

---

### 3.3 Track coverage gaps and session banners

These run on **every parsed line**, including lines later classified as noise.

**Coverage gaps:** If a chatty file (not Browser.log) goes silent for **more than 2 minutes**, that silence is recorded. Browser.log is skipped because Chromium only writes on errors — quiet browser logs are normal.

**Session phases:** Only banner lines such as `### ---- Session Start Procedure ---- ###`. The AI later infers a fuller “journey”; the reducer does not invent phases.

**Why:** A clean-looking session with a 10-minute hole in Client.log is not a clean session. Banners tell us launch vs running vs stop.

---

### 3.4 Collapse bulk process termination

**What:** At startup, SEB often kills blacklisted or leftover processes. That can be **thousands of lines** (`belongs to blacklisted application`, `Attempting to close…`, `Successfully terminated…`).

**Output:** One row per application, e.g. “Zoom.exe, 3 instances, all terminated, force kill used” vs “msedgewebview2.exe, 85 instances” (SEB’s own browser engine — housekeeping, not cheating).

**Why:** Without this, the model (or a human) would drown in kill spam or treat WebView2 deaths as malpractice.

**Fate of the raw lines:** **Summarized.** Individual kill attempts are not kept.

---

### 3.5 Collapse noise

**What:** High-volume, low-value chatter:

- Keepalive / ping-style lines  
- Repeated “checking N configurations”  
- Microphone level read failures  
- WebRTC capture errors, GCM registration, detached CEF frames, TURN errors  
- macOS window-geometry and kiosk-level churn  

**Output:** Counts + first/last time, e.g. “3,698 microphone read errors over the session.”

**Why:** Proves the session was “alive” without flooding the model. 3,698 mic errors are **not** 3,698 integrity incidents.

**Fate:** **Summarized.** Line text is discarded from the prompt (only the count is sent).

---

### 3.6 Extract security signals

**What:** Pattern rules for events that matter: blocked keystrokes, blocked mouse, blacklisted apps, injected keystrokes, VM/remote session, integrity failures, display policy, DevTools, OS lockdown drift, and platform-specific macOS rules.

Repeats of the **same** rule become **one signal** with:

- `occurrenceCount`  
- first / last seen  
- up to **5** sample log lines (redacted)  
- deterministic **severity** and **enforcedByTsb** (was the action blocked?)

Example: 57 `Blocked 'Alt+F4'` lines → one signal `blocked_keystroke`, count 57, severity `low`, `enforcedByTsb: true`.

**Why:** This is the integrity backbone. The model must interpret these IDs; it must not rediscover raw regexes.

**Important honesty for stakeholders:** Absence of a signal means **the extractor did not match a rule**, not mathematical proof the event never occurred. That limitation is stated in the model prompt on purpose.

**Fate:** **Aggregated and retained** as the primary evidence list.

---

### 3.7 Collect supporting context (metadata)

From lines that were neither bulk nor noise:

| Collected | Typical log origin | Why it matters |
|---|---|---|
| Environment | File headers (OS, machine, version); VM/RDP lines | Identity and environment risk |
| Effective config | Service.log registry / setting lines | Was the behaviour *allowed* by lockdown? |
| Processes started | `Process 'foo.exe' (pid) has been started` | What else ran |
| Foreground windows | `Window has changed from … to …` | Focus / quit-password dialogs / other apps |
| Browser activity | Navigations, blocked requests, popups | Off-exam browsing |
| Displays | Display detection lines | Extra monitors |

Sensitive values in evidence, titles, and URLs are **redacted** (emails, IPs, tokens, user folder names).

**Fate:** **Transformed** into structured lists. Most routine INFO/DEBUG lines that match nothing are **discarded**.

Warning/error lines that survive become a **timeline**, capped at 100 (log level ≠ integrity severity).

---

### 3.8 Correlate and assemble the evidence pack

- Cluster signals whose timestamps fall within **10 seconds** and involve **at least two different** signal types → `correlatedIncidents`  
- Compare expected monitors (keyboard, mouse, clipboard, …) to what SEB logged as started → `monitoringCoverage`  
- Hard caps: timeline 100, navigations 20, page errors 10  

**Output contract:** `ReducedSessionEvidence` (JSON). This pack is **stored with the review**. The LLM does **not** receive this JSON in full.

---

### 3.9 Second reduction: the prompt the model actually sees

A text user prompt is built (`build_seb_log_analysis_user_prompt`) with further caps, for example:

- at most **60** signals, **3** evidence lines each  
- **80** timeline rows, **40** processes, **40** window spans  
- **30** config fields, **12** noise categories  

The prompt also includes a **citation inventory**: the exact signal IDs, evidence references (`client:11683`), config names, process `name:pid`, and window titles the model is allowed to copy.

**System instructions** tell the model:

- Do not invent events  
- Blocked ≠ breached  
- Technical failures (network, crash) are not malpractice  
- Never conclude cheating  
- Config that permits an action is not a violation  
- Cover confidence honestly (`could_not_be_established` vs `no_suspicious_activity_observed`)

---

## 4. What the LLM is asked to produce

The model returns **one JSON object** per session:

| Section | Purpose |
|---|---|
| `sessionJourney` | Narrative phases (launch → exam → shutdown), including inferred phases |
| `findings` | Prioritized integrity items, each citing signal IDs and evidence refs |
| `correlatedIncidents` | Interpreted clusters (suspicious / technical / mixed) |
| `technicalProblems` | Crashes, network, capture failures — not findings |
| `configurationAssessment` | What lockdown was in force; which signals it **explains** |
| `coverageAssessment` | How much we could see; confidence; conclusion qualifier |
| `reviewerSummary` | Short briefing for a human (including ~250–300 word `summaryText`) |

### Configuration assessment (detail)

Interpretation of `effectiveConfig` and related signals — **not** a raw registry dump.

- **appliedConfigSummary** — one-line policy picture (e.g. lockdown with screen capture permitted)  
- **deviations** — drift the *logs* show (e.g. Windows Update re-enabled)  
- **protectionsWeakened** — guards that were off (even if config allowed it)  
- **configExplainedSignals** — “window guard off because screen capture is allowed”  
- **unknownOrDefaulted** / **notes** — uncertain or extra config facts  

Code does **not** generate or rewrite this object.

### Coverage assessment (detail)

How complete the observation was.

- **analysisConfidence** — `high` / `medium` / `low`  
- **missingLogFiles** / **logGaps** / **limitations** (unarmed clipboard, integrity module down, …)  
- **conclusionQualifier** (only two values):  
  - `no_suspicious_activity_observed` — coverage was sufficient **and** nothing suspicious  
  - `could_not_be_established` — do **not** close the case as clean  

There is **no** model-produced “cheating confirmed” verdict. Suspicion lives in findings for humans.

---

## 5. Post-LLM validation

Implemented in `integrity_review_pipeline/seb_logs/analysis/engine.py` and `analysis/rules.py`.

Retries: **one extra LLM call** only if JSON cannot be parsed or the JSON **shape** fails schema. Citation/severity/enforcement issues do **not** trigger a retry.

### 5.1 Parse JSON

Strip markdown fences if needed. Unparseable after retry → session marked `analysisFailed` (reduction is still kept).

### 5.2 Citation check

Build an allow-list from **reduced evidence**. For each finding:

| LLM field | Check | If it fails |
|---|---|---|
| `signalsCited` | ID must exist in extracted signals | **Error logged**; finding **kept** |
| `evidenceCited` | `fileType:lineNumber` must be on a kept evidence line | **Warning**; finding kept |
| `configCited` | Name must exist in effective config | **Warning**; finding kept |

Incidents: `findingsCited` must refer to findings **in this same JSON** (warning if not).

Evidence allow-list is **intentionally incomplete** (only a few sample lines per signal), so missing evidence is a warning, not a hard reject.

**Not checked today:** process names/PIDs, window titles, monitor names, whether AI incident IDs match reducer incidents, or whether the *prose* matches the cited line. Those names are listed in the prompt to steer the model; they are not a hard gate.

### 5.3 Severity ceiling

If a finding cites signals, the model’s severity **should not exceed** the highest deterministic severity of those signals.

Exceeding → **warning only**. Severity is **not** rewritten.  
Example seen in production-style tests: model said `low` for a foreground-window finding whose extracted severity is `info`.

### 5.4 Enforcement gate (the only finding removal)

If the extractor marked `enforcedByTsb: true` (action was **blocked**) and the model says `restriction_breached` (action **succeeded**), the finding is **removed**.

The reverse (model says blocked, extractor says it was not) is also removed.

This is the main guard against “they cheated” when SEB actually stopped the action.

### 5.5 Coverage conclusion

If the model claims `no_suspicious_activity_observed` but Windows log files are missing or there are gaps **> 2 minutes** → **warning**. Value is not auto-corrected. macOS is not flagged for “missing” Windows-style files (single log is expected).

### 5.6 Summary list coercion

If the model returns objects instead of strings in reviewer-summary lists, they are flattened to text so schema can succeed.

### 5.7 Schema

Pydantic requires the right types and enums (`blocked_by_tsb`, conclusion qualifier, confidence). Extra unknown keys fail. After retry, schema failure → `analysisFailed`.

Validation messages are stored on the result as `validationErrors` (errors **and** warnings). `analysisFailed` stays **false** unless parse/schema/API failed.

---

## 6. What a stored review contains

For each session:

1. **Full reduction JSON** — signals, bulk kills, coverage, processes, etc. (audit trail; humans can inspect this without trusting the model)  
2. **AI analysis** — journey, findings, assessments, reviewer summary  
3. **Raw model text** — for debugging  
4. **Validation notes** — warnings/errors from the checks above  

The worker can still succeed if **some** sessions analyse and others fail; if **all** AI sessions fail, the job is retried.

---

## 7. Worked example (same bytes through the pipe)

**Raw Client.log (many lines):**

```
Process 'Zoom.exe' (5672) belongs to blacklisted application
Attempting to close process 'Zoom.exe' (5672)
Successfully terminated process 'Zoom.exe' (5672)
```

**After reduction**

- Bulk row: Zoom.exe, 3 instances, all terminated  
- Signal: `blacklisted_application_terminated`, count 3, severity medium, `enforcedByTsb: true`, a few `client:NNNN` evidence lines  

**LLM finding (typical)**

- Title: blacklisted application terminated  
- Enforcement: `blocked_by_tsb`  
- Cites `blacklisted_application_terminated` and those line refs  

**Validation:** citations exist; severity does not exceed medium; blocked matches `enforcedByTsb: true` → finding **kept**.

If the model had said the restriction was **breached**, that finding would be **deleted**.

---

## 8. Division of trust (for decision-makers)

| Claim | Who is allowed to make it |
|---|---|
| “This log line happened” | **Code** (extractor) |
| “This is the 57th blocked Alt+F4” | **Code** (counts) |
| “SEB blocked it vs the candidate succeeded” | **Code** (`enforcedByTsb`); model must agree or the finding is dropped |
| “These two events in 10 seconds are one incident” | Code proposes; model may reinterpret |
| “This matters to a reviewer / looks like a pattern” | **Model**, grounded in cited signal IDs |
| “Coverage was too weak to call the session clean” | **Model**, with a light code warning if it is too optimistic |
| “The candidate cheated” | **Neither** — forbidden in the prompt; no automated cheating verdict |

**Current design choice:** prefer **not discarding** a complete analysis over aggressive filtering. Most citation problems are logged, not stripped. The hard stop is **enforcement contradiction** and **unusable JSON**.

---

## 9. Platforms

| | Windows | macOS |
|---|---|---|
| Logs | Four files + optional manifest | One unified log |
| Missing Service.log | Coverage caveat (lockdowns unverified) | N/A |
| Bulk kills / noise / signals | Windows rules | Separate macOS rules, same pipeline |

---

## 10. Implementation map (for engineering follow-up)

| Stage | Primary files |
|---|---|
| Discovery | `integrity_review_pipeline/seb_logs/discovery.py` |
| Parse | `grammars/windows_dotnet.py`, `cef_browser.py`, `macos.py` |
| Reduce | `reduction/session_reducer.py`, `bulk_event_aggregator.py`, `noise_rules.py`, `signal_rules.py`, `correlator.py`, `redaction.py` |
| Contracts | `integrity_review_pipeline/seb_logs/contracts.py` |
| Prompt | `prompts/seb_log_analysis.py` |
| LLM call | `analysis/engine.py` |
| Validation | `analysis/rules.py` |
| Output schema | `analysis/contracts.py` |
| Worker | `worker/message_processor.py` |

---

*This document describes the system as implemented. Prompt wording that is not backed by code (for example recomputing a session verdict after the model runs) is not treated as product behaviour.*
