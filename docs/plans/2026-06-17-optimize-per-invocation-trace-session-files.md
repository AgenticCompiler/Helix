# Optimize Per-Invocation Trace Session Files Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change optimize so each agent launch owns its own trace file, trace summary file, and session metadata file, and remove redundant Claude tool ids from human-readable show-output logs.

**Architecture:** Keep one optimize run archive directory per run id, but derive trace and session file paths from the same per-launch label already used by show-output. Preserve structured trace schemas and non-optimize trace behavior, while making optimize trace summaries launch-local and making Claude show-output cleaner without changing machine-readable trace correlation.

**Tech Stack:** Python, unittest, existing optimize runtime/archive helpers, Claude stream-json rendering

---

### Task 1: Lock The New File Naming Contract With Tests

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/tests/test_optimize_guidance.py`
- Modify: `/Users/cdj/Projects/triton-agent/tests/test_optimize_runtime.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_guidance.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing tests for per-launch archive paths**

Add assertions that optimize archive state uses launch-local file naming instead of `agent-sessions.jsonl` and `otel/trace.jsonl`.

Use assertions shaped like:

```python
self.assertEqual(
    state.agent_session_path("baseline"),
    state.run_archive_dir / "agent-session-baseline.json",
)
self.assertEqual(
    state.trace_path("batch-1-5"),
    state.run_archive_dir / "trace-batch-1-5.jsonl",
)
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests tests.test_optimize_runtime.OptimizeRuntimeTests -v
```

Expected: FAIL because the archive state still exposes `agent-sessions.jsonl` and `otel/trace.jsonl`.

- [ ] **Step 3: Add failing runtime expectations for one file per launch**

Update supervised optimize runtime tests so they expect:

```python
self.assertTrue((run_archive / "agent-session-batch-1-1.json").exists())
self.assertTrue((run_archive / "agent-session-supervisor.json").exists())
self.assertFalse((run_archive / "agent-sessions.jsonl").exists())
```

and, for `--log-tools`-style launch-local trace paths:

```python
self.assertTrue((run_archive / "trace-batch-1-1.jsonl").exists())
```

- [ ] **Step 4: Run the same targeted tests again**

Run:

```bash
uv run python -m unittest tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests tests.test_optimize_runtime.OptimizeRuntimeTests -v
```

Expected: FAIL with missing new file names and/or unexpected old aggregate files.

### Task 2: Implement Optimize Archive Helpers For Per-Launch Trace And Session Files

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/archive.py`
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/session_artifacts.py`
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/execution.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_guidance.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_runtime.py`

- [ ] **Step 1: Replace aggregate archive path fields with per-label helpers**

Update `ArchiveState` so it exposes helper methods instead of fixed optimize-only aggregate paths:

```python
def trace_path(self, label: str) -> Path:
    return self.run_archive_dir / f"trace-{label}.jsonl"

def trace_summary_path(self, label: str) -> Path:
    return self.run_archive_dir / f"trace-{label}.summary.json"

def agent_session_path(self, label: str) -> Path:
    return self.run_archive_dir / f"agent-session-{label}.json"
```

Keep `show_output_path` behavior unchanged.

- [ ] **Step 2: Update archive acceptance rules for the new optimize file set**

Adjust `_EXPECTED_NAMES` and stale-archive checks so they allow:

```python
path.name.startswith("show-output-")
path.name.startswith("trace-") and path.name.endswith(".jsonl")
path.name.startswith("trace-") and path.name.endswith(".summary.json")
path.name.startswith("agent-session-") and path.name.endswith(".json")
```

and stop expecting `otel` or `agent-sessions.jsonl`.

- [ ] **Step 3: Change optimize execution to derive label-local trace paths**

In `_run_request`, use the passed `show_output_label` to derive the optimize trace path:

```python
label = show_output_label or "run"
if request.log_tools:
    env[TRACE_PATH_ENV] = str(self._artifacts_state.archive.trace_path(label))
```

Keep `run_id` and workspace env behavior unchanged.

- [ ] **Step 4: Change optimize session recording to write one JSON file per launch**

Replace append-style recording with overwrite of one JSON object per launch:

```python
def record_agent_session(..., label: str, ...):
    path = state.agent_session_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
```

Update the execution call site to pass the launch label used for `show-output`.

- [ ] **Step 5: Run the targeted optimize tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests tests.test_optimize_runtime.OptimizeRuntimeTests -v
```

Expected: PASS

### Task 3: Make Optimize Trace Summaries Launch-Local Without Breaking Other Commands

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/otel_trace.py`
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/trace_analyze/analyzer.py`
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/commands/trace_analyze.py`
- Modify: `/Users/cdj/Projects/triton-agent/tests/test_trace_analyze_analyzer.py`
- Modify: `/Users/cdj/Projects/triton-agent/tests/test_backends_base.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_trace_analyze_analyzer.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_backends_base.py`

- [ ] **Step 1: Write failing tests for optimize trace summary naming**

Add assertions that optimize-style trace files produce launch-local summary names:

```python
trace_path = Path("/tmp/triton-agent-logs/run-001/trace-batch-1-5.jsonl")
summary = build_summary([], trace_path=trace_path)
self.assertEqual(
    summary["paths"]["summary_json"],
    "/tmp/triton-agent-logs/run-001/trace-batch-1-5.summary.json",
)
```

and that non-optimize trace paths keep `summary.json`.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_trace_analyze_analyzer tests.test_backends_base -v
```

Expected: FAIL because summary naming still always uses `summary.json`.

- [ ] **Step 3: Implement a shared trace-summary path helper**

Add a helper in `otel_trace.py` such as:

```python
def trace_summary_path(trace_path: Path) -> Path:
    name = trace_path.name
    if name.startswith("trace-") and name.endswith(".jsonl"):
        return trace_path.with_name(name[:-6] + ".summary.json")
    return trace_path.parent / "summary.json"
```

Use it in both `write_tool_trace_summary()` and `trace_analyze/analyzer.py`.

- [ ] **Step 4: Update `trace-analyze` output message to use the computed summary path**

Change the command handler to print the actual computed summary file path instead of always `trace_path.parent / "summary.json"`.

- [ ] **Step 5: Run the targeted trace tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_trace_analyze_analyzer tests.test_backends_base -v
```

Expected: PASS

### Task 4: Remove Claude Tool Ids From Show-Output Rendering

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/backends/claude_trace.py`
- Modify: `/Users/cdj/Projects/triton-agent/tests/test_claude_trace.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_claude_trace.py`

- [ ] **Step 1: Write failing tests for human-readable Claude tool lines**

Add or update renderer tests so they expect:

```python
self.assertIn("[tool:start] Read", rendered)
self.assertNotIn("call_00_abcd", rendered)
self.assertIn("[tool:end] Read ok in", rendered)
```

while existing structured trace event assertions still require `tool_use_id`.

- [ ] **Step 2: Run the targeted Claude trace tests to verify they fail**

Run:

```bash
uv run python -m unittest tests.test_claude_trace -v
```

Expected: FAIL because the renderer still includes the tool id in show-output text.

- [ ] **Step 3: Update the Claude show-output renderer only**

Change:

```python
lines = [f"[tool:start] {tool} {tool_use_id or 'unknown'}"]
```

to:

```python
lines = [f"[tool:start] {tool}"]
```

and change:

```python
f"[tool:end] {tool} {tool_use_id or 'unknown'} "
```

to:

```python
f"[tool:end] {tool} "
```

Do not change trace event payload generation.

- [ ] **Step 4: Run the targeted Claude trace tests to verify they pass**

Run:

```bash
uv run python -m unittest tests.test_claude_trace -v
```

Expected: PASS

### Task 5: Update Upload Filtering And User-Facing Docs

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/src/triton_agent/optimize_upload/collector.py`
- Modify: `/Users/cdj/Projects/triton-agent/README.md`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_upload.py`

- [ ] **Step 1: Write a failing upload collector test for new session file names**

Add a test that includes:

```python
workspace / "triton-agent-logs" / "run-001" / "agent-session-batch-1-5.json"
```

and asserts it is excluded from upload.

- [ ] **Step 2: Run the targeted upload test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_optimize_upload -v
```

Expected: FAIL because only `agent-sessions.jsonl` is currently excluded.

- [ ] **Step 3: Update optimize upload exclusion logic and README text**

Exclude any optimize session metadata file matching `agent-session-*.json`, and update the README optimize section so it no longer references:

- `optimize-logs/...`
- `agent-sessions.jsonl`

Use wording that matches the new per-launch files under `triton-agent-logs/<run-id>/`.

- [ ] **Step 4: Run the targeted upload test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_optimize_upload -v
```

Expected: PASS

### Task 6: Final Verification

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/docs/specs/2026-06-17-optimize-per-invocation-trace-session-files-design.md`
- Modify: `/Users/cdj/Projects/triton-agent/docs/plans/2026-06-17-optimize-per-invocation-trace-session-files.md`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_guidance.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_runtime.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_trace_analyze_analyzer.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_backends_base.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_claude_trace.py`
- Test: `/Users/cdj/Projects/triton-agent/tests/test_optimize_upload.py`

- [ ] **Step 1: Run the complete targeted verification set**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  tests.test_trace_analyze_analyzer \
  tests.test_backends_base \
  tests.test_claude_trace \
  tests.test_optimize_upload -v
```

Expected: PASS

- [ ] **Step 2: Run the repository standard static checks**

Run:

```bash
uv run --group dev ruff check
uv run pyright
```

Expected: PASS

- [ ] **Step 3: Update the plan/spec inline if implementation drift required a small wording correction**

If code forced a naming nuance, adjust the spec and plan documents so they match the final shipped behavior exactly.

- [ ] **Step 4: Prepare the final summary**

Report:

- which files changed
- which old optimize log files disappear
- which new optimize log files appear
- what verification commands passed
