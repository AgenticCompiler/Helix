# Optimize Auto Bench Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `optimize` and `optimize-batch` to accept `--resume auto --bench-mode ...` while keeping recorded benchmark metadata authoritative for resumed optimize sessions.

**Architecture:** Keep the change local to optimize resume resolution. `resolve_optimize_resume()` should continue rejecting `--bench-mode` for explicit `continue`, but `auto` should stop failing on resumable workspaces and instead reuse the recorded benchmark mode. CLI tests should lock both single-workspace and mixed batch behavior, and README should describe the fresh-vs-resume split.

**Tech Stack:** Python `argparse`, optimize orchestration helpers, Python `unittest`, Markdown docs

---

### Task 1: Lock Single-Workspace `auto` Resume Behavior

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add a test near the existing optimize resume cases that creates a resumable optimize workspace with recorded `# bench-mode: msprof`, then runs:

```python
exit_code = main(
    [
        "optimize",
        "-i",
        str(operator),
        "--resume",
        "auto",
        "--bench-mode",
        "standalone",
    ]
)
```

Assert:

```python
self.assertEqual(exit_code, 0)
self.assertTrue(captured["resume_existing_session"])
self.assertEqual(captured["bench_mode"], "msprof")
self.assertEqual(request.bench_mode, "msprof")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_cli.PathResolutionTests.test_main_optimize_resume_auto_accepts_explicit_bench_mode_for_resumable_session -v
```

Expected: FAIL because resume auto currently rejects explicit `--bench-mode` for resumable sessions.

- [ ] **Step 3: Write minimal implementation**

Update `src/triton_agent/optimize/resume.py` so the `resume_mode == "auto"` resumable-session path no longer raises on `requested_bench_mode`, and instead returns:

```python
return ResumeResolution(
    workspace_state="resumable-session",
    resume_existing_session=True,
    test_mode=inspection.test_mode,
    bench_mode=inspection.bench_mode,
)
```

Keep the existing explicit `requested_test_mode` rejection for resumed auto sessions, and keep the `resume_mode == "continue"` explicit bench-mode rejection unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_cli.PathResolutionTests.test_main_optimize_resume_auto_accepts_explicit_bench_mode_for_resumable_session -v
```

Expected: PASS

### Task 2: Lock Mixed Batch `auto` Behavior

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add a batch test that builds:

- one resumable workspace with recorded `# bench-mode: standalone`
- one fresh workspace with no optimize artifacts

Run:

```python
exit_code = main(
    [
        "optimize-batch",
        "-i",
        str(root),
        "--resume",
        "auto",
        "--bench-mode",
        "msprof",
    ]
)
```

Capture each `request.bench_mode` in the patched batch runner and assert:

```python
self.assertEqual(exit_code, 0)
self.assertEqual(captured_modes["resume_ws"], "standalone")
self.assertEqual(captured_modes["fresh_ws"], "msprof")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_cli.PathResolutionTests.test_main_optimize_batch_resume_auto_accepts_explicit_bench_mode_for_mixed_workspaces -v
```

Expected: FAIL because the resumable workspace currently causes resume validation failure.

- [ ] **Step 3: Write minimal implementation**

Rely on the same `resolve_optimize_resume()` change from Task 1. Do not add batch-specific override logic unless the batch test shows a separate bug.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_cli.PathResolutionTests.test_main_optimize_batch_resume_auto_accepts_explicit_bench_mode_for_mixed_workspaces -v
```

Expected: PASS

### Task 3: Document The Fresh-vs-Resume Rule

**Files:**
- Modify: `README.md`
- Test: none

- [ ] **Step 1: Update optimize documentation**

Adjust the optimize and optimize-batch option descriptions so they explicitly say:

```markdown
- `--bench-mode standalone|msprof`: sets the benchmark mode for fresh runs. With `--resume auto`, resumable workspaces keep the benchmark mode recorded in their existing benchmark harness.
```

- [ ] **Step 2: Review for consistency**

Check the optimize and optimize-batch sections in `README.md` so they describe the same rule and do not imply that resumed sessions switch benchmark mode from the CLI flag.

### Task 4: Focused Verification

**Files:**
- Modify: none
- Test: `tests/test_cli.py`

- [ ] **Step 1: Run the targeted CLI tests**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.PathResolutionTests.test_main_optimize_resume_auto_accepts_explicit_bench_mode_for_resumable_session \
  tests.test_cli.PathResolutionTests.test_main_optimize_batch_resume_auto_accepts_explicit_bench_mode_for_mixed_workspaces \
  tests.test_cli.PathResolutionTests.test_main_optimize_resume_continue_rejects_explicit_bench_mode \
  -v
```

Expected: PASS

- [ ] **Step 2: Run a broader optimize CLI slice**

Run:

```bash
uv run python -m unittest tests.test_cli.PathResolutionTests -v
```

Expected: PASS
