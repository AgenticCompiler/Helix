# Optimize Supervised Log Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve supervised optimize orchestration logs by archiving `.triton-agent` handoff artifacts into a persistent `optimize-logs/` directory while keeping the live `.triton-agent/` runtime-only and temporary.

**Architecture:** Keep supervised optimize using the same live `.triton-agent/` paths during execution, but add immutable per-round history snapshots plus a final archive copy under `optimize-logs/triton-agent/<run-id>/` before cleanup removes the live runtime directory. Keep the behavior scoped to `--supervise on` so unsupervised optimize remains unchanged.

**Tech Stack:** Python `pathlib`, `dataclasses`, existing optimize runtime/guidance modules, Python `unittest`

---

## File Structure

**New files**

- `docs/plans/2026-04-13-optimize-supervised-log-archive.md`
  This implementation plan.

**Existing files to modify**

- `src/triton_agent/optimize_guidance.py`
  Extend optimize guidance state with runtime history and archive paths, and add archive-before-cleanup behavior.
- `src/triton_agent/optimize/runtime.py`
  Write immutable per-round history snapshots whenever live handoff files are updated.
- `tests/test_optimize_guidance.py`
  Cover archive layout, cleanup ordering, and failure-safe behavior.
- `tests/test_optimize_runtime.py`
  Cover per-round history snapshots and supervised-only archive behavior.
- `README.md`
  Document supervised log archival at a workflow level if this repository already documents supervised optimize behavior there.

## Task 1: Extend Guidance State With History And Archive Paths

**Files:**
- Modify: `src/triton_agent/optimize_guidance.py`
- Test: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Write the failing guidance-state tests**

Add tests that lock the new filesystem layout returned by `prepare()`:

```python
def test_prepare_exposes_history_and_archive_paths(self) -> None:
    state = manager.prepare(...)
    self.assertEqual(state.history_dir, workdir / ".triton-agent" / "history")
    self.assertEqual(state.archive_root, workdir / "optimize-logs" / "triton-agent")
    self.assertTrue(state.run_archive_dir.parent == state.archive_root)
```

- [ ] **Step 2: Run the focused guidance tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance -v`
Expected: FAIL because guidance state does not yet expose history or archive locations.

- [ ] **Step 3: Implement the minimal state additions**

Extend `OptimizeGuidanceState` with explicit paths such as:

```python
history_dir: Path
archive_root: Path
run_archive_dir: Path
shared_guidance_snapshot_path: Path
```

Update `prepare()` to initialize the live `history/` directory and allocate a unique `run-id`-based archive directory path without writing the archive yet.

- [ ] **Step 4: Run the focused guidance tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_guidance -v`
Expected: PASS

- [ ] **Step 5: Commit the state-path groundwork**

```bash
git add src/triton_agent/optimize_guidance.py tests/test_optimize_guidance.py
git commit -m "feat: add optimize supervised log archive paths"
```

## Task 2: Snapshot Per-Round Handoff History During Supervised Runs

**Files:**
- Modify: `src/triton_agent/optimize/runtime.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing runtime snapshot tests**

Add tests that prove each supervisor handoff produces immutable history files:

```python
def test_supervised_gate_writes_round_history_snapshots(self) -> None:
    request = make_request(supervise="on")
    run_optimize_request(request)
    self.assertTrue((workdir / ".triton-agent" / "history" / "round-001-brief.md").exists())
    self.assertTrue(
        (workdir / ".triton-agent" / "history" / "round-001-supervisor-report.md").exists()
    )

def test_history_snapshots_do_not_overwrite_previous_rounds(self) -> None:
    ...
    self.assertTrue((history_dir / "round-001-brief.md").exists())
    self.assertTrue((history_dir / "round-002-brief.md").exists())
```

- [ ] **Step 2: Run the focused runtime tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_runtime -v`
Expected: FAIL because supervised runs currently overwrite only the live `round-brief.md` and `supervisor-report.md`.

- [ ] **Step 3: Implement immutable history snapshots**

In `OptimizeLoopRunner._write_gate_handoff()`:

- keep writing the live `.triton-agent/round-brief.md`
- keep writing the live `.triton-agent/supervisor-report.md`
- additionally write immutable copies under `.triton-agent/history/`

Use a deterministic round label such as `round-001` based on the latest round directory or an explicit round index helper. Keep filenames stable and sortable.

- [ ] **Step 4: Run the focused runtime tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 5: Commit the history snapshot behavior**

```bash
git add src/triton_agent/optimize/runtime.py tests/test_optimize_runtime.py
git commit -m "feat: snapshot optimize supervisor handoff history"
```

## Task 3: Archive Supervised Logs Before Cleanup

**Files:**
- Modify: `src/triton_agent/optimize_guidance.py`
- Modify: `src/triton_agent/optimize/runtime.py`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing archive tests**

Add tests that pin the final archive layout:

```python
def test_supervised_cleanup_archives_history_and_final_files(self) -> None:
    state = manager.prepare(...)
    ...  # write live handoff files and history files
    manager.archive(state)
    self.assertTrue((state.run_archive_dir / "history" / "round-001-brief.md").exists())
    self.assertTrue((state.run_archive_dir / "final" / "supervisor-report.md").exists())
    self.assertTrue((state.run_archive_dir / "roles" / "optimize-worker.md").exists())

def test_unsupervised_optimize_does_not_create_archive(self) -> None:
    request = make_request(supervise="off")
    run_optimize_request(request)
    self.assertFalse((workdir / "optimize-logs").exists())
```

- [ ] **Step 2: Run the focused archive tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`
Expected: FAIL because no archive is written today.

- [ ] **Step 3: Implement archive-before-cleanup**

Add an explicit archive step that:

- writes `shared-guidance.md` as a snapshot of the rendered shared guidance
- copies role briefs into `roles/`
- copies final live `round-brief.md` and `supervisor-report.md` into `final/`
- copies live `history/` into `history/`

Run this step only for supervised optimize and only before live cleanup removes `.triton-agent/`.

- [ ] **Step 4: Run the focused archive tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 5: Commit the archive flow**

```bash
git add src/triton_agent/optimize_guidance.py src/triton_agent/optimize/runtime.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py
git commit -m "feat: archive optimize supervised logs"
```

## Task 4: Keep Cleanup Safe And Supervised-Only

**Files:**
- Modify: `src/triton_agent/optimize_guidance.py`
- Test: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Write the failing cleanup-safety tests**

Add tests that prove cleanup still removes only live runtime files and preserves the new archive:

```python
def test_cleanup_preserves_archived_logs(self) -> None:
    ...
    manager.cleanup(state)
    self.assertTrue(state.run_archive_dir.exists())
    self.assertFalse((workdir / ".triton-agent").exists())

def test_cleanup_warns_when_archive_write_fails(self) -> None:
    ...
    self.assertTrue(any("archive" in warning for warning in warnings))
```

- [ ] **Step 2: Run the focused cleanup tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance -v`
Expected: FAIL because cleanup currently knows only about deleting live temporary files.

- [ ] **Step 3: Implement safe cleanup ordering**

Adjust cleanup flow to:

1. attempt archive creation
2. collect short warnings if archive creation fails
3. continue best-effort removal of live `.triton-agent/` files
4. restore or remove the top-level shared guidance file exactly as before

Do not delete or rewrite any archive path during cleanup.

- [ ] **Step 4: Run the focused cleanup tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_guidance -v`
Expected: PASS

- [ ] **Step 5: Commit the cleanup safeguards**

```bash
git add src/triton_agent/optimize_guidance.py tests/test_optimize_guidance.py
git commit -m "fix: preserve optimize supervised log archives during cleanup"
```

## Task 5: Document The Supervised Archive Behavior And Run Full Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Update workflow documentation**

Document that `--supervise on` keeps live `.triton-agent/` temporary but archives supervised orchestration artifacts under `optimize-logs/triton-agent/<run-id>/` after the run.

- [ ] **Step 2: Run targeted regression tests**

Run: `uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 3: Run repository verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 4: Commit the documentation and verification pass**

```bash
git add README.md src/triton_agent/optimize_guidance.py src/triton_agent/optimize/runtime.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py docs/plans/2026-04-13-optimize-supervised-log-archive.md
git commit -m "docs: describe optimize supervised log archives"
```
