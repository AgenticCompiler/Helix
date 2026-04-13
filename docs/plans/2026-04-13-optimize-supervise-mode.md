# Optimize Supervise Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `--supervise on|off` optimize mode so ordinary optimize runs stay single-agent by default while supervised round-gate orchestration remains available when requested.

**Architecture:** Keep `optimize` as one command surface with two orchestration modes. Route the choice through `OptimizeRunOptions`, build one `AgentRequest`, then split in `run_optimize_request()` between the existing single-agent optimize path and the existing worker-plus-supervisor round-gate path so preparation, cleanup, and output rendering stay shared.

**Tech Stack:** Python `argparse`, `dataclasses`, existing optimize runtime/supervisor modules, Python `unittest`

---

## File Structure

**New files**

- `docs/plans/2026-04-13-optimize-supervise-mode.md`
  This implementation plan.

**Existing files to modify**

- `src/triton_agent/cli.py`
  Add `--supervise` to `optimize` and `optimize-batch`.
- `src/triton_agent/commands/optimize.py`
  Parse supervise mode into `OptimizeRunOptions`.
- `src/triton_agent/optimize/models.py`
  Extend `OptimizeRunOptions` with the supervise setting.
- `src/triton_agent/optimize/runtime.py`
  Split optimize execution into unsupervised versus supervised paths while sharing setup and cleanup.
- `src/triton_agent/supervisor.py`
  Reuse the existing recovery loop for the unsupervised path without changing supervised semantics.
- `src/triton_agent/optimize/batch.py`
  Keep batch routing unchanged except for passing the selected supervise mode through request construction.
- `tests/test_cli.py`
  Add parser and command-behavior tests for the new option.
- `tests/test_optimize_runtime.py`
  Add path-selection tests for supervised versus unsupervised execution.
- `README.md`
  Document `--supervise on|off` and its default.

## Task 1: Add The Supervise Option To Optimize Models And CLI

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/models.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser tests**

Add tests that lock the public CLI contract before implementation:

```python
def test_optimize_accepts_supervise_on(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--supervise", "on"])
    self.assertEqual(args.supervise, "on")

def test_optimize_defaults_supervise_off(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py"])
    self.assertEqual(args.supervise, "off")
```

Also add the same coverage for `optimize-batch`.

- [ ] **Step 2: Run the focused parser tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: FAIL because `--supervise` does not exist yet.

- [ ] **Step 3: Implement the minimal option plumbing**

Add `--supervise` only to `optimize` and `optimize-batch`:

```python
subparser.add_argument(
    "--supervise",
    default="off",
    choices=["on", "off"],
)
```

Extend `OptimizeRunOptions` with:

```python
supervise: str
```

and thread `args.supervise` through `optimize_run_options_from_args()`.

- [ ] **Step 4: Run the focused parser tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: PASS

- [ ] **Step 5: Commit the CLI contract change**

```bash
git add src/triton_agent/cli.py src/triton_agent/commands/optimize.py src/triton_agent/optimize/models.py tests/test_cli.py
git commit -m "feat: add optimize supervise mode option"
```

## Task 2: Split Optimize Runtime Into Unsupervised And Supervised Paths

**Files:**
- Modify: `src/triton_agent/optimize/runtime.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing runtime-path tests**

Add tests that prove `run_optimize_request()` chooses the right orchestration path:

```python
def test_run_optimize_request_uses_unsupervised_path_by_default(self) -> None:
    request = make_request(supervise="off")
    result = run_optimize_request(request)
    self.assertEqual(result.return_code, 0)
    self.assertEqual(runner.calls, ["run"])

def test_run_optimize_request_uses_round_gate_when_supervise_on(self) -> None:
    request = make_request(supervise="on")
    result = run_optimize_request(request)
    self.assertEqual(result.return_code, 0)
    self.assertEqual(runner.calls, ["run_worker", "run_supervisor"])
```

- [ ] **Step 2: Run the focused runtime tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_runtime -v`
Expected: FAIL because optimize currently always enters the round-gate path.

- [ ] **Step 3: Implement shared runtime preparation plus mode dispatch**

Refactor `run_optimize_request()` into one shared shell plus two internal execution helpers:

```python
def run_optimize_request(...):
    ...
    if request.supervise == "on":
        return _run_optimize_request_supervised(...)
    return _run_optimize_request_unsupervised(...)
```

Guidelines:

- Keep skill staging shared.
- Keep result rendering and cleanup shared.
- Only the orchestration branch should differ.
- Leave `OptimizeLoopRunner` for supervised mode.
- Add a lightweight unsupervised runner wrapper that uses the existing `run()`/`resume()` recovery interface and avoids worker/supervisor briefs.

- [ ] **Step 4: Run the focused runtime tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 5: Commit the runtime split**

```bash
git add src/triton_agent/optimize/runtime.py tests/test_optimize_runtime.py
git commit -m "feat: split optimize supervised and unsupervised runtime paths"
```

## Task 3: Keep Guidance Rendering Scoped To Supervised Mode

**Files:**
- Modify: `src/triton_agent/optimize/runtime.py`
- Modify: `src/triton_agent/optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`
- Test: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Write the failing guidance-scope tests**

Add tests that lock the filesystem behavior:

```python
def test_unsupervised_optimize_does_not_prepare_role_briefs(self) -> None:
    request = make_request(supervise="off")
    run_optimize_request(request)
    self.assertFalse((workdir / ".triton-agent" / "roles").exists())

def test_supervised_optimize_prepares_role_briefs(self) -> None:
    request = make_request(supervise="on")
    run_optimize_request(request)
    self.assertTrue(guidance_manager.prepare_called)
```

- [ ] **Step 2: Run the focused guidance tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_optimize_guidance -v`
Expected: FAIL because runtime currently always prepares the supervised guidance layout.

- [ ] **Step 3: Implement mode-specific guidance behavior**

For `supervise="off"`:

- skip `OptimizeGuidanceManager.prepare()`
- skip `.triton-agent/roles/`
- skip `round-brief.md` and `supervisor-report.md`
- keep only the preparation required by the original single-agent optimize path

For `supervise="on"`:

- preserve the existing shared role-neutral guidance layout
- preserve detailed verbose logging and cleanup

If the unsupervised path still needs a top-level temporary guidance file, keep that behavior isolated and do not create role-brief artifacts.

- [ ] **Step 4: Run the focused guidance tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_optimize_guidance -v`
Expected: PASS

- [ ] **Step 5: Commit the guidance split**

```bash
git add src/triton_agent/optimize/runtime.py src/triton_agent/optimize_guidance.py tests/test_optimize_runtime.py tests/test_optimize_guidance.py
git commit -m "feat: scope optimize guidance to supervised mode"
```

## Task 4: Preserve Legacy Recovery Semantics For Unsupervised Optimize

**Files:**
- Modify: `src/triton_agent/supervisor.py`
- Modify: `src/triton_agent/optimize/runtime.py`
- Test: `tests/test_supervisor.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing legacy-path tests**

Add tests that pin expected unsupervised behavior:

```python
def test_unsupervised_optimize_uses_recovery_runner_interfaces(self) -> None:
    request = make_request(supervise="off")
    OptimizeSupervisor().run(runner, request)
    self.assertEqual(runner.calls, ["run"])

def test_unsupervised_interact_returns_failed_result_without_resume(self) -> None:
    request = make_request(supervise="off", interact=True)
    result = OptimizeSupervisor().run(runner, request)
    self.assertFalse(result.succeeded)
    self.assertEqual(runner.resume_calls, 0)
```

- [ ] **Step 2: Run the focused supervisor tests to verify they fail**

Run: `uv run python -m unittest tests.test_supervisor tests.test_optimize_runtime -v`
Expected: FAIL because runtime still always wraps optimize in the round-gate runner.

- [ ] **Step 3: Reconnect the unsupervised path to the recovery loop**

Use the existing `RunnerWithStreams` plus `OptimizeSupervisor.run()` recovery path for `supervise="off"`:

- non-interactive runs may still use `resume()` when stalled
- `min_rounds` remains handled only by the legacy recovery path
- `--interact` remains a single interactive optimize session

Do not let the unsupervised path accidentally inherit:

- `optimize_role="supervisor"`
- `skill_name="optimize-supervisor"`
- non-interactive supervisor forcing
- round-gate-specific prompt rewriting

- [ ] **Step 4: Run the focused supervisor tests to verify they pass**

Run: `uv run python -m unittest tests.test_supervisor tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 5: Commit the compatibility restoration**

```bash
git add src/triton_agent/supervisor.py src/triton_agent/optimize/runtime.py tests/test_supervisor.py tests/test_optimize_runtime.py
git commit -m "feat: restore legacy optimize flow when supervision is off"
```

## Task 5: Pass Supervise Mode Through Batch Optimize

**Files:**
- Modify: `src/triton_agent/optimize/batch.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the failing batch tests**

Add tests that prove batch requests preserve the selected mode:

```python
def test_optimize_batch_builds_unsupervised_requests_by_default(self) -> None:
    ...
    self.assertEqual(captured_request.supervise, "off")

def test_optimize_batch_builds_supervised_requests_when_requested(self) -> None:
    ...
    self.assertEqual(captured_request.supervise, "on")
```

- [ ] **Step 2: Run the focused batch tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_runtime -v`
Expected: FAIL because the request model does not yet carry the mode through batch execution.

- [ ] **Step 3: Implement the minimal batch threading**

Keep `run_optimize_batch()` structurally unchanged. Rely on `OptimizeRunOptions.supervise` flowing into `build_optimize_request()` and assert that per-workspace requests preserve the selected mode.

- [ ] **Step 4: Run the focused batch tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_runtime -v`
Expected: PASS

- [ ] **Step 5: Commit the batch propagation**

```bash
git add src/triton_agent/optimize/batch.py tests/test_cli.py tests/test_optimize_runtime.py
git commit -m "feat: thread supervise mode through batch optimize"
```

## Task 6: Document The New Mode And Run Full Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write the doc-facing assertions and examples**

Add or update tests where helpful so docs and behavior stay aligned, then update README examples:

```bash
uv run triton-agent optimize --input operator.py
uv run triton-agent optimize --input operator.py --supervise on
uv run triton-agent optimize-batch --input operators_root --supervise on
```

Make the default explicit in prose: ordinary optimize is unsupervised unless `--supervise on` is passed.

- [ ] **Step 2: Run focused checks before the full sweep**

Run:

```bash
uv run python -m unittest tests.test_cli tests.test_optimize_runtime tests.test_supervisor -v
uv run --group dev ruff check
```

Expected: PASS

- [ ] **Step 3: Run the full project verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```

Expected: PASS

- [ ] **Step 4: Commit the final mode documentation and verification pass**

```bash
git add README.md tests/test_cli.py tests/test_optimize_runtime.py tests/test_supervisor.py
git commit -m "docs: describe optimize supervise mode"
```
