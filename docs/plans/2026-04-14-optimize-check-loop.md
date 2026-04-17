# Optimize Check Loop Implementation Plan

> **Historical note:** This plan predates the final removal of `skills/optimize-supervisor/`. Read remaining supervisor-skill references as historical implementation context only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `triton-npu-optimize-check` skill plus a simplified optimize loop where workers validate baseline and each round themselves, and supervisor becomes an optional metadata-only audit layer.

**Architecture:** Introduce one shared triton-npu-optimize-check contract in Python so skill scripts, worker prompts, and supervisor logic all reuse the same baseline and round validation rules. Keep `--supervisor off` as a long-running worker-owned session, keep `--supervisor on` as a one-round worker plus optional supervisor loop, and remove runtime-owned technical gate decisions that duplicate skill behavior.

**Tech Stack:** Python `argparse`, `dataclasses`, existing optimize baseline/round helpers, workspace skills and scripts, Python `unittest`

---

## File Structure

**New files**

- `docs/plans/2026-04-14-triton-npu-optimize-check-loop.md`
  This implementation plan.
- `src/triton_agent/optimize/checks.py`
  Shared baseline and round check API that returns structured results for scripts, runtime helpers, and tests.
- `skills/triton-npu-optimize-check/SKILL.md`
  Workflow contract for baseline and round checking.
- `skills/triton-npu-optimize-check/scripts/optimize_check.py`
  Standalone script entrypoint with `check-baseline` and `check-round` subcommands.
- `tests/test_optimize_checks.py`
  Focused unit tests for shared triton-npu-optimize-check behavior.

**Existing files to modify**

- `src/triton_agent/optimize/models.py`
  Add structured triton-npu-optimize-check result models if they do not fit cleanly in `checks.py`.
- `src/triton_agent/prompts.py`
  Update worker and supervisor prompts to require triton-npu-optimize-check usage and clarify per-mode ownership.
- `src/triton_agent/optimize/orchestration.py`
  Simplify orchestration around worker-owned checks and optional supervisor passes.
- `src/triton_agent/optimize/run_loop.py`
  Narrow supervised loop behavior to worker launch, supervisor audit, and continue-or-stop handling.
- `src/triton_agent/optimize/guidance.py`
  Keep only the guidance files needed for the simplified supervised handoff model.
- `skills/triton-npu-optimize/SKILL.md`
  Require baseline and round checks in both optimize modes.
- `skills/optimize-supervisor/SKILL.md`
  Limit supervisor to metadata-only repair and explicit continuation decisions.
- `src/triton_agent/skills.py`
  Ensure `triton-npu-optimize-check` stages alongside optimize-related skills where needed.
- `tests/test_cli.py`
  Extend prompt and request-construction assertions for the new worker and supervisor contracts.
- `tests/test_optimize_runtime.py`
  Replace runtime-side gate expectations with worker-self-check and supervisor-loop expectations.
- `tests/test_supervisor.py`
  Update loop tests to reflect supervisor as audit plus continue-or-stop control, not artifact gate evaluator.
- `tests/test_skills.py`
  Cover staging of `triton-npu-optimize-check`.
- `tests/test_skill_command_script.py`
  Cover `optimize_check.py --help` and direct script execution without installed console entrypoints.
- `README.md`
  Document `triton-npu-optimize-check`, worker-owned validation, and the updated supervised versus unsupervised semantics.

## Task 1: Add The Shared Optimize-Check Contract And Script

**Files:**
- Create: `src/triton_agent/optimize/checks.py`
- Create: `skills/triton-npu-optimize-check/SKILL.md`
- Create: `skills/triton-npu-optimize-check/scripts/optimize_check.py`
- Modify: `src/triton_agent/optimize/models.py`
- Create: `tests/test_optimize_checks.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Write the failing check-layer tests**

Add focused tests that pin the shared result shape and the two check entrypoints before implementation:

```python
def test_check_baseline_reports_missing_perf_artifact(self) -> None:
    result = check_baseline(workdir / "baseline")
    self.assertFalse(result.ok)
    self.assertEqual(result.kind, "baseline")
    self.assertEqual(result.decision, "revise-required")
    self.assertIn("missing perf artifact", result.issues)

def test_check_round_passes_with_complete_round_artifacts(self) -> None:
    result = check_round(workdir / "opt-round-1")
    self.assertTrue(result.ok)
    self.assertEqual(result.kind, "round")
    self.assertEqual(result.decision, "pass")
```

Also add script tests like:

```python
completed = subprocess.run([sys.executable, str(script), "--help"], ...)
self.assertIn("check-baseline", completed.stdout)
self.assertIn("check-round", completed.stdout)
```

and a staging test that asserts `triton-npu-optimize-check` is copied into backend-native skills directories.

- [ ] **Step 2: Run the focused check and script tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_checks tests.test_skill_command_script tests.test_skills -v`
Expected: FAIL because the shared triton-npu-optimize-check layer and skill do not exist yet.

- [ ] **Step 3: Implement the shared check API and thin script wrapper**

Create a small structured result type and keep the real logic in Python modules, not in shell text parsing:

```python
@dataclass(frozen=True)
class OptimizeCheckResult:
    ok: bool
    kind: Literal["baseline", "round"]
    decision: Literal["pass", "revise-required", "hard-fail"]
    issues: tuple[str, ...]
    summary: str
```

Guidelines:

- Reuse existing baseline helpers from `src/triton_agent/optimize/baseline.py`.
- Reuse existing round helpers from `src/triton_agent/optimize/round_contract.py`.
- Reuse or adapt existing round validation from `src/triton_agent/optimize/gate.py` instead of re-encoding rules in the script.
- Make `optimize_check.py` a thin CLI wrapper that prints machine-readable JSON plus a concise human-readable summary.
- Ensure the script works when run directly with `python`, without assuming the `triton-agent` console entrypoint is installed.

- [ ] **Step 4: Run the focused check and script tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_checks tests.test_skill_command_script tests.test_skills -v`
Expected: PASS

- [ ] **Step 5: Commit the triton-npu-optimize-check foundation**

```bash
git add src/triton_agent/optimize/checks.py src/triton_agent/optimize/models.py skills/triton-npu-optimize-check/SKILL.md skills/triton-npu-optimize-check/scripts/optimize_check.py tests/test_optimize_checks.py tests/test_skill_command_script.py tests/test_skills.py
git commit -m "feat: add optimize check skill and script"
```

## Task 2: Update Worker And Supervisor Contracts Around Optimize-Check

**Files:**
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/optimize-supervisor/SKILL.md`
- Modify: `src/triton_agent/prompts.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing prompt and skill-contract tests**

Add tests that lock the behavioral wording before changing prompts:

```python
def test_build_optimize_unsupervised_prompt_requires_round_checks(self) -> None:
    prompt = build_optimize_unsupervised_prompt(...)
    self.assertIn("check-baseline", prompt)
    self.assertIn("check-round", prompt)
    self.assertIn("continue optimizing until the session should stop", prompt)

def test_build_optimize_worker_prompt_requires_single_round_check_before_exit(self) -> None:
    prompt = build_optimize_worker_prompt(...)
    self.assertIn("owns exactly one optimization round", prompt)
    self.assertIn("must pass `check-round` before the invocation ends", prompt)

def test_build_optimize_supervisor_prompt_limits_repairs_to_metadata(self) -> None:
    prompt = build_optimize_supervisor_prompt(...)
    self.assertIn("Do not edit the operator implementation", prompt)
```

- [ ] **Step 2: Run the focused prompt tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: FAIL because current prompts still describe runtime-owned gate behavior and do not require triton-npu-optimize-check usage.

- [ ] **Step 3: Update the optimize and optimize-supervisor contracts**

In `skills/triton-npu-optimize/SKILL.md`, make both modes require:

- baseline validation through `triton-npu-optimize-check check-baseline`
- round validation through `triton-npu-optimize-check check-round`
- repair-and-recheck behavior before moving on

In `skills/optimize-supervisor/SKILL.md`, narrow the contract to:

- review the already-validated round
- repair only metadata, briefs, summaries, and session notes derived from existing facts
- decide `continue` or `stop`

In `src/triton_agent/prompts.py`, align the public prompt text with those contracts and avoid mixing worker self-check rules with runtime gate wording.

- [ ] **Step 4: Run the focused prompt tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli -v`
Expected: PASS for the updated prompt semantics and any existing optimize prompt regressions.

- [ ] **Step 5: Commit the prompt and skill contract changes**

```bash
git add skills/triton-npu-optimize/SKILL.md skills/optimize-supervisor/SKILL.md src/triton_agent/prompts.py tests/test_cli.py
git commit -m "feat: require optimize self-checks in worker prompts"
```

## Task 3: Simplify Runtime And Supervisor Around Worker-Owned Checks

**Files:**
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/optimize/run_loop.py`
- Modify: `src/triton_agent/optimize/guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write the failing orchestration tests**

Add tests that pin the new loop semantics:

```python
def test_unsupervised_optimize_runs_single_worker_session_without_supervisor_pass(self) -> None:
    request = make_request(supervise="off")
    result = run_optimize_request(request)
    self.assertEqual(result.return_code, 0)
    self.assertEqual([call.optimize_role for call in runner.calls], [None])

def test_supervised_optimize_runs_worker_then_supervisor_without_runtime_gate_eval(self) -> None:
    request = make_request(supervise="on")
    result = run_optimize_request(request)
    self.assertEqual(result.return_code, 0)
    self.assertEqual([call.optimize_role for call in runner.requests], ["worker", "supervisor"])

def test_supervised_loop_relaunches_worker_when_supervisor_requests_continue(self) -> None:
    result = OptimizeRunLoop().run(runner, request)
    self.assertEqual(runner.events, ["worker-run", "supervisor-run", "worker-run"])
```

Also add assertions that supervised runtime no longer turns missing artifacts into a local `evaluate_round_gate()` decision after the supervisor invocation.

- [ ] **Step 2: Run the focused orchestration tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_supervisor -v`
Expected: FAIL because current supervised runtime still evaluates round gates itself and current supervisor loop still carries runtime-owned gate summaries.

- [ ] **Step 3: Refactor runtime and supervisor to match the new mode semantics**

Implementation guidelines:

- Keep `--supervisor off` as one long-running worker-owned optimize session.
- Keep `--supervisor on` as:
  1. one worker invocation
  2. one supervisor invocation
  3. continue-or-stop decision
- Remove runtime-owned technical gate decisions that duplicate `triton-npu-optimize-check`.
- Keep `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md` only for supervised handoff.
- Let supervisor parse and emit a small decision artifact instead of requiring runtime to infer the decision from workspace state.

When adjusting `src/triton_agent/optimize/run_loop.py`, preserve useful retry and stall-recovery behavior, but scope it to orchestration concerns rather than round artifact policy.

- [ ] **Step 4: Run the focused orchestration tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_runtime tests.test_supervisor -v`
Expected: PASS

- [ ] **Step 5: Commit the loop simplification**

```bash
git add src/triton_agent/optimize/orchestration.py src/triton_agent/optimize/run_loop.py src/triton_agent/optimize/guidance.py tests/test_optimize_runtime.py tests/test_supervisor.py
git commit -m "refactor: simplify optimize supervisor loop"
```

## Task 4: Document The New Workflow And Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Update user-facing optimize documentation**

Document:

- what `triton-npu-optimize-check` is
- that workers validate baseline and each round themselves
- that `--supervisor off` means one worker owns the full session
- that `--supervisor on` means worker owns one round and supervisor owns the outer loop
- that supervisor repairs are metadata-only

Include at least one concrete optimize example that mentions the supervised and unsupervised difference.

- [ ] **Step 2: Run the focused optimize verification suite**

Run: `uv run python -m unittest tests.test_cli tests.test_optimize_checks tests.test_optimize_runtime tests.test_supervisor tests.test_skill_command_script tests.test_skills -v`
Expected: PASS

- [ ] **Step 3: Run repository verification required by project policy**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 4: Review the diff for accidental runtime policy leakage**

Manually verify:

- optimize technical checks live in shared check helpers and skill scripts
- prompts do not tell supervisor to modify operator code
- runtime does not silently reintroduce artifact-based round gating outside `triton-npu-optimize-check`
- docs describe worker-owned validation accurately

- [ ] **Step 5: Commit docs and final verification updates**

```bash
git add README.md tests/test_cli.py tests/test_optimize_checks.py tests/test_optimize_runtime.py tests/test_supervisor.py tests/test_skill_command_script.py tests/test_skills.py
git commit -m "docs: describe optimize self-check workflow"
```
