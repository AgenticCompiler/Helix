# Optimize Round Strategy State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `ascend-npu-optimize-state` so optimize rounds carry explicit `round_strategy`, `analysis_policy`, and `reason` state in `.helix/state.json`, support mid-round updates through `set-current-round-state`, and mirror state changes into structured `opt-round-N/attempts.md` entries.

**Architecture:** Keep workflow-state authority inside `skills/common/ascend-npu-optimize-state/scripts/state_manage/`. `start-round` will become the initialization path for round strategy state, a new `set-current-round-state` subcommand will own same-round updates, and `workflow.py` will enforce enum validation, transition rules, and attempts-log writes. Runtime code remains a thin loader bridge through `src/helix/optimize/workflow_state.py`, while optimize skills and prompts are updated to treat workflow state as the authority and `attempts.md` as the structured history mirror.

**Tech Stack:** Python 3.11, `argparse`, `json`, `pathlib`, `tempfile`, `unittest`, `load_skill_script_module`, optimize-state skill scripts, optimize prompt builders, `uv`

---

## File Map

- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/workflow.py`
  Add strategy-state schemas, enum validation, transition validation, attempts-log append helpers, round initialization, and current-round update helpers.
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`
  Require `--round-strategy`, `--analysis-policy`, and `--reason`; emit the new success payload fields.
- Create: `skills/common/ascend-npu-optimize-state/scripts/state_manage/set_current_round_state.py`
  Implement the new CLI handler that updates the active round strategy state without `--round-dir`.
- Modify: `skills/common/ascend-npu-optimize-state/scripts/cli.py`
  Register and dispatch the new `set-current-round-state` subcommand.
- Modify: `skills/common/ascend-npu-optimize-state/SKILL.md`
  Document the new subcommand and the state-writing semantics.
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
  Explain that state blocks in `attempts.md` are script-written and that active strategy state is owned by `ascend-npu-optimize-state`.
- Modify: `skills/tilelang/tilelang-npu-optimize/SKILL.md`
  Mirror the same workflow-state and `attempts.md` guidance for TileLang.
- Modify: `src/helix/optimize/prompts.py`
  Update optimize guidance so `start-round` is treated as explicit state initialization and mid-round state changes use `set-current-round-state`.
- Modify: `tests/test_optimize_workflow_state.py`
  Add unit coverage for the new workflow helper signatures, legacy-session initialization, transition rules, and attempts-log mirroring.
- Modify: `tests/test_skill_command_script.py`
  Add CLI tests for the new required `start-round` arguments, the new `set-current-round-state` command, payloads, and `attempts.md` writes.
- Modify: `tests/test_generation_contracts.py`
  Update skill-contract expectations around the optimize-state skill surface and optimize skill wording if the tests pin those strings.
- Modify: `tests/test_cli.py`
  Add or update CLI-facing prompt text expectations if optimize prompt wording changes.

### Task 1: Lock the workflow helper contract with failing tests

**Files:**
- Modify: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_optimize_workflow_state.py`

- [ ] **Step 1: Add strategy-state initialization expectations to the existing `start_round()` tests**

```python
def test_start_round_records_strategy_state(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        state_path = workspace / ".helix" / "state.json"
        round_dir = workspace / "opt-round-1"
        attempts_path = round_dir / "attempts.md"
        state_path.parent.mkdir()
        round_dir.mkdir()
        module.bootstrap_state(
            state_path,
            run_id="optimize-20260627-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )

        module.start_round(
            state_path,
            round_dir.name,
            round_strategy="exploration",
            analysis_policy="pattern_entry",
            reason="Need to narrow the first promising direction.",
        )
        payload = json.loads(state_path.read_text(encoding="utf-8"))

    strategy_state = payload["rounds"]["1"]["strategy_state"]
    self.assertEqual(strategy_state["round_strategy"], "exploration")
    self.assertEqual(strategy_state["analysis_policy"], "pattern_entry")
    self.assertEqual(strategy_state["updated_by"], "start-round")
    self.assertTrue(attempts_path.is_file())
```

- [ ] **Step 2: Add a failing helper test for `set_current_round_state()` no-op rejection, rollback rejection, and successful update**

```python
def test_set_current_round_state_rejects_noop_and_policy_rollback(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        state_path = workspace / ".helix" / "state.json"
        round_dir = workspace / "opt-round-2"
        state_path.parent.mkdir()
        round_dir.mkdir()
        module.bootstrap_state(
            state_path,
            run_id="optimize-20260627-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )
        module.start_round(
            state_path,
            round_dir.name,
            round_strategy="structural_change",
            analysis_policy="profile_required",
            reason="Need profiler-backed structural evidence first.",
        )

        with self.assertRaisesRegex(ValueError, "state update would be a no-op"):
            module.set_current_round_state(
                state_path,
                round_strategy="structural_change",
                analysis_policy="profile_required",
                reason="same state",
            )

        with self.assertRaisesRegex(ValueError, "analysis_policy cannot become shallower"):
            module.set_current_round_state(
                state_path,
                round_strategy="focused_tuning",
                analysis_policy="pattern_entry",
                reason="rollback should fail",
            )
```

- [ ] **Step 3: Add a failing helper test that legacy active rounds without `strategy_state` can still be initialized**

```python
def test_set_current_round_state_initializes_missing_legacy_strategy_state(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        state_path = workspace / ".helix" / "state.json"
        round_dir = workspace / "opt-round-4"
        state_path.parent.mkdir()
        round_dir.mkdir()
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "optimize-20260627-123456-abcdef",
                    "phase": "round_active",
                    "source_operator": "kernel.py",
                    "current_round": 4,
                    "baseline": {"status": "passed", "submitted_at": "2026-06-27T12:34:56Z"},
                    "rounds": {
                        "4": {
                            "status": "active",
                            "round_dir": "opt-round-4",
                            "started_at": "2026-06-27T12:40:00Z",
                            "ended_at": None,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        module.set_current_round_state(
            state_path,
            round_strategy="stabilization",
            analysis_policy="ir_required",
            reason="Legacy active round needs explicit repair state.",
        )
        payload = json.loads(state_path.read_text(encoding="utf-8"))

    self.assertEqual(payload["rounds"]["4"]["strategy_state"]["round_strategy"], "stabilization")
```

- [ ] **Step 4: Add a failing helper test that verifies structured `attempts.md` state-update blocks are appended**

```python
def test_state_updates_append_structured_attempts_log_blocks(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        state_path = workspace / ".helix" / "state.json"
        round_dir = workspace / "opt-round-3"
        attempts_path = round_dir / "attempts.md"
        state_path.parent.mkdir()
        round_dir.mkdir()
        module.bootstrap_state(
            state_path,
            run_id="optimize-20260627-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )
        module.start_round(
            state_path,
            round_dir.name,
            round_strategy="exploration",
            analysis_policy="pattern_entry",
            reason="Start from pattern triage.",
        )
        module.set_current_round_state(
            state_path,
            round_strategy="structural_change",
            analysis_policy="profile_required",
            reason="Profiler evidence is now required before the main rewrite.",
        )

        attempts_text = attempts_path.read_text(encoding="utf-8")

    self.assertIn("## State Update", attempts_text)
    self.assertIn("Source: start-round", attempts_text)
    self.assertIn("Source: set-current-round-state", attempts_text)
    self.assertIn("Round strategy: exploration -> structural_change", attempts_text)
```

- [ ] **Step 5: Run the focused workflow helper tests to verify they fail before implementation**

Run:

```bash
uv run python -m unittest tests.test_optimize_workflow_state -v
```

Expected: FAIL because `workflow.py` does not yet accept strategy-state arguments, does not expose `set_current_round_state()`, and does not write structured `attempts.md` blocks.

### Task 2: Lock the optimize-state CLI surface with failing tests

**Files:**
- Modify: `tests/test_skill_command_script.py`
- Test: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add failing help and validation tests for the new `start-round` required arguments**

```python
def test_optimize_state_start_round_help_includes_strategy_arguments(self) -> None:
    completed = subprocess.run(
        [sys.executable, str(_OPTIMIZE_STATE_SCRIPT), "start-round", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    self.assertEqual(completed.returncode, 0)
    self.assertIn("--round-strategy", completed.stdout)
    self.assertIn("--analysis-policy", completed.stdout)
    self.assertIn("--reason", completed.stdout)
```

- [ ] **Step 2: Add a failing `start-round` success test that checks the echoed payload and attempts-log creation**

```python
def test_optimize_state_start_round_success_returns_strategy_state(self) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "start-round",
            "--round-dir",
            str(round_dir),
            "--round-strategy",
            "exploration",
            "--analysis-policy",
            "pattern_entry",
            "--reason",
            "Need to narrow the first promising direction.",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
        env=env,
    )
    payload = json.loads(completed.stdout)
    self.assertEqual(payload["round_strategy"], "exploration")
    self.assertEqual(payload["analysis_policy"], "pattern_entry")
    self.assertIn("Need to narrow", payload["reason"])
    self.assertTrue((round_dir / "attempts.md").is_file())
```

- [ ] **Step 3: Add failing tests for the new `set-current-round-state` command**

```python
def test_optimize_state_set_current_round_state_updates_active_round(self) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "set-current-round-state",
            "--round-strategy",
            "focused_tuning",
            "--analysis-policy",
            "ir_required",
            "--reason",
            "Need IR before the next code change.",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
        env=env,
    )
    payload = json.loads(completed.stdout)
    self.assertEqual(payload["status"], "pass")
    self.assertEqual(payload["round_strategy"]["from"], "structural_change")
    self.assertEqual(payload["round_strategy"]["to"], "focused_tuning")
```

- [ ] **Step 4: Add failing CLI tests for no active round, no-op update, and analysis-policy rollback**

```python
def test_optimize_state_set_current_round_state_rejects_noop_update(self) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "set-current-round-state",
            "--round-strategy",
            "structural_change",
            "--analysis-policy",
            "profile_required",
            "--reason",
            "same state",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=workspace,
        env=env,
    )
    payload = json.loads(completed.stdout)
    self.assertEqual(payload["status"], "fail")
    self.assertIn("no-op", payload["issues"][0])
```

- [ ] **Step 5: Run the focused optimize-state CLI tests to verify they fail before implementation**

Run:

```bash
uv run python -m unittest tests.test_skill_command_script.OptimizeStateCommandScriptTests -v
```

Expected: FAIL because `cli.py` does not register `set-current-round-state`, `start-round` does not accept the new required arguments, and the new success/failure payloads are not implemented.

### Task 3: Implement workflow-state mutation and CLI handlers

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/workflow.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`
- Create: `skills/common/ascend-npu-optimize-state/scripts/state_manage/set_current_round_state.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/cli.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/cli.py`
- Test: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_skill_command_script.py`

- [ ] **Step 1: Extend `workflow.py` with strategy-state enums, validation, and attempts-log append helpers**

```python
ROUND_STRATEGIES = {
    "exploration",
    "structural_change",
    "focused_tuning",
    "stabilization",
    "plateau_review",
}
ANALYSIS_POLICIES = {
    "pattern_entry",
    "profile_required",
    "ir_required",
    "compiler_source_required",
}


def _strategy_state_payload(
    *,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
    updated_by: str,
) -> dict[str, object]:
    return {
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "reason": reason,
        "updated_at": _utc_now(),
        "updated_by": updated_by,
    }
```

- [ ] **Step 2: Change `start_round()` to require and persist strategy state**

```python
def start_round(
    state_path: Path,
    round_dir: str,
    *,
    round_strategy: str,
    analysis_policy: str,
    reason: str,
) -> None:
    payload = load_state(state_path)
    ...
    rounds[round_key] = {
        "status": "active",
        "round_dir": round_dir,
        "started_at": _utc_now(),
        "ended_at": None,
        "strategy_state": _strategy_state_payload(
            round_strategy=round_strategy,
            analysis_policy=analysis_policy,
            reason=reason,
            updated_by="start-round",
        ),
    }
    _atomic_write_json(state_path, payload)
    _append_state_update_block(...)
```

- [ ] **Step 3: Add `set_current_round_state()` with no-op rejection, rollback rejection, and legacy initialization support**

```python
def set_current_round_state(
    state_path: Path,
    *,
    round_strategy: str | None,
    analysis_policy: str | None,
    reason: str,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    payload = load_state(state_path)
    round_entry = _require_active_round_entry(payload)
    previous = _existing_strategy_state(round_entry)
    ...
    if previous is not None and next_round_strategy == previous["round_strategy"] and next_analysis_policy == previous["analysis_policy"]:
        raise ValueError("state update would be a no-op")
    if _analysis_policy_rank(next_analysis_policy) < _analysis_policy_rank(previous_policy):
        raise ValueError("analysis_policy cannot become shallower within the same round")
```

- [ ] **Step 4: Implement the new CLI handler and wire it into `scripts/cli.py`**

```python
if command == "set-current-round-state":
    return set_current_round_state_cmd.main(
        [command, *remaining],
        prog_name=f"{parser.prog} {command}",
    )
```

- [ ] **Step 5: Extend success payload helpers so `start-round` and `set-current-round-state` can return structured state fields**

```python
def build_state_success_payload(
    *,
    round_name: str,
    guideline: str,
    round_strategy: object,
    analysis_policy: object,
    reason: str,
    warnings: list[str],
    hard_rules: list[str] | None = None,
) -> dict[str, object]:
    ...
```

- [ ] **Step 6: Run the focused tests to verify the implementation turns green**

Run:

```bash
uv run python -m unittest tests.test_optimize_workflow_state -v
uv run python -m unittest tests.test_skill_command_script.OptimizeStateCommandScriptTests -v
```

Expected: PASS, including `attempts.md` state-update mirroring and CLI payload assertions.

### Task 4: Update skill and prompt contracts, then run regression checks

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/SKILL.md`
- Modify: `src/helix/optimize/prompts.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_generation_contracts.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_skill_command_script.py`

- [ ] **Step 1: Update the optimize-state skill docs to document the new subcommand and the new `start-round` argument contract**

```md
python3 scripts/cli.py start-round --round-dir opt-round-1 --round-strategy exploration --analysis-policy pattern_entry --reason "..."
python3 scripts/cli.py set-current-round-state --round-strategy focused_tuning --analysis-policy ir_required --reason "..."
```

- [ ] **Step 2: Update Triton and TileLang optimize skill docs so state blocks in `attempts.md` are script-written and `summary.md` is no longer the state-history ledger**

```md
- Structured round strategy state updates in `opt-round-N/attempts.md` are written by the staged `ascend-npu-optimize-state` workflow commands.
- Do not manually duplicate the same `round_strategy`, `analysis_policy`, and `reason` bookkeeping in both `attempts.md` and `summary.md`.
```

- [ ] **Step 3: Update optimize prompts to mention explicit state initialization and mid-round state changes through the optimize-state skill**

```python
"Use the staged `ascend-npu-optimize-state` skill's `start-round` subcommand to open the next round and declare its `round_strategy` and `analysis_policy` before editing code.",
"If the round's intent or required evidence depth changes mid-round, use `set-current-round-state` instead of silently changing the round contract in prose only.",
```

- [ ] **Step 4: Add or update pinned wording tests**

```python
self.assertIn("set-current-round-state", optimize_state_skill)
self.assertIn("declare its `round_strategy` and `analysis_policy`", prompt)
```

- [ ] **Step 5: Run the focused contract and prompt regressions**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
uv run python -m unittest tests.test_cli -v
```

Expected: PASS with updated CLI help and skill-doc wording.

- [ ] **Step 6: Run the repository-standard verification commands for the touched area**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS without new lint, type, or test regressions.
