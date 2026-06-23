# Optimize Workflow State Phase Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hook-gated optimize workflow state under `.triton-agent/state.json`, track per-round start/end timestamps, and archive completed round timings into `triton-agent-logs/<run-id>/round-timings.json` without changing non-hook optimize behavior.

**Architecture:** Keep workflow-state authority in a skill-side helper under `skills/triton-npu-optimize/scripts/`, then load it from runtime through a thin `src/` bridge. Baseline submit, round submit, and the new start-round script mutate state automatically when the runtime bootstrapped the state file, while runtime owns hook-gated bootstrap, prompt-phase summary rendering, and cleanup-time archive projection.

**Tech Stack:** Python 3.11, `argparse`, `json`, `pathlib`, `tempfile`, `unittest`, `load_skill_script_module`, optimize runtime/session-artifact code, skill-script pyright wrapper, `uv`

**Implementation note:** Do not create commits unless the user explicitly asks for them.

---

## File Map

- Create: `skills/triton-npu-optimize/scripts/optimize_workflow_state.py`
  Own JSON loading, validation, legal phase transitions, UTC timestamp generation, atomic writes, phase-summary rendering, and `round-timings.json` projection.
- Create: `skills/triton-npu-optimize-start-round/scripts/optimize_start_round.py`
  Provide the `start-round` CLI that opens one round in workflow state immediately before the next optimize round begins.
- Create: `src/triton_agent/optimize/workflow_state.py`
  Runtime-only bridge that loads the skill helper via `load_skill_script_module` and exposes typed wrappers for bootstrap, prompt summaries, and cleanup-time archive writing.
- Modify: `skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline.py`
  Keep pure validation when workflow state is absent, but advance state automatically when `.triton-agent/state.json` is present.
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py`
  Keep pure validation when workflow state is absent, but enforce active-round matching and complete the round automatically when state is present.
- Modify: `skills/triton-npu-optimize-start-round/SKILL.md`
  Tell the agent to call the new script directly before editing the next round.
- Modify: `src/triton_agent/optimize/session_artifacts.py`
  Hook-gate workflow-state bootstrap and cleanup, including checked-mode `.triton-agent/` creation when hooks are enabled.
- Modify: `src/triton_agent/optimize/archive.py`
  Reserve `round-timings.json` in the existing archive namespace and keep archive warnings aligned with current behavior.
- Modify: `src/triton_agent/optimize/execution.py`
  Pass `enable_agent_hooks` into artifact preparation and inject prompt summaries derived from workflow state before worker/supervisor launches.
- Modify: `src/triton_agent/optimize/prompts.py`
  Accept a rendered workflow-phase summary on baseline, round, and supervisor prompt builders.
- Modify: `src/triton_agent/prompts.py`
  Thread the optional workflow-phase summary through the optimize prompt dispatch path.
- Create: `tests/test_optimize_workflow_state.py`
  Focused helper/bridge tests for schema validation, transitions, idempotency, and round-timing projection.
- Modify: `tests/test_skill_command_script.py`
  CLI tests for `optimize_submit_baseline.py`, `optimize_submit_round.py`, and `optimize_start_round.py`.
- Modify: `tests/test_optimize_guidance.py`
  Session-artifact/bootstrap/archive tests for checked/supervised cleanup behavior.
- Modify: `tests/test_optimize_runtime.py`
  Prompt-injection and runtime hook-gating tests.

### Task 1: Lock the workflow-state helper contract with failing tests

**Files:**
- Create: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_optimize_workflow_state.py`

- [ ] **Step 1: Add a dedicated test module that loads the skill helper through the same bridge the runtime will use**

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skill_loader import load_skill_script_module


def load_workflow_state_module():
    return load_skill_script_module("triton-npu-optimize", "optimize_workflow_state")
```

- [ ] **Step 2: Add a failing bootstrap/validation test for the baseline phase and the reusable-baseline shortcut**

```python
def test_bootstrap_state_writes_expected_hook_gated_baseline_payload(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state_path = root / ".triton-agent" / "state.json"
        state_path.parent.mkdir()

        module.bootstrap_state(
            state_path,
            run_id="optimize-20260622-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=False,
        )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["phase"], "baseline")
        self.assertEqual(payload["baseline"], {"status": "pending", "submitted_at": None})

        module.bootstrap_state(
            state_path,
            run_id="optimize-20260622-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["phase"], "awaiting_round_start")
        self.assertEqual(payload["baseline"], {"status": "passed", "submitted_at": None})
```

- [ ] **Step 3: Add a failing transition/idempotency test for `start_round()`**

```python
def test_start_round_is_idempotent_for_same_active_round(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / ".triton-agent" / "state.json"
        state_path.parent.mkdir()
        module.bootstrap_state(
            state_path,
            run_id="optimize-20260622-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )

        module.start_round(state_path, "opt-round-1")
        first_text = state_path.read_text(encoding="utf-8")
        module.start_round(state_path, "opt-round-1")
        second_text = state_path.read_text(encoding="utf-8")

    self.assertEqual(first_text, second_text)
```

- [ ] **Step 4: Add failing completion and archive-projection tests**

```python
def test_complete_round_records_end_time_and_resets_phase(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / ".triton-agent" / "state.json"
        state_path.parent.mkdir()
        module.bootstrap_state(
            state_path,
            run_id="optimize-20260622-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )
        module.start_round(state_path, "opt-round-1")
        module.complete_round(state_path, "opt-round-1", current_round_arg=1)
        payload = json.loads(state_path.read_text(encoding="utf-8"))

    self.assertEqual(payload["phase"], "awaiting_round_start")
    self.assertIsNone(payload["current_round"])
    self.assertEqual(payload["rounds"]["1"]["status"], "passed")
    self.assertIsNotNone(payload["rounds"]["1"]["ended_at"])


def test_write_round_timings_archive_only_includes_passed_rounds(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state_path = root / ".triton-agent" / "state.json"
        archive_path = root / "triton-agent-logs" / "optimize-20260622" / "round-timings.json"
        state_path.parent.mkdir()
        module.bootstrap_state(
            state_path,
            run_id="optimize-20260622-123456-abcdef",
            source_operator="kernel.py",
            baseline_reused=True,
        )
        module.start_round(state_path, "opt-round-1")
        module.complete_round(state_path, "opt-round-1", current_round_arg=1)
        module.start_round(state_path, "opt-round-2")

        wrote = module.write_round_timings_archive(state_path, archive_path)
        payload = json.loads(archive_path.read_text(encoding="utf-8"))

    self.assertTrue(wrote)
    self.assertEqual(payload, [{"round": 1, "started_at": payload[0]["started_at"], "ended_at": payload[0]["ended_at"]}])
```

- [ ] **Step 5: Add failing invalid-input tests for malformed JSON, unsupported schema version, and passed-round missing `ended_at`**

```python
def test_load_state_rejects_unknown_schema_version(self) -> None:
    module = load_workflow_state_module()
    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / ".triton-agent" / "state.json"
        state_path.parent.mkdir()
        state_path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported workflow state schema_version"):
            module.load_state(state_path)
```

- [ ] **Step 6: Run the focused helper tests to verify they fail before implementation**

Run:

```bash
uv run python -m unittest tests.test_optimize_workflow_state -v
```

Expected: FAIL because `skills/triton-npu-optimize/scripts/optimize_workflow_state.py` does not exist yet and none of the transition/archive helpers are implemented.

### Task 2: Implement the skill-side helper and the runtime bridge

**Files:**
- Create: `skills/triton-npu-optimize/scripts/optimize_workflow_state.py`
- Create: `src/triton_agent/optimize/workflow_state.py`
- Test: `tests/test_optimize_workflow_state.py`

- [ ] **Step 1: Add the canonical helper module with JSON validation, round parsing, and atomic writes**

```python
WORKFLOW_SCHEMA_VERSION = 1
PHASE_BASELINE = "baseline"
PHASE_AWAITING_ROUND_START = "awaiting_round_start"
PHASE_ROUND_ACTIVE = "round_active"


def bootstrap_state(
    state_path: Path,
    *,
    run_id: str,
    source_operator: str,
    baseline_reused: bool,
) -> None:
    payload = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "run_id": run_id,
        "phase": PHASE_AWAITING_ROUND_START if baseline_reused else PHASE_BASELINE,
        "source_operator": source_operator,
        "current_round": None,
        "baseline": {
            "status": "passed" if baseline_reused else "pending",
            "submitted_at": None,
        },
        "rounds": {},
    }
    _atomic_write_json(state_path, payload)
```

- [ ] **Step 2: Implement the three state transitions exactly as approved**

```python
def mark_baseline_passed(state_path: Path) -> None:
    payload = load_state(state_path)
    payload["baseline"] = {"status": "passed", "submitted_at": _utc_now()}
    payload["phase"] = PHASE_AWAITING_ROUND_START
    payload["current_round"] = None
    _atomic_write_json(state_path, payload)


def start_round(state_path: Path, round_dir: str) -> None:
    payload = load_state(state_path)
    round_number = _parse_round_number(round_dir)
    _require_can_start_round(payload, round_number)
    if _is_same_active_round(payload, round_number, round_dir):
        return
    payload["phase"] = PHASE_ROUND_ACTIVE
    payload["current_round"] = round_number
    payload["rounds"][str(round_number)] = {
        "status": "active",
        "round_dir": round_dir,
        "started_at": _utc_now(),
        "ended_at": None,
    }
    _atomic_write_json(state_path, payload)


def complete_round(state_path: Path, round_dir: str, current_round_arg: int | None) -> None:
    payload = load_state(state_path)
    round_number = _parse_round_number(round_dir)
    _require_matching_active_round(payload, round_number, current_round_arg)
    payload["rounds"][str(round_number)]["status"] = "passed"
    payload["rounds"][str(round_number)]["ended_at"] = _utc_now()
    payload["phase"] = PHASE_AWAITING_ROUND_START
    payload["current_round"] = None
    _atomic_write_json(state_path, payload)
```

- [ ] **Step 3: Implement prompt-summary rendering and the minimal completed-round archive projection**

```python
def render_phase_summary(state_path: Path) -> str:
    payload = load_state(state_path)
    baseline = payload["baseline"]
    reused = baseline["status"] == "passed" and baseline["submitted_at"] is None
    lines = [
        f"Current phase: {payload['phase']}",
        f"Current round: {payload['current_round']}" if payload["current_round"] is not None else "Current round: none",
        f"Baseline source: {'reused' if reused else 'freshly passed in this run' if baseline['status'] == 'passed' else 'pending'}",
        f"Workflow state path: {state_path.as_posix()}",
    ]
    return "\n".join(lines)


def write_round_timings_archive(state_path: Path, archive_path: Path) -> bool:
    payload = load_state(state_path)
    rows = [
        {
            "round": int(round_key),
            "started_at": round_state["started_at"],
            "ended_at": round_state["ended_at"],
        }
        for round_key, round_state in sorted(payload["rounds"].items(), key=lambda item: int(item[0]))
        if round_state["status"] == "passed"
    ]
    if not rows:
        return False
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8")
    return True
```

- [ ] **Step 4: Add the runtime bridge that uses the existing skill-loader instead of importing the skill helper directly**

```python
from triton_agent.skill_loader import load_skill_script_module


def _workflow_module():
    return load_skill_script_module("triton-npu-optimize", "optimize_workflow_state")


def bootstrap_optimize_workflow_state(state_path: Path, *, run_id: str, source_operator: Path, baseline_reused: bool) -> None:
    _workflow_module().bootstrap_state(
        state_path,
        run_id=run_id,
        source_operator=source_operator.name,
        baseline_reused=baseline_reused,
    )


def render_optimize_phase_summary(state_path: Path | None) -> str | None:
    if state_path is None or not state_path.exists():
        return None
    return str(_workflow_module().render_phase_summary(state_path))


def archive_round_timings_from_state(state_path: Path | None, archive_path: Path) -> bool:
    if state_path is None or not state_path.exists():
        return False
    return bool(_workflow_module().write_round_timings_archive(state_path, archive_path))
```

- [ ] **Step 5: Re-run the helper tests and make them pass**

Run:

```bash
uv run python -m unittest tests.test_optimize_workflow_state -v
```

Expected: PASS

### Task 3: Wire baseline submit, round submit, and start-round onto the helper

**Files:**
- Modify: `skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline.py`
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py`
- Create: `skills/triton-npu-optimize-start-round/scripts/optimize_start_round.py`
- Modify: `skills/triton-npu-optimize-start-round/SKILL.md`
- Modify: `tests/test_skill_command_script.py`
- Test: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add failing CLI tests for hook-gated state mutation and the new start-round script**

```python
def test_optimize_start_round_script_help_runs_without_installed_entrypoint(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "triton-npu-optimize-start-round"
        / "scripts"
        / "optimize_start_round.py"
    )
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False, env=env)
    self.assertEqual(completed.returncode, 0)
    self.assertIn("start-round", completed.stdout)
```

```python
def test_optimize_submit_baseline_updates_workflow_state_when_present(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "triton-npu-optimize-submit-baseline"
        / "scripts"
        / "optimize_submit_baseline.py"
    )
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    script_dir = str(script.parent)
    env["PYTHONPATH"] = ":".join(
        entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
    )

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        baseline_dir = workspace / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "prepared",
                    "source_operator": "kernel.py",
                    "baseline_operator": "baseline/kernel.py",
                    "test_file": "differential_test_kernel.py",
                    "test_mode": "differential",
                    "bench_file": "bench_kernel.py",
                    "bench_mode": "torch-npu-profiler",
                    "perf_artifact": "baseline/perf.txt",
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
    (workspace / ".triton-agent").mkdir()
    (workspace / ".triton-agent" / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "optimize-20260622-123456-abcdef",
                "phase": "baseline",
                "source_operator": "kernel.py",
                "current_round": None,
                "baseline": {"status": "pending", "submitted_at": None},
                "rounds": {},
            }
        ),
        encoding="utf-8",
    )
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "check-baseline",
                "--baseline-dir",
                str(baseline_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=workspace,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
    state_payload = json.loads((workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8"))
    self.assertEqual(state_payload["phase"], "awaiting_round_start")
    self.assertEqual(state_payload["baseline"]["status"], "passed")
```

```python
def test_optimize_submit_round_updates_workflow_state_when_present(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "triton-npu-optimize-submit-round"
        / "scripts"
        / "optimize_submit_round.py"
    )
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    script_dir = str(script.parent)
    env["PYTHONPATH"] = ":".join(
        entry for entry in (src_dir, script_dir, env.get("PYTHONPATH", "")) if entry
    )

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        baseline_dir = workspace / "baseline"
        round_dir = workspace / "opt-round-4"
        baseline_dir.mkdir()
        round_dir.mkdir()
        (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        (workspace / "opt-note.md").write_text("## Round\n", encoding="utf-8")
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "prepared",
                    "source_operator": "kernel.py",
                    "baseline_operator": "baseline/kernel.py",
                    "test_file": "differential_test_kernel.py",
                    "test_mode": "differential",
                    "bench_file": "bench_kernel.py",
                    "bench_mode": "torch-npu-profiler",
                    "perf_artifact": "baseline/perf.txt",
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
        (round_dir / "opt_kernel.py").write_text("print('round')\n", encoding="utf-8")
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "opt_kernel_perf.txt").write_text("latency-a: 0.9\n", encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": "opt-round-4",
                    "parent_round": "round-3",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "comparison_target": "baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            ),
            encoding="utf-8",
        )
        (workspace / ".triton-agent").mkdir()
        (workspace / ".triton-agent" / "state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "optimize-20260622-123456-abcdef",
                    "phase": "round_active",
                    "source_operator": "kernel.py",
                    "current_round": 4,
                    "baseline": {"status": "passed", "submitted_at": "2026-06-22T12:34:56Z"},
                    "rounds": {
                        "4": {
                            "status": "active",
                            "round_dir": "opt-round-4",
                            "started_at": "2026-06-22T12:40:00Z",
                            "ended_at": None,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "check-round",
                "--round-dir",
                str(round_dir),
                "--current-round",
                "4",
                "--final-round",
                "25",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=workspace,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        state_payload = json.loads((workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8"))
    self.assertEqual(state_payload["phase"], "awaiting_round_start")
    self.assertIsNone(state_payload["current_round"])
    self.assertEqual(state_payload["rounds"]["4"]["status"], "passed")
```

- [ ] **Step 2: Add the shared cross-skill import helper inside the baseline and round submit scripts**

```python
def _load_workflow_state_module():
    skills_root = Path(__file__).resolve().parents[2]
    shared_scripts_dir = skills_root / "triton-npu-optimize" / "scripts"
    module_path = shared_scripts_dir / "optimize_workflow_state.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing optimize workflow helper: {module_path}")
    inserted = False
    shared_path = str(shared_scripts_dir)
    if shared_path not in sys.path:
        sys.path.insert(0, shared_path)
        inserted = True
    try:
        return importlib.import_module("optimize_workflow_state")
    finally:
        if inserted:
            sys.path.remove(shared_path)
```

- [ ] **Step 3: Update `optimize_submit_baseline.py` so hook-enabled sessions advance state automatically**

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    result = check_baseline(baseline_dir)
    state_path = baseline_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass" and state_path.exists():
        _load_workflow_state_module().mark_baseline_passed(state_path)
    print(json.dumps(_build_cli_payload(result), ensure_ascii=True))
    return 0 if result.status == "pass" else 1
```

- [ ] **Step 4: Update `optimize_submit_round.py` so hook-enabled sessions validate the active round and close it on success**

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    round_dir = Path(args.round_dir).expanduser().resolve()
    result = check_round(
        round_dir,
        current_round=args.current_round,
        final_round=args.final_round,
        optimize_target=args.optimize_target,
    )
    state_path = round_dir.parent / ".triton-agent" / "state.json"
    if result.status == "pass" and state_path.exists():
        _load_workflow_state_module().complete_round(
            state_path,
            round_dir.name,
            current_round_arg=args.current_round,
        )
    print(json.dumps(_build_cli_payload(result), ensure_ascii=True))
    return 0 if result.status == "pass" else 1
```

- [ ] **Step 5: Create `optimize_start_round.py` and teach the skill to call it**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start-round")
    start.add_argument("--round-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    round_dir = Path(args.round_dir).expanduser().resolve()
    state_path = round_dir.parent / ".triton-agent" / "state.json"
    if not state_path.exists():
        raise RuntimeError("optimize workflow state is not available; start-round requires --enable-agent-hook")
    _load_workflow_state_module().start_round(state_path, round_dir.name)
    print(json.dumps({"status": "pass", "round": round_dir.name}, ensure_ascii=True))
    return 0
```

and add this explicit command to `skills/triton-npu-optimize-start-round/SKILL.md`:

```markdown
Run:

```bash
python3 scripts/optimize_start_round.py start-round --round-dir opt-round-1
```
```

- [ ] **Step 6: Run the targeted script tests and the required skill-script pyright checks**

Run:

```bash
uv run python -m unittest tests.test_skill_command_script -v
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize/scripts/optimize_workflow_state.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-start-round/scripts/optimize_start_round.py
```

Expected: PASS

### Task 4: Hook-gate runtime bootstrap, prompt summaries, and cleanup-time `round-timings.json`

**Files:**
- Create: `src/triton_agent/optimize/workflow_state.py`
- Modify: `src/triton_agent/optimize/session_artifacts.py`
- Modify: `src/triton_agent/optimize/archive.py`
- Modify: `src/triton_agent/optimize/execution.py`
- Modify: `src/triton_agent/optimize/prompts.py`
- Modify: `src/triton_agent/prompts.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add failing session-artifact tests for checked-mode hook gating and cleanup archive projection**

```python
def test_prepare_checked_session_bootstraps_workflow_state_only_when_hooks_enabled(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
        manager = OptimizeSessionArtifactsManager()

        state_without_hooks = manager.prepare_checked_session(
            workdir,
            agent_name="codex",
            enable_agent_hooks=False,
            source_operator_path=workdir / "kernel.py",
        )
        self.assertIsNone(state_without_hooks.workflow_state_path)
        self.assertFalse((workdir / ".triton-agent").exists())

        state_with_hooks = manager.prepare_checked_session(
            workdir,
            agent_name="codex",
            enable_agent_hooks=True,
            source_operator_path=workdir / "kernel.py",
        )
        self.assertEqual(state_with_hooks.workflow_state_path, workdir / ".triton-agent" / "state.json")
        self.assertTrue(state_with_hooks.workflow_state_path.exists())
```

```python
def test_cleanup_supervised_session_writes_round_timings_archive_for_passed_rounds(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        (workdir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
        manager = OptimizeSessionArtifactsManager()
        state = manager.prepare_supervised_session(
            workdir,
            agent_name="codex",
            enable_agent_hooks=True,
            source_operator_path=workdir / "kernel.py",
        )
        assert state.workflow_state_path is not None
        state.workflow_state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": state.archive.run_id,
                    "phase": "round_active",
                    "source_operator": "kernel.py",
                    "current_round": 2,
                    "baseline": {"status": "passed", "submitted_at": "2026-06-22T12:34:56Z"},
                    "rounds": {
                        "1": {
                            "status": "passed",
                            "round_dir": "opt-round-1",
                            "started_at": "2026-06-22T12:40:00Z",
                            "ended_at": "2026-06-22T12:55:00Z",
                        },
                        "2": {
                            "status": "active",
                            "round_dir": "opt-round-2",
                            "started_at": "2026-06-22T13:10:00Z",
                            "ended_at": None,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        warnings = manager.cleanup_supervised_session(state)
        self.assertEqual(warnings, [])
    payload = json.loads((state.run_archive_dir / "round-timings.json").read_text(encoding="utf-8"))
    self.assertEqual([row["round"] for row in payload], [1])
```

- [ ] **Step 2: Add failing prompt/runtime tests for phase-summary injection only when hooks are enabled**

```python
def test_build_optimize_round_prompt_includes_phase_summary_when_present(self) -> None:
    prompt = build_optimize_round_prompt(
        Path("kernel.py"),
        None,
        test_mode="differential",
        bench_mode="torch-npu-profiler",
        round_mode="checked",
        current_round=1,
        final_round=1,
        workflow_phase_summary="Current phase: round_active\nCurrent round: 1",
    )
    self.assertIn("Workflow phase summary:", prompt)
    self.assertIn("Current phase: round_active", prompt)


def test_build_prompt_omits_phase_summary_when_not_provided(self) -> None:
    prompt = build_prompt(
        CommandKind.OPTIMIZE,
        Path("kernel.py"),
        Path("kernel.py"),
        None,
        "differential",
        "torch-npu-profiler",
        False,
    )
    self.assertNotIn("Workflow phase summary:", prompt)
```

- [ ] **Step 3: Extend artifact state and archive state so runtime owns both paths explicitly**

```python
@dataclass
class OptimizeSessionArtifactsState:
    memory_file: MemoryFileState
    archive: ArchiveState
    subagent_stage_set: SubagentStageSet | None = None
    hidden_triton_agent_dir: Path | None = None
    supervisor_report_path: Path | None = None
    supervisor_history_dir: Path | None = None
    workflow_state_path: Path | None = None


@property
def round_timings_archive_path(self) -> Path:
    return self.archive.run_archive_dir / "round-timings.json"
```

and in `ArchiveManager.archive` reserve the new name:

```python
_EXPECTED_NAMES = frozenset({
    "show-output.log",
    "tool-traces.jsonl",
    "history",
    "shared-guidance.md",
    "supervisor-report.md",
    "round-timings.json",
})
```

- [ ] **Step 4: Hook-gate state bootstrap and cleanup in `session_artifacts.py`**

```python
archive_state = self._archives.prepare(workdir, include_shared_guidance_snapshot=True)
workflow_state_path = None
hidden_triton_agent_dir = None
if enable_agent_hooks:
    hidden_triton_agent_dir = self._prepare_hidden_triton_agent_dir(workdir)
    workflow_state_path = hidden_triton_agent_dir / "state.json"
    bootstrap_optimize_workflow_state(
        workflow_state_path,
        run_id=archive_state.run_id,
        source_operator=source_operator_path or workdir,
        baseline_reused=False,
    )

return OptimizeSessionArtifactsState(
    memory_file=memory_file_state,
    archive=archive_state,
    subagent_stage_set=subagent_stage_set,
    hidden_triton_agent_dir=hidden_triton_agent_dir,
    workflow_state_path=workflow_state_path,
)
```

and in `OptimizeSessionArtifactsManager.archive`:

```python
if state.workflow_state_path is not None:
    try:
        archive_round_timings_from_state(
            state.workflow_state_path,
            state.round_timings_archive_path,
        )
    except Exception as exc:
        warnings.append(f"Failed to archive optimize round timings: {exc}")
```

- [ ] **Step 5: Inject phase summaries through runtime and prompt builders**

```python
def build_optimize_round_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
    round_mode: Literal["checked", "supervised"],
    baseline_ready: bool = True,
    current_round: int = 1,
    final_round: int = 1,
    round_batch_size: int = 5,
    workflow_phase_summary: str | None = None,
) -> str:
    if workflow_phase_summary is not None:
        lines.extend(["", "Workflow phase summary:", workflow_phase_summary])
    return _finalize_optimize_prompt_lines(
        lines=lines,
        resume_existing_session=resume_existing_session,
        compiler_source_path=compiler_source_path,
        compiler_source_commit=compiler_source_commit,
        enable_cann_ext_api=enable_cann_ext_api,
    )
```

```python
phase_summary = render_optimize_phase_summary(self._artifacts_state.workflow_state_path)

baseline_request = replace(
    request,
    prompt=build_optimize_baseline_prompt(
        request.input_path,
        request.output_path,
        test_mode=request.test_mode,
        bench_mode=request.bench_mode,
        target_chip=request.target_chip,
        optimize_target=request.optimize_target,
        compiler_source_path=request.compiler_source_path,
        compiler_source_commit=request.compiler_source_commit,
        enable_cann_ext_api=_request_enables_cann_ext_api(request),
        baseline_state=preflight.state.value,
        base_prompt=_request_user_prompt(request),
        remote=request.remote,
        remote_workdir=request.remote_workdir,
        workflow_phase_summary=phase_summary,
    ),
)
```

Also thread the same optional argument through `src/triton_agent/prompts.py`:

```python
def build_prompt(
    command_kind: CommandKind,
    input_path: Path,
    operator_path: Path | None,
    output_path: Path | None,
    test_mode: str | None,
    bench_mode: str | None,
    force_overwrite: bool,
    remote: str | None = None,
    remote_workdir: str | None = None,
    min_rounds: int | None = 5,
    continue_optimize: bool = False,
    resume_existing_session: bool | None = None,
    round_mode: Literal["checked", "supervised"] = "checked",
    target_chip: str | None = None,
    optimize_target: str = "kernel",
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
    current_round: int = 1,
    final_round: int | None = None,
    round_batch_size: int = 5,
    optimize_baseline_ready: bool = True,
    workflow_phase_summary: str | None = None,
) -> str:
    lines.extend(
        build_optimize_round_prompt(
            input_path,
            output_path,
            test_mode=test_mode,
            bench_mode=bench_mode,
            target_chip=target_chip or "A5",
            optimize_target=optimize_target,
            resume_existing_session=should_resume_existing_session,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
            enable_subagent=enable_subagent,
            round_mode=round_mode,
            baseline_ready=optimize_baseline_ready,
            current_round=current_round,
            final_round=resolved_final_round,
            round_batch_size=round_batch_size,
            workflow_phase_summary=workflow_phase_summary,
        ).splitlines()
    )
```

- [ ] **Step 6: Re-run the targeted runtime and archive tests**

Run:

```bash
uv run python -m unittest tests.test_optimize_guidance tests.test_optimize_runtime -v
```

Expected: PASS

### Task 5: Run focused regression coverage, required pyright checks, and full repository verification

**Files:**
- Test: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_skill_command_script.py`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Run the focused optimize regression suite that covers the new helper, runtime archive path, and CLI scripts together**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_workflow_state \
  tests.test_skill_command_script \
  tests.test_optimize_guidance \
  tests.test_optimize_runtime \
  -v
```

Expected: PASS

- [ ] **Step 2: Re-run the required strict skill-script pyright checks after the code settles**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize/scripts/optimize_workflow_state.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-start-round/scripts/optimize_start_round.py
```

Expected: PASS

- [ ] **Step 3: Run the repository-standard lint, type-check, and test commands**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS

- [ ] **Step 4: Verify the two hook-gating invariants manually from test fixtures before declaring completion**

Check these assertions in the final green test run:

```python
self.assertFalse((workdir / ".triton-agent" / "state.json").exists())
self.assertFalse((state.run_archive_dir / "round-timings.json").exists())
```

for non-hook optimize sessions, and:

```python
self.assertTrue((workdir / ".triton-agent" / "state.json").exists())
self.assertTrue((state.run_archive_dir / "round-timings.json").exists())
```

for hook-enabled optimize sessions with at least one completed round.
