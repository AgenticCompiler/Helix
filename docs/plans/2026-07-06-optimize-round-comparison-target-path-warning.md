# Optimize Round Comparison Target Path Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the canonical round comparison field to `comparison_target_path`, keep legacy `comparison_target` inputs working, and make round/baseline path warnings tell operators exactly which field is wrong and how to repair it.

**Architecture:** Keep the contract rename and compatibility logic centered in the optimize-state round loader so downstream runtime code sees one canonical property name: `RoundState.comparison_target_path`. Reuse a small shared path-diagnostic formatter from the optimize-state skill scripts so `round/check.py` and `baseline/check.py` produce the same field-aware warning style, then sweep canonical docs and fixtures to emit the new field name while preserving a narrow legacy-read compatibility path.

**Tech Stack:** Python 3.11, `json`, `pathlib`, `unittest`, optimize-state skill scripts, Triton/TileLang skill references, `uv`, `ruff`, `pyright`

---

## File Map

- Modify: `skills/common/ascend-npu-optimize-state/references/round-contract.json`
  Rename the required field to `comparison_target_path`.
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/models.py`
  Rename the `RoundState` dataclass field to `comparison_target_path`.
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/paths.py`
  Add field-aware path diagnostic helpers shared by baseline and round checks.
- Modify: `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
  Normalize legacy/new round-state fields and upgrade comparison/round artifact diagnostics.
- Modify: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
  Reuse the shared path diagnostic helpers for baseline artifact warnings.
- Run: `python3 skills/triton/triton-npu-optimize/script/update-artifacts.py`
  Regenerate `skills/triton/triton-npu-optimize/references/artifacts.md` from the updated contract.
- Modify: `skills/tilelang/tilelang-npu-optimize/references/artifacts.md`
  Mirror the canonical field rename in the TileLang round-state reference.
- Modify: `tests/test_optimize_contract.py`
  Assert the round contract now requires `comparison_target_path`.
- Modify: `tests/test_optimize_round_contract.py`
  Add loader coverage for the new field, legacy alias fallback, and conflicting dual-field rejection.
- Modify: `tests/test_optimize_checks.py`
  Add regression coverage for the improved field-aware path diagnostics and update canonical round-state fixtures.
- Modify: `tests/test_verify.py`
  Update canonical round-state fixtures to emit `comparison_target_path`.
- Modify: `tests/test_status.py`
  Update canonical round-state fixtures to emit `comparison_target_path`.
- Modify: `tests/test_skill_command_script.py`
  Update canonical round-state fixtures to emit `comparison_target_path`.
- Modify: `tests/test_optimize_profile_cleanup.py`
  Update canonical round-state fixtures to emit `comparison_target_path`.
- Modify: `tests/test_optimize_runtime.py`
  Update canonical round-state fixtures to emit `comparison_target_path`.
- Modify: `docs/specs/2026-04-13-optimize-baseline-prep-design.md`
  Update the copied round-state field name in historical contract prose.
- Modify: `docs/specs/2026-05-21-optimize-round-state-simplify-design.md`
  Update the contract rename discussion and warning wording examples.
- Modify: `docs/plans/2026-04-13-optimize-baseline-prep.md`
  Update the copied round-state field name in implementation examples.
- Modify: `docs/plans/2026-06-22-optimize-workflow-state-phase-tracking-implementation-plan.md`
  Update the copied round-state payload example.

### Task 1: Lock the contract rename and loader compatibility with failing tests

**Files:**
- Modify: `tests/test_optimize_contract.py`
- Modify: `tests/test_optimize_round_contract.py`
- Test: `tests/test_optimize_contract.py`
- Test: `tests/test_optimize_round_contract.py`

- [ ] **Step 1: Update the contract test to require `comparison_target_path`**

```python
def test_round_contract_uses_described_field_maps_without_baseline_duplication(
    self,
) -> None:
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-optimize-state"
        / "references"
        / "round-contract.json"
    )
    data = json.loads(contract_path.read_text(encoding="utf-8"))

    field_map = data["round_state_required_fields"]
    self.assertIn("comparison_target_path", field_map)
    self.assertNotIn("comparison_target", field_map)
    for key in ROUND_STATE_REQUIRED_FIELDS:
        self.assertIn(key, field_map)
        self.assertIsInstance(field_map[key], str)
```

- [ ] **Step 2: Add a failing loader test for the new canonical field**

```python
def test_load_round_state_reads_comparison_target_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        round_dir = Path(tmp) / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": "opt-round-1",
                    "parent_round": "round-0",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "comparison_target_path": "../baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            ),
            encoding="utf-8",
        )

        state = load_round_state(round_dir)

    self.assertEqual(state.comparison_target_path, "../baseline/perf.txt")
```

- [ ] **Step 3: Add a failing compatibility test for legacy `comparison_target` input**

```python
def test_load_round_state_accepts_legacy_comparison_target_alias(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        round_dir = Path(tmp) / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": "opt-round-1",
                    "parent_round": "round-0",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "comparison_target": "../baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            ),
            encoding="utf-8",
        )

        state = load_round_state(round_dir)

    self.assertEqual(state.comparison_target_path, "../baseline/perf.txt")
```

- [ ] **Step 4: Add a failing conflict test for dual fields with different values**

```python
def test_load_round_state_rejects_conflicting_comparison_target_fields(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        round_dir = Path(tmp) / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": "opt-round-1",
                    "parent_round": "round-0",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "comparison_target_path": "../baseline/kernel_perf.txt",
                    "comparison_target": "../baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "comparison_target_path and comparison_target disagree",
        ):
            load_round_state(round_dir)
```

- [ ] **Step 5: Add a failing required-field test that now names `comparison_target_path`**

```python
def test_load_round_state_requires_new_comparison_target_path_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        round_dir = Path(tmp) / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "round-state.json").write_text(
            json.dumps({"round": "opt-round-1"}),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as ctx:
            load_round_state(round_dir)

    self.assertIn("missing required round-state fields", str(ctx.exception))
    self.assertIn("comparison_target_path", str(ctx.exception))
```

- [ ] **Step 6: Run the focused contract/loader tests to verify they fail before implementation**

Run:

```bash
uv run python -m unittest tests.test_optimize_contract tests.test_optimize_round_contract -v
```

Expected: FAIL because the contract still names `comparison_target`, `RoundState` still exposes `comparison_target`, and `load_round_state()` does not yet normalize the legacy/new field pair.

### Task 2: Lock the warning improvements with failing regression tests

**Files:**
- Modify: `tests/test_optimize_checks.py`
- Test: `tests/test_optimize_checks.py`

- [ ] **Step 1: Add a failing round-check test for a missing declared comparison target path**

```python
def test_check_round_reports_missing_comparison_target_path_with_field_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        self._write_baseline(workdir)
        round_dir = self._write_round(workdir, "opt-round-1")
        payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
        payload["comparison_target_path"] = "../baseline/missing_perf.txt"
        payload.pop("comparison_target", None)
        (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

        result = optimize_checks.check_round(round_dir)

    self.assertEqual(result.status, "fail")
    self.assertIn(
        "comparison_target_path points to a missing file: ../baseline/missing_perf.txt",
        result.issues,
    )
    self.assertTrue(
        any("expected ../baseline/perf.txt" in issue for issue in result.issues),
    )
```

- [ ] **Step 2: Add a failing round-check test for a wrong-but-existing comparison target path**

```python
def test_check_round_reports_noncanonical_comparison_target_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        self._write_baseline(workdir)
        (workdir / "baseline" / "other_perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
        round_dir = self._write_round(workdir, "opt-round-1")
        payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
        payload["comparison_target_path"] = "../baseline/other_perf.txt"
        payload.pop("comparison_target", None)
        (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

        result = optimize_checks.check_round(round_dir)

    self.assertEqual(result.status, "fail")
    self.assertIn(
        "comparison_target_path must point to the canonical baseline perf artifact ../baseline/perf.txt (got ../baseline/other_perf.txt)",
        result.issues,
    )
```

- [ ] **Step 3: Add a failing round-check test for baseline-invalid comparison diagnostics**

```python
def test_check_round_reports_baseline_invalid_reason_when_comparison_target_cannot_be_validated(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        self._write_baseline(workdir)
        baseline_payload = json.loads((workdir / "baseline" / "state.json").read_text(encoding="utf-8"))
        baseline_payload.pop("perf_artifact")
        (workdir / "baseline" / "state.json").write_text(json.dumps(baseline_payload), encoding="utf-8")
        round_dir = self._write_round(workdir, "opt-round-1")

        result = optimize_checks.check_round(round_dir)

    self.assertEqual(result.status, "fail")
    self.assertTrue(
        any(
            issue.startswith(
                "cannot validate comparison_target_path because baseline/state.json is invalid:"
            )
            for issue in result.issues
        )
    )
```

- [ ] **Step 4: Add failing artifact-inspection tests for the neighboring path-bearing warnings**

```python
def test_check_round_reports_summary_path_field_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        round_dir = Path(tmp) / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "opt_kernel_perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
        (round_dir / "opt_kernel.py").write_text("print('x')\n", encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": "opt-round-1",
                    "parent_round": "round-0",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "comparison_target_path": "baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                }
            ),
            encoding="utf-8",
        )

        result = optimize_checks.check_round(round_dir)

    self.assertEqual(result.status, "fail")
    self.assertIn(
        "summary_path points to a missing file: summary.md (expected summary.md)",
        result.issues,
    )


def test_check_round_reports_declared_perf_analysis_field_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        self._write_baseline(workdir)
        round_dir = self._write_round(
            workdir,
            "opt-round-1",
            perf_analysis_path="perf-analysis.md",
        )

        result = optimize_checks.check_round(round_dir)

    self.assertEqual(result.status, "fail")
    self.assertIn(
        "perf_analysis_path points to a missing file: perf-analysis.md",
        result.issues,
    )


def test_check_baseline_reports_declared_perf_artifact_field_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        baseline_dir = workdir / "baseline"
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
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

        result = optimize_checks.check_baseline(baseline_dir)

    self.assertEqual(result.status, "fail")
    self.assertIn(
        "perf_artifact points to a missing file: baseline/perf.txt",
        result.issues,
    )


def test_check_baseline_reports_declared_operator_field_name(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        baseline_dir = workdir / "baseline"
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
        (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")

        result = optimize_checks.check_baseline(baseline_dir)

    self.assertEqual(result.status, "fail")
    self.assertIn(
        "baseline_operator points to a missing file: baseline/kernel.py",
        result.issues,
    )
```

- [ ] **Step 5: Update the canonical round fixture helper to emit `comparison_target_path`**

```python
payload = {
    "round": round_name,
    "parent_round": "round-0",
    "hypothesis": "vectorize loads",
    "evidence_sources": ["benchmark"],
    "correctness_status": "passed",
    "benchmark_status": "passed",
    "perf_artifact": "opt_kernel_perf.txt",
    "comparison_target_path": "baseline/perf.txt",
    "effective_metric_source": effective_metric_source,
    "summary_path": "summary.md",
    "opt_note_updated": True,
}
```

- [ ] **Step 6: Run the focused warning tests to verify they fail before implementation**

Run:

```bash
uv run python -m unittest tests.test_optimize_checks -v
```

Expected: FAIL because the checkers still emit terse path messages, the baseline-invalid warning still names `comparison_target`, and canonical fixtures still rely on the legacy field name.

### Task 3: Implement the contract rename, compatibility logic, and shared warning formatter

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/references/round-contract.json`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/models.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/paths.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
- Test: `tests/test_optimize_contract.py`
- Test: `tests/test_optimize_round_contract.py`
- Test: `tests/test_optimize_checks.py`

- [ ] **Step 1: Rename the contract field in `round-contract.json`**

```json
{
  "round_state_required_fields": {
    "perf_artifact": "record the path from the directory that contains `round-state.json` to the canonical round perf artifact, normally `opt_<operator>_perf.txt`.",
    "comparison_target_path": "record the path from the directory that contains `round-state.json` to the canonical baseline perf artifact used for comparison, normally `../baseline/<operator>_perf.txt` or `../baseline/perf.txt`.",
    "effective_metric_source": "record the resolved `compare-perf` basis that decided the round outcome: `kernel`, `total-op`, or `mixed`."
  }
}
```

- [ ] **Step 2: Rename the runtime dataclass field**

```python
@dataclass(frozen=True)
class RoundState:
    round_name: str
    parent_round: str
    hypothesis: str
    evidence_sources: tuple[str, ...]
    correctness_status: str
    benchmark_status: str
    perf_artifact: str
    comparison_target_path: str
    effective_metric_source: str
    summary_path: str
    opt_note_updated: bool
```

- [ ] **Step 3: Add shared field-aware path warning helpers**

```python
def missing_path_issue(
    field_name: str,
    declared_path: str | None,
    *,
    expected_path: str | None = None,
) -> str:
    if declared_path is None:
        if expected_path is None:
            return f"missing required path field: {field_name}"
        return f"{field_name} is missing (expected {expected_path})"
    if expected_path is None:
        return f"{field_name} points to a missing file: {declared_path}"
    return f"{field_name} points to a missing file: {declared_path} (expected {expected_path})"


def noncanonical_path_issue(field_name: str, declared_path: str, *, expected_path: str) -> str:
    return (
        f"{field_name} must point to the canonical baseline perf artifact "
        f"{expected_path} (got {declared_path})"
    )


def invalid_dependency_issue(field_name: str, dependency_label: str, reason: str) -> str:
    return f"cannot validate {field_name} because {dependency_label} is invalid: {reason}"
```

- [ ] **Step 4: Normalize `comparison_target_path` / `comparison_target` in `load_round_state()`**

```python
comparison_target_path = data.get("comparison_target_path")
legacy_comparison_target = data.get("comparison_target")
if comparison_target_path is None and legacy_comparison_target is not None:
    comparison_target_path = legacy_comparison_target
elif (
    comparison_target_path is not None
    and legacy_comparison_target is not None
    and str(comparison_target_path) != str(legacy_comparison_target)
):
    raise ValueError(
        "comparison_target_path and comparison_target disagree: "
        f"{comparison_target_path!r} != {legacy_comparison_target!r}"
    )

missing_fields = [
    field_name
    for field_name in ROUND_STATE_REQUIRED_FIELDS
    if field_name not in data and not (
        field_name == "comparison_target_path" and legacy_comparison_target is not None
    )
]
```

- [ ] **Step 5: Replace the inline comparison-target warning strings in `check_round()`**

```python
comparison_target_resolved = declared_state_file(
    round_dir,
    round_dir.parent,
    round_state.comparison_target_path,
)
if comparison_target_resolved is None:
    semantic_issues.append(
        missing_path_issue(
            "comparison_target_path",
            round_state.comparison_target_path,
            expected_path=expected_comparison_target,
        )
    )
elif (
    baseline_perf_path is not None
    and comparison_target_resolved.resolve() != baseline_perf_path.resolve()
):
    semantic_issues.append(
        noncanonical_path_issue(
            "comparison_target_path",
            round_state.comparison_target_path,
            expected_path=expected_comparison_target,
        )
    )
```

- [ ] **Step 6: Upgrade the artifact-inspection and baseline warning call sites to use the shared helper**

```python
if summary_path is None:
    issues.append(
        missing_path_issue(
            "summary_path",
            declared_summary,
            expected_path="summary.md",
        )
    )
if perf_path is None:
    issues.append(
        missing_path_issue(
            "perf_artifact",
            declared_perf,
            expected_path=expected_perf_name_value,
        )
    )
if perf_analysis_path is None and declared_analysis is not None:
    issues.append(
        missing_path_issue(
            "perf_analysis_path",
            declared_analysis,
            expected_path="perf-analysis.md",
        )
    )
```

- [ ] **Step 7: Re-run the focused unit tests**

Run:

```bash
uv run python -m unittest tests.test_optimize_contract tests.test_optimize_round_contract tests.test_optimize_checks -v
```

Expected: PASS

- [ ] **Step 8: Run the required strict Pyright checks for each touched skill script**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/models.py
```

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/shared/paths.py
```

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/check.py
```

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/baseline/check.py
```

Expected: PASS

### Task 4: Sync generated/manual docs and sweep canonical fixtures

**Files:**
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/references/artifacts.md`
- Modify: `tests/test_verify.py`
- Modify: `tests/test_status.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_optimize_profile_cleanup.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `docs/specs/2026-04-13-optimize-baseline-prep-design.md`
- Modify: `docs/specs/2026-05-21-optimize-round-state-simplify-design.md`
- Modify: `docs/plans/2026-04-13-optimize-baseline-prep.md`
- Modify: `docs/plans/2026-06-22-optimize-workflow-state-phase-tracking-implementation-plan.md`
- Test: `tests/test_verify.py`
- Test: `tests/test_status.py`
- Test: `tests/test_skill_command_script.py`
- Test: `tests/test_optimize_profile_cleanup.py`
- Test: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Regenerate the Triton artifact reference from the updated contract**

Run:

```bash
python3 skills/triton/triton-npu-optimize/script/update-artifacts.py
```

Expected: prints `skills/triton/triton-npu-optimize/references/artifacts.md`

- [ ] **Step 2: Update the TileLang artifact reference and historical copied examples**

```json
{
  "perf_artifact": "opt_kernel_perf.txt",
  "comparison_target_path": "../baseline/perf.txt",
  "effective_metric_source": "kernel",
  "summary_path": "summary.md"
}
```

- [ ] **Step 3: Sweep canonical round-state test fixtures to emit the new field name**

```python
(round_dir / "round-state.json").write_text(
    json.dumps(
        {
            "round": "opt-round-1",
            "parent_round": "baseline",
            "hypothesis": "faster",
            "evidence_sources": ["benchmark"],
            "correctness_status": "passed",
            "benchmark_status": "passed",
            "perf_artifact": "opt_kernel_perf.txt",
            "comparison_target_path": "baseline/perf.txt",
            "effective_metric_source": "kernel",
            "summary_path": "summary.md",
            "opt_note_updated": True,
        }
    ),
    encoding="utf-8",
)
```

- [ ] **Step 4: Run the impacted caller/fixture tests**

Run:

```bash
uv run python -m unittest tests.test_verify tests.test_status tests.test_skill_command_script tests.test_optimize_profile_cleanup tests.test_optimize_runtime -v
```

Expected: PASS

### Task 5: Run repository verification

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/references/round-contract.json`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/models.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/paths.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/references/artifacts.md`
- Modify: `tests/test_optimize_contract.py`
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_verify.py`
- Modify: `tests/test_status.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_optimize_profile_cleanup.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `docs/specs/2026-04-13-optimize-baseline-prep-design.md`
- Modify: `docs/specs/2026-05-21-optimize-round-state-simplify-design.md`
- Modify: `docs/plans/2026-04-13-optimize-baseline-prep.md`
- Modify: `docs/plans/2026-06-22-optimize-workflow-state-phase-tracking-implementation-plan.md`

- [ ] **Step 1: Run Ruff**

Run:

```bash
uv run --group dev ruff check
```

Expected: PASS

- [ ] **Step 2: Run Pyright**

Run:

```bash
uv run pyright
```

Expected: PASS

- [ ] **Step 3: Run pytest**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS
