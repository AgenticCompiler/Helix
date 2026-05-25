# Compare Perf Metric Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--metric-source auto|kernel|total-op` option to `compare-perf` so callers can explicitly choose which timing source drives comparison metrics while preserving current default behavior.

**Architecture:** Extend the `compare-perf` option surface in the repo CLI, comparison command wrapper, and `triton-npu-run-eval` skill script, then teach `perf_artifacts.py` to parse baseline and compare inputs under an explicit metric-source policy instead of only the current implicit auto-fallback behavior. Keep `auto` as the default compatibility mode, and reuse the existing `--skip-latency-errors` flow to decide whether metric-source-specific invalid cases abort immediately or are skipped and summarized at the end.

**Tech Stack:** Python 3.12, `argparse`, `unittest`, repository `uv` tooling, `pyright`, `ruff`

---

### Task 1: Lock the New CLI Contract With Failing Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_comparison_commands.py`
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Write the failing parser and forwarding tests**

Add assertions that `compare-perf` accepts `--metric-source`, defaults it to `"auto"`, and forwards explicit values through the repo CLI and command wrapper.

```python
def test_compare_perf_accepts_metric_source_flag(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-perf",
            "--baseline",
            "baseline_perf.txt",
            "--compare",
            "candidate_perf.txt",
            "--metric-source",
            "total-op",
        ]
    )
    self.assertEqual(args.metric_source, "total-op")


def test_compare_perf_defaults_metric_source_to_auto(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "compare-perf",
            "--baseline",
            "baseline_perf.txt",
            "--compare",
            "candidate_perf.txt",
        ]
    )
    self.assertEqual(args.metric_source, "auto")


def test_handle_compare_perf_forwards_metric_source(self) -> None:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline = root / "baseline_perf.txt"
        compare = root / "candidate_perf.txt"
        baseline.write_text("latency-a: 10\n", encoding="utf-8")
        compare.write_text("latency-a: 11\n", encoding="utf-8")
        args = parser.parse_args(
            [
                "compare-perf",
                "--baseline",
                str(baseline),
                "--compare",
                str(compare),
                "--metric-source",
                "kernel",
            ]
        )

        with patch(
            "triton_agent.commands.comparison.compare_perf_files",
            return_value=0,
        ) as mocked:
            exit_code = handle_compare_perf(parser, args)

    self.assertEqual(exit_code, 0)
    mocked.assert_called_once_with(
        baseline.resolve(),
        compare.resolve(),
        skip_latency_errors=False,
        metric_source="kernel",
    )
```

- [ ] **Step 2: Run the CLI-focused tests to verify they fail**

Run: `uv run python -m pytest tests/test_cli.py tests/test_comparison_commands.py tests/test_skill_command_script.py -k "compare_perf or metric_source" -v`

Expected: FAIL because `metric_source` is not accepted or not forwarded yet.

- [ ] **Step 3: Add the skill-script parser coverage for the same option**

Extend the `run-command.py` parser test so the staged skill command surface matches the repo CLI.

```python
def test_compare_perf_parser_accepts_metric_source_flag(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "triton-npu-run-eval"
        / "scripts"
        / "run-command.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "compare-perf",
            "--baseline",
            "baseline_perf.txt",
            "--compare",
            "candidate_perf.txt",
            "--metric-source",
            "kernel",
        ]
    )

    self.assertEqual(args.metric_source, "kernel")
```

- [ ] **Step 4: Re-run the same test command and confirm the failure is still about missing implementation**

Run: `uv run python -m pytest tests/test_cli.py tests/test_comparison_commands.py tests/test_skill_command_script.py -k "compare_perf or metric_source" -v`

Expected: FAIL only because the production code still lacks `metric_source` wiring.

- [ ] **Step 5: Commit the failing contract tests**

```bash
git add tests/test_cli.py tests/test_comparison_commands.py tests/test_skill_command_script.py
git commit -m "test: cover compare-perf metric source option"
```

### Task 2: Add Metric-Source Wiring Through All Compare-Perf Entrypoints

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/comparison.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_comparison_commands.py`
- Test: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add `--metric-source` to the repo CLI parser**

In the `compare-perf` branch of `_add_primary_arguments`, add:

```python
subparser.add_argument(
    "--metric-source",
    default="auto",
    choices=("auto", "kernel", "total-op"),
)
```

- [ ] **Step 2: Add the same option to the skill command parser**

In `skills/triton-npu-run-eval/scripts/run-command.py`, update the `compare_perf` parser block:

```python
compare_perf.add_argument(
    "--metric-source",
    default="auto",
    choices=["auto", "kernel", "total-op"],
)
```

- [ ] **Step 3: Thread `metric_source` through the comparison wrapper API**

Update the protocol and wrapper signatures in `src/triton_agent/commands/comparison.py`:

```python
class ComparePerfModule(Protocol):
    def compare_perf_files(
        self,
        baseline_perf: Path,
        compare_perf: Path,
        *,
        skip_latency_errors: bool = False,
        metric_source: str = "auto",
    ) -> int: ...


def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: str = "auto",
) -> int:
    return _load_compare_perf().compare_perf_files(
        baseline_perf,
        compare_perf,
        skip_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )
```

- [ ] **Step 4: Forward `metric_source` from both CLI handlers**

Update both handlers to pass the parsed option:

```python
return compare_perf_files(
    baseline_perf,
    compare_perf,
    skip_latency_errors=args.skip_latency_errors,
    metric_source=args.metric_source,
)
```

and in the skill command script:

```python
return compare_perf_files(
    baseline_perf,
    compare_perf,
    skip_latency_errors=args.skip_latency_errors,
    metric_source=args.metric_source,
)
```

- [ ] **Step 5: Run the entrypoint tests and verify they pass**

Run: `uv run python -m pytest tests/test_cli.py tests/test_comparison_commands.py tests/test_skill_command_script.py -k "compare_perf or metric_source" -v`

Expected: PASS

- [ ] **Step 6: Commit the entrypoint wiring**

```bash
git add src/triton_agent/cli.py src/triton_agent/commands/comparison.py skills/triton-npu-run-eval/scripts/run-command.py tests/test_cli.py tests/test_comparison_commands.py tests/test_skill_command_script.py
git commit -m "feat: wire compare-perf metric source option"
```

### Task 3: Drive Perf Parsing and Comparison by Explicit Metric Source

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Test: `tests/test_bench_runner.py`

- [ ] **Step 1: Write behavior tests for `auto`, `kernel`, and `total-op`**

Add focused tests covering:

```python
def test_compare_perf_files_auto_falls_back_to_total_op_when_kernel_is_missing(self) -> None:
    ...
    return_code = module.compare_perf_files(baseline, compare, metric_source="auto")
    self.assertEqual(return_code, 0)
    self.assertIn("Metric source: total-op", output)


def test_compare_perf_files_kernel_mode_fails_when_kernel_is_missing(self) -> None:
    ...
    return_code = module.compare_perf_files(baseline, compare, metric_source="kernel")
    self.assertEqual(return_code, 1)
    self.assertIn("requires kernel latency", output)


def test_compare_perf_files_total_op_mode_uses_raw_totals_even_when_kernel_exists(self) -> None:
    ...
    return_code = module.compare_perf_files(baseline, compare, metric_source="total-op")
    self.assertEqual(return_code, 0)
    self.assertIn("compare=total-op=", output)
    self.assertIn("Metric source: total-op", output)


def test_compare_perf_files_total_op_mode_fails_when_raw_totals_are_missing(self) -> None:
    ...
    return_code = module.compare_perf_files(baseline, compare, metric_source="total-op")
    self.assertEqual(return_code, 1)
    self.assertIn("requires '# raw-op-statistic-", output)
```

- [ ] **Step 2: Run the bench-runner comparison tests to verify they fail**

Run: `uv run python -m pytest tests/test_bench_runner.py -k compare_perf_files -v`

Expected: FAIL because `metric_source` is not yet implemented in the comparison logic.

- [ ] **Step 3: Introduce a typed metric-source selector in `perf_artifacts.py`**

Add a new literal near `ComparisonMode`:

```python
MetricSource = Literal["auto", "kernel", "total-op"]
```

and extend the top-level compare function signature:

```python
def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: MetricSource = "auto",
) -> int:
```

- [ ] **Step 4: Refactor parsing so selected source controls entry creation**

Replace the current implicit “kernel or fallback total-op” behavior with a source-aware helper, for example:

```python
def _build_perf_entry_for_source(
    *,
    path: Path,
    line_no: int,
    latency_id: str,
    value_text: str,
    raw_totals: dict[str, float],
    metric_source: MetricSource,
) -> PerfEntry:
    if metric_source == "kernel":
        if value_text == "NA":
            raise ValueError(
                f"{path}:{line_no} requires kernel latency for '{latency_id}' under --metric-source kernel"
            )
        return PerfEntry(
            display_value=value_text,
            numeric_value=float(value_text),
            comparison_mode="latency",
        )

    if metric_source == "total-op":
        total_op_value = _require_raw_total(path, line_no, latency_id, raw_totals)
        display_value = (
            f"NA ({_format_total_op_display(total_op_value)})"
            if value_text == "NA"
            else _format_total_op_display(total_op_value)
        )
        return PerfEntry(
            display_value=display_value,
            numeric_value=total_op_value,
            comparison_mode="total-op",
        )

    if value_text == "NA":
        total_op_value = _require_raw_total(path, line_no, latency_id, raw_totals)
        return PerfEntry(
            display_value=f"NA ({_format_total_op_display(total_op_value)})",
            numeric_value=total_op_value,
            comparison_mode="total-op",
        )

    return PerfEntry(
        display_value=value_text,
        numeric_value=float(value_text),
        comparison_mode="latency",
    )
```

- [ ] **Step 5: Pass `metric_source` through both baseline and compare-side parsing helpers**

Update the comparison-only parsing helpers so they receive the selected source:

```python
baseline_outcome = _parse_perf_entries_for_comparison(
    baseline_perf,
    skip_latency_errors=skip_latency_errors,
    metric_source=metric_source,
)
compare_outcome = _parse_required_perf_entries_for_comparison(
    compare_perf,
    baseline_outcome.entries,
    skip_latency_errors=skip_latency_errors,
    metric_source=metric_source,
)
```

and make the underlying `_parse_*_impl(...)` functions accept and use the same `metric_source` parameter.

- [ ] **Step 6: Keep skip behavior but make metric-source-specific failures actionable**

When `skip_latency_errors` is true, skip cases that fail because the chosen source is unavailable. Preserve existing latency-error skipping, but make the final messages source-aware:

```python
f"{path}:{line_no} requires kernel latency for '{latency_id}' under --metric-source kernel"
```

and

```python
f"{path}:{line_no} requires '# raw-op-statistic-{...}: ...' for '{latency_id}' under --metric-source total-op"
```

- [ ] **Step 7: Update the `bench_runner.py` forwarding signature**

Keep the bridge aligned with the skill implementation:

```python
def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: str = "auto",
) -> int:
    return _compare_perf_files(
        baseline_perf,
        compare_perf,
        skip_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )
```

- [ ] **Step 8: Run the comparison behavior tests and verify they pass**

Run: `uv run python -m pytest tests/test_bench_runner.py -k compare_perf_files -v`

Expected: PASS

- [ ] **Step 9: Run the required strict skill-script type checks**

Run: `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/perf_artifacts.py`

Expected: `0 errors, 0 warnings, 0 informations`

Run: `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py`

Expected: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 10: Commit the comparison-logic change**

```bash
git add skills/triton-npu-run-eval/scripts/perf_artifacts.py skills/triton-npu-run-eval/scripts/bench_runner.py tests/test_bench_runner.py
git commit -m "feat: add compare-perf metric source selection"
```

### Task 4: Document the New Option and Re-Run Full Verification

**Files:**
- Modify: `README.md`
- Modify: `skills/triton-npu-run-eval/references/compare-perf.md`
- Verify: repository-wide checks from `README.md`

- [ ] **Step 1: Update the README compare-perf section**

Add usage guidance like:

```md
Pass `--metric-source kernel` to require kernel-only comparison, or `--metric-source total-op`
to force total-op aggregation for every case. The default `--metric-source auto` preserves the
existing behavior of preferring kernel latency and falling back to total-op when kernel timing
is unavailable.
```

- [ ] **Step 2: Update the skill reference doc**

Add the same contract to `skills/triton-npu-run-eval/references/compare-perf.md` and keep the wording consistent with README.

```md
- `--metric-source auto|kernel|total-op` selects how `compare-perf` derives each case's timing:
  `auto` preserves the current kernel-first fallback behavior, `kernel` requires kernel latency,
  and `total-op` requires raw op statistics for total-op aggregation.
```

- [ ] **Step 3: Run the targeted tests once more after the doc-adjacent code is settled**

Run: `uv run python -m pytest tests/test_bench_runner.py tests/test_comparison_commands.py tests/test_cli.py tests/test_skill_command_script.py -k "compare_perf or metric_source" -v`

Expected: PASS

- [ ] **Step 4: Run the full repository verification suite**

Run: `uv run --group dev ruff check`

Expected: `All checks passed!`

Run: `uv run pyright`

Expected: `0 errors, 0 warnings, 0 informations`

Run: `uv run python -m unittest discover -s tests -v`

Expected: `OK`

- [ ] **Step 5: Commit the docs and verification pass**

```bash
git add README.md skills/triton-npu-run-eval/references/compare-perf.md
git commit -m "docs: describe compare-perf metric source modes"
```

## Self-Review

- Spec coverage: this plan includes dedicated tasks for the new `--metric-source` flag surface, the `auto|kernel|total-op` behavior, interaction with `--skip-latency-errors`, output expectations, tests, and doc updates.
- Placeholder scan: every task names exact files, commands, and code snippets; there are no `TODO` or “implement later” placeholders.
- Type consistency: the plan uses `metric_source` consistently across CLI parsing, wrappers, bridges, and `perf_artifacts.py`, and keeps the canonical values `auto`, `kernel`, and `total-op` unchanged throughout.
