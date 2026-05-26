# Mixed Perf Case-ID Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make legacy `msprof` perf files and new JSONL perf files compare cleanly when they describe the same benchmark cases with different serialized case-id shapes.

**Architecture:** Keep the fix local to the perf compatibility layer plus the `msprof` JSONL writer. First add a regression that proves `latency-case-<N>` and JSONL numeric `case_label` values currently mismatch, then add narrow required-id compatibility and restore stable future writer output.

**Tech Stack:** Python 3.11, `unittest`, existing perf artifact helpers, strict skill-script pyright

---

### Task 1: Lock the mixed-format failure with a regression

**Files:**
- Modify: `tests/test_comparison_commands.py`

- [ ] **Step 1: Add a failing compare-perf regression**

```python
def test_compare_perf_files_accepts_legacy_msprof_baseline_against_numeric_jsonl_case_labels(self) -> None:
    module = load_perf_artifacts_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline = root / "baseline_perf.txt"
        baseline.write_text(
            'latency-case-1: 10.0\n'
            '# raw-op-statistic-case-1: {"ops":[{"op_type":"K","avg_time_us":10.0}]}\n',
            encoding="utf-8",
        )
        compare = root / "compare_perf.txt"
        compare.write_text(
            '{"case_label":"1","kernel_names":["K"],"kernel_source":"metadata","kernel_avg_time_us":8.0,"total_op_avg_time_us":8.0,"error_message":null,"case_wall_clock_seconds":0.1}\n',
            encoding="utf-8",
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = module.compare_perf_files(baseline, compare)

        self.assertEqual(exit_code, 0)
        self.assertIn("latency-case-1", stdout.getvalue())
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `uv run python -m unittest tests.test_comparison_commands.JsonlPerfArtifactParserTests.test_compare_perf_files_accepts_legacy_msprof_baseline_against_numeric_jsonl_case_labels -v`
Expected: `FAIL` with a missing required latency id such as `latency-case-1`

### Task 2: Add narrow required-id compatibility

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Test: `tests/test_comparison_commands.py`

- [ ] **Step 1: Add a helper that recognizes the legacy `case-<N>` alias**

```python
def _legacy_msprof_latency_alias(latency_id: str) -> str | None:
    ...
```

- [ ] **Step 2: Use the helper in required-id parsing so exact ids still win first**

```python
required_match = _resolve_required_latency_id_match(latency_id, required_ids)
if required_match is None:
    continue
```

- [ ] **Step 3: Re-run the focused regression**

Run: `uv run python -m unittest tests.test_comparison_commands.JsonlPerfArtifactParserTests.test_compare_perf_files_accepts_legacy_msprof_baseline_against_numeric_jsonl_case_labels -v`
Expected: `PASS`

### Task 3: Stabilize future msprof JSONL output

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`
- Test: `tests/test_bench_runner.py`

- [ ] **Step 1: Update msprof case labels to keep the `case-<N>` public contract**

```python
case_label=f"case-{case_idx}"
```

- [ ] **Step 2: Add or update an assertion that the msprof JSONL output uses `case-<N>`**

Run: `uv run python -m unittest tests.test_bench_runner -k case_label -v`
Expected: the relevant msprof output assertion passes

### Task 4: Verify the touched code paths

**Files:**
- Modify: `tests/test_comparison_commands.py`
- Modify: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`

- [ ] **Step 1: Run the focused comparison and bench tests**

Run: `uv run python -m unittest tests.test_comparison_commands tests.test_bench_runner -v`
Expected: comparison and bench suites pass

- [ ] **Step 2: Run strict pyright for the touched skill script**

Run: `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/perf_artifacts.py`
Expected: `0 errors`
