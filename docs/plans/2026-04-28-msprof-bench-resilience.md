# Msprof Bench Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run-bench --bench-mode msprof` continue across failed cases, always persist a mixed-result perf artifact with per-case error details, and aggregate kernels from both benchmark metadata and the runtime operator file.

**Architecture:** Keep all behavior inside `skills/triton-npu-run-eval/scripts/bench_runner.py` so the CLI remains thin. Add per-case result accumulation helpers, extend perf artifact comments without changing the primary `latency-case-*` comparison key, and introduce AST-based operator kernel discovery that unions with metadata kernels in a stable order.

**Tech Stack:** Python 3.11, `ast`, `json`, existing run-eval skill scripts, Markdown docs, Python `unittest`, `uv`, `pyright`

---

### Task 1: Lock the new runtime contract with failing tests

**Files:**
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_remote_execution.py`

- [ ] **Step 1: Add a failing local test that proves `msprof` continues after a failed case and still writes a perf file**

```python
self.assertEqual(
    perf_path.read_text(encoding="utf-8"),
    (
        "latency-case-1: NA\n"
        "# latency-error-case-1: msprof command failed with return code 1\n"
        "latency-case-2: 5.0\n"
        '# raw-op-statistic-case-2: {"ops":[{"op_type":"KernelB","avg_time_us":5.0}]}\n'
    ),
)
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_run_local_bench_msprof_continues_after_failed_case_and_persists_perf -v`
Expected: FAIL because the current implementation returns early and never writes the mixed perf artifact.

- [ ] **Step 2: Add a failing remote test for the same best-effort continuation contract**

```python
self.assertEqual(result["return_code"], 1)
self.assertEqual(removed_tmp_dirs, ["/tmp/msprof-case-1", "/tmp/msprof-case-2"])
```

Run: `uv run python -m unittest tests.test_remote_execution.RemoteExecutionTests.test_run_remote_bench_msprof_continues_after_failed_case_and_persists_perf -v`
Expected: FAIL because the current remote flow also returns early on the first failed case.

- [ ] **Step 3: Add failing tests for persisted CSV parse errors and missing-kernel error comments**

```python
self.assertIn("# latency-error-case-1: No op_statistic_*.csv found under", text)
self.assertIn("# latency-error-case-1: no resolved kernels matched op_statistic csv", text)
```

Run: `uv run python -m unittest tests.test_bench_runner -v`
Expected: FAIL because the current implementation raises instead of recording an error comment.

- [ ] **Step 4: Add failing kernel-resolution tests for metadata/operator union behavior**

```python
self.assertEqual(
    module.resolve_bench_kernel_names(bench_file, operator_file),
    ["MetaKernel", "NewKernel"],
)
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_resolve_bench_kernel_names_unions_metadata_and_operator_kernels -v`
Expected: FAIL because the current helper reads metadata only and has no operator-file input.

### Task 2: Implement per-case accumulation and persisted error annotations

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`

- [ ] **Step 1: Introduce per-case result records that can render success, missing-kernel, and failure lines**

```python
@dataclass(frozen=True)
class MsprofCaseRecord:
    case_idx: int
    latency_value: str
    raw_ops: list[MsprofAvgRow] | None
    error_message: str | None
    resolved_kernels: list[str]
    kernel_source: str
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_run_local_bench_msprof_continues_after_failed_case_and_persists_perf -v`
Expected: still FAIL until the control flow is updated.

- [ ] **Step 2: Change local and remote `msprof` loops to catch per-case failures, append a failure record, and continue**

```python
try:
    metrics = _read_local_msprof_metrics(output_dir, kernel_names)
    records.append(_success_case_record(...))
except Exception as exc:
    had_case_failures = True
    records.append(_failed_case_record(case_idx, str(exc), resolved_kernels, kernel_source))
    continue
```

Run: `uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`
Expected: PASS for continuation and persisted perf-artifact tests.

- [ ] **Step 3: Write the perf file from accumulated records even when some cases failed, and return the perf path alongside a non-zero aggregate result**

```python
perf_path = _write_perf_lines(_perf_output_path(bench_file, operator_file), _render_case_records(records))
return make_result(return_code=1 if had_case_failures else 0, ...), perf_path
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_run_local_bench_msprof_continues_after_failed_case_and_persists_perf -v`
Expected: PASS with `perf_path` present and `result["return_code"] == 1`.

### Task 3: Add operator kernel discovery and stable union resolution

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`

- [ ] **Step 1: Add AST-based Triton kernel discovery for the runtime operator file**

```python
def _discover_operator_triton_kernels(operator_file: Path) -> list[str]:
    tree = ast.parse(operator_file.read_text(encoding="utf-8"), filename=str(operator_file))
    ...
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_resolve_bench_kernel_names_unions_metadata_and_operator_kernels -v`
Expected: PASS.

- [ ] **Step 2: Change kernel resolution to union metadata kernels with operator-discovered kernels in stable order**

```python
def resolve_bench_kernel_names(bench_file: Path, operator_file: Path) -> tuple[list[str], str]:
    metadata_kernels = _parse_kernel_names(...)
    operator_kernels = _discover_operator_triton_kernels(operator_file)
    return _stable_union(metadata_kernels, operator_kernels), _describe_kernel_source(...)
```

Run: `uv run python -m unittest tests.test_bench_runner -v`
Expected: PASS for metadata-only, operator-only, union, and malformed-source failure coverage.

- [ ] **Step 3: Thread resolved kernels and kernel source through local and remote case rendering**

```python
resolved_kernels, kernel_source = resolve_bench_kernel_names(bench_file, operator_file)
```

Run: `uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`
Expected: PASS with `# resolved-kernels-case-*` and `# kernel-source-case-*` lines present.

### Task 4: Keep `compare-perf` honest around execution failures

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `tests/test_bench_runner.py`

- [ ] **Step 1: Extend perf parsing helpers to collect `latency-error-case-*` comments**

```python
if line.startswith("# latency-error-"):
    ...
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_compare_perf_files_fails_on_case_execution_error_marker -v`
Expected: FAIL until comparison logic consumes the new metadata.

- [ ] **Step 2: Reject comparisons for cases marked with execution or CSV parsing failures while preserving total-op fallback for missing-kernel `NA`**

```python
if latency_id in error_map and error_map[latency_id] != "no resolved kernels matched op_statistic csv":
    raise ValueError(...)
```

Run: `uv run python -m unittest tests.test_bench_runner -v`
Expected: PASS with explicit failure for broken runs and unchanged total-op fallback for valid missing-kernel cases.

### Task 5: Update docs and run required verification

**Files:**
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Modify: `README.md`

- [ ] **Step 1: Document the new best-effort `msprof` bench behavior and error annotations**

```md
- In `msprof` mode, failed benchmark cases do not stop later cases from running.
- The generated perf file keeps successful cases and records `# latency-error-case-*` comments for failed ones.
```

Run: `uv run python -m unittest tests.test_generation_contracts tests.test_skill_command_script -v`
Expected: PASS if no contract tests require additional updates.

- [ ] **Step 2: Run focused unit tests**

Run: `uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`
Expected: PASS.

- [ ] **Step 3: Run strict file-scoped pyright for the touched skill script**

Run:

```bash
tmpdir=$(mktemp -d)
printf '[tool.pyright]\npythonVersion = "3.11"\ninclude = ["%s"]\ntypeCheckingMode = "strict"\n' \
  "$PWD/skills/triton-npu-run-eval/scripts/bench_runner.py" > "$tmpdir/pyproject.toml"
uv run pyright --project "$tmpdir/pyproject.toml"
```

Expected: `0 errors`.
