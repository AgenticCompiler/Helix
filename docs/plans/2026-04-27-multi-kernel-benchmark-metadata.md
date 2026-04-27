# Multi-Kernel Benchmark Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support multi-kernel benchmark metadata so generated harnesses can declare more than one Triton kernel, `run-bench --bench-mode msprof` can aggregate multiple kernel timings, and `profile-bench --bench-mode msprof` can require or auto-select a single profiling kernel at runtime.

**Architecture:** Keep the CLI thin and treat generated benchmark metadata as the source of truth. Normalize old `# kernel:` and new `# kernels:` headers into a single runtime kernel-list helper inside `skills/triton-npu-run-eval/scripts/`, use that helper for msprof CSV aggregation and profiling-kernel validation, and update generation contracts so new harnesses emit only `# kernels:` while runtime parsing remains backward-compatible.

**Tech Stack:** Python 3.11, `argparse`, existing run-eval skill scripts, Markdown skill/spec docs, Python `unittest`, `uv`, `pyright`

---

### Task 1: Lock the new contract with failing tests

**Files:**
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_profile_runner.py`
- Modify: `tests/test_remote_execution.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing benchmark metadata parsing tests for `# kernels:` and old `# kernel:` compatibility**

```python
def test_parse_bench_metadata_reads_multi_kernel_header(self) -> None:
    bench_file.write_text(
        "# bench-mode: msprof\n# api-name: fused\n# kernels: KernelA, KernelB\nprint('x')\n",
        encoding="utf-8",
    )
    metadata = module.parse_bench_metadata(bench_file)
    self.assertEqual(metadata["kernels"], "KernelA, KernelB")
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_parse_bench_metadata_reads_multi_kernel_header -v`
Expected: FAIL because the runtime does not yet normalize or use the new field.

- [ ] **Step 2: Add failing local and remote msprof benchmark tests that require summed latency across multiple declared kernels**

```python
self.assertEqual(
    perf_path.read_text(encoding="utf-8"),
    (
        'latency-case-1: 4.0\n'
        '# raw-op-statistic-case-1: {"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n'
    ),
)
```

Run: `uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`
Expected: FAIL because current msprof aggregation returns only one matched kernel.

- [ ] **Step 3: Add failing profile runner tests for explicit `--kernel-name`, automatic single-kernel selection, and multi-kernel rejection without a kernel name**

```python
with self.assertRaisesRegex(
    ValueError,
    "Multiple benchmark kernels declared; rerun profile-bench with --kernel-name",
):
    module.run_local_profile_bench(
        bench_file,
        operator_file,
        "msprof",
    )
```

Run: `uv run python -m unittest tests.test_profile_runner.ProfileRunnerTests -v`
Expected: FAIL because `profile_runner.py` currently only reads a single `# kernel:` field and has no runtime kernel selector.

- [ ] **Step 4: Add failing command help and contract tests for `--kernel-name` and `# kernels:`**

```python
self.assertIn("--kernel-name", completed.stdout)
self.assertIn("# kernels:", msprof)
self.assertIn("# kernels:", standalone)
```

Run: `uv run python -m unittest tests.test_skill_command_script tests.test_generation_contracts -v`
Expected: FAIL because the command help and generation docs still describe the single-kernel contract.

### Task 2: Implement multi-kernel benchmark aggregation in the run-eval runtime

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`

- [ ] **Step 1: Add shared kernel-list parsing helpers that normalize `# kernels:` and legacy `# kernel:` into a validated `list[str]`**

```python
def _parse_kernel_names(metadata: dict[str, str], bench_file: Path) -> list[str]:
    if "kernels" in metadata:
        kernel_names = [part.strip() for part in metadata["kernels"].split(",") if part.strip()]
    else:
        kernel_name = metadata.get("kernel", "").strip()
        kernel_names = [kernel_name] if kernel_name else []
    if not kernel_names:
        raise ValueError(f"Benchmark metadata is missing required 'kernels' entry: {bench_file}")
    return kernel_names
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_parse_bench_metadata_reads_multi_kernel_header -v`
Expected: PASS.

- [ ] **Step 2: Replace single-kernel msprof aggregation with summed matched-kernel latency**

```python
def _resolve_msprof_metrics(
    rows: list[MsprofAvgRow],
    kernel_names: list[str],
) -> MsprofMetrics:
    matched = [float(row["avg_time_us"]) for row in rows if str(row["op_type"]) in set(kernel_names)]
    kernel_avg_time_us = sum(matched) if matched else None
    return {"kernel_avg_time_us": kernel_avg_time_us, "ops": [...]}
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_run_local_bench_msprof_queries_case_count_and_runs_each_case -v`
Expected: PASS with summed latency values.

- [ ] **Step 3: Update the remote msprof CSV parser payload to accept and sum multiple kernel names**

```python
kernel_names = [part for part in sys.argv[2].split("\n") if part]
matched = [row["avg_time_us"] for row in ops if row["op_type"] in set(kernel_names)]
kernel_avg_time_us = sum(matched) if matched else None
```

Run: `uv run python -m unittest tests.test_remote_execution.RemoteExecutionTests.test_run_remote_bench_msprof_sums_avg_time_from_remote_csv_and_cleans_profiler_tmpdirs -v`
Expected: PASS with remote summed latency and unchanged raw-op payload output.

- [ ] **Step 4: Run the focused benchmark runner suite and strict file-scoped pyright for the touched skill script**

Run: `uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`
Expected: PASS.

Run:

```bash
tmpdir=$(mktemp -d)
printf '[tool.pyright]\npythonVersion = "3.11"\ninclude = ["%s"]\ntypeCheckingMode = "strict"\n' \
  "$PWD/skills/triton-npu-run-eval/scripts/bench_runner.py" > "$tmpdir/pyproject.toml"
uv run pyright --project "$tmpdir/pyproject.toml"
```

Expected: `0 errors`.

### Task 3: Add runtime profiling kernel selection for `profile-bench`

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/profile_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`

- [ ] **Step 1: Add `--kernel-name` to the `profile-bench` parser and thread it into local and remote profiling calls**

```python
profile_bench.add_argument("--kernel-name")
...
result, profile_dir = run_local_profile_bench(
    bench_file,
    operator_file,
    resolved_bench_mode,
    bench_case=args.bench,
    kernel_name=args.kernel_name,
)
```

Run: `uv run python -m unittest tests.test_skill_command_script.SkillCommandScriptTests.test_script_exposes_profile_bench_help -v`
Expected: PASS with `--kernel-name` in help output.

- [ ] **Step 2: Replace single-field kernel resolution in `profile_runner.py` with validated runtime selection**

```python
def _resolve_profile_kernel_name(
    bench_file: Path,
    requested_kernel_name: str | None,
) -> str:
    kernel_names = _resolve_kernel_names(bench_file)
    if requested_kernel_name is not None:
        if requested_kernel_name not in kernel_names:
            raise ValueError(
                f"Requested kernel '{requested_kernel_name}' is not declared in benchmark metadata: {kernel_names}"
            )
        return requested_kernel_name
    if len(kernel_names) == 1:
        return kernel_names[0]
    raise ValueError(
        "Multiple benchmark kernels declared; rerun profile-bench with --kernel-name <name>."
    )
```

Run: `uv run python -m unittest tests.test_profile_runner.ProfileRunnerTests -v`
Expected: PASS for explicit selection, single-kernel auto-selection, and actionable multi-kernel failure.

- [ ] **Step 3: Keep `msprof op --kernel-name=<name>` command shape unchanged while using the resolved runtime kernel**

```python
[
    "msprof",
    "op",
    f"--kernel-name={kernel_name}",
    sys.executable,
    bench_file.name,
    "--operator-file",
    operator_arg,
    "--bench",
    str(selected_case),
]
```

Run: `uv run python -m unittest tests.test_profile_runner.ProfileRunnerTests.test_run_local_profile_bench_msprof_requires_kernel_metadata_and_selected_case -v`
Expected: PASS with the selected runtime kernel forwarded unchanged to `msprof op`.

- [ ] **Step 4: Run the focused profile suite and strict file-scoped pyright for the touched profile script**

Run: `uv run python -m unittest tests.test_profile_runner tests.test_skill_command_script -v`
Expected: PASS.

Run:

```bash
tmpdir=$(mktemp -d)
printf '[tool.pyright]\npythonVersion = "3.11"\ninclude = ["%s"]\ntypeCheckingMode = "strict"\n' \
  "$PWD/skills/triton-npu-run-eval/scripts/profile_runner.py" > "$tmpdir/pyproject.toml"
uv run pyright --project "$tmpdir/pyproject.toml"
```

Expected: `0 errors`.

### Task 4: Align generation contracts, docs, and end-to-end verification

**Files:**
- Modify: `skills/triton-npu-gen-bench/SKILL.md`
- Modify: `skills/triton-npu-gen-bench/references/bench-standalone-spec.md`
- Modify: `skills/triton-npu-gen-bench/references/bench-msprof-spec.md`
- Modify: `skills/triton-npu-gen-test/SKILL.md`
- Modify: `skills/triton-npu-gen-test/references/test-standalone-spec.md`
- Modify: `skills/triton-npu-gen-test/references/test-differential-spec.md`
- Modify: `skills/triton-npu-profile-operator/SKILL.md`
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Modify: `README.md`
- Modify: `docs/notes/2026-04-01-generated-harness-metadata.md`
- Modify: `docs/specs/2026-04-24-msprof-csv-latency-design.md`

- [ ] **Step 1: Update generation docs and specs so new harnesses emit `# kernels:` instead of `# kernel:`**

```markdown
# bench-mode: msprof
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_benchmark_generation_specs_use_header_metadata_and_no_runtime_api_flag -v`
Expected: PASS with the new header contract.

- [ ] **Step 2: Update run-eval and profiler docs to describe multi-kernel aggregation plus `profile-bench --kernel-name` behavior**

```markdown
- In `msprof` mode, `run-bench` aggregates all kernel names declared by `# kernels:`.
- `profile-bench` accepts `--kernel-name`; omit it only when benchmark metadata resolves to exactly one kernel.
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_profiler_skill_documents_profile_bench_mode_contracts -v`
Expected: PASS with the new profiling contract language.

- [ ] **Step 3: Run the targeted documentation/contract suites**

Run: `uv run python -m unittest tests.test_generation_contracts tests.test_skill_command_script -v`
Expected: PASS.

- [ ] **Step 4: Run repository verification for the full change**

Run: `uv run python -m unittest tests.test_bench_runner tests.test_profile_runner tests.test_remote_execution tests.test_generation_contracts tests.test_skill_command_script -v`
Expected: PASS.

Run: `uv run pyright`
Expected: `0 errors`.

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS.
