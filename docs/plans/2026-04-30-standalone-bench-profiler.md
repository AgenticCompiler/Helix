# Standalone Bench Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement import-only standalone benchmark modules with hook-based case construction, profiler-derived perf artifacts, `profile-bench --case-id`, and compatibility updates for remote execution plus IR capture.

**Architecture:** Add a feature-local helper script under `skills/triton-npu-run-eval/scripts/` that owns standalone benchmark hook loading, case validation, `torch_npu.profiler` execution, and `operator_details.csv` normalization. Keep `bench_runner.py` and `profile_runner.py` as orchestration wrappers that delegate to the helper locally and copy the helper into remote workspaces for remote execution. The helper also provides a non-profiler `run-one` mode that defaults to the first declared standalone case when no explicit selector is provided, so `capture_ir.py` can stay deterministic without adding a new required user flag.

**Tech Stack:** Python 3.11, `importlib.util`, `argparse`, `tempfile`, `torch_npu.profiler`, existing run-eval runtime helpers, Markdown docs, Python `unittest`, `bash`

---

### Task 1: Lock the standalone contract in docs and help output

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add a failing generation-contract test for the new standalone hook contract**

```python
standalone = _read("skills/triton-npu-gen-bench/references/bench-standalone-spec.md")
bench_gen = _read("skills/triton-npu-gen-bench/SKILL.md")

self.assertIn("build_operator_api(operator_module)", standalone)
self.assertIn("build_standalone_bench_cases(operator_api)", standalone)
self.assertNotIn('parser.add_argument("--operator-file"', standalone)
self.assertNotIn("def run_bench(operator_api):", standalone)
self.assertNotIn('print(f"latency-{case_id}: {latency}")', standalone)
self.assertIn("build_operator_api(operator_module)", bench_gen)
self.assertIn("build_standalone_bench_cases(operator_api)", bench_gen)
self.assertNotIn("accept only `--operator-file` at runtime for standalone mode", bench_gen)
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_benchmark_generation_specs_use_hooked_standalone_contract -v`
Expected: FAIL because the current standalone spec and generation skill still describe a directly executable benchmark script.

- [ ] **Step 2: Add a failing profiler-skill contract test for `--case-id` standalone profiling**

```python
profiler = _read("skills/triton-npu-profile-operator/SKILL.md")

self.assertIn("--case-id <id>", profiler)
self.assertIn("profile one selected `--case-id <id>` case", profiler)
self.assertNotIn("profile one selected `--bench <N>` case", profiler)
self.assertNotIn("msprof python3 bench_<op>.py --operator-file <operator-file>", profiler)
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_profiler_skill_documents_standalone_case_id_contract -v`
Expected: FAIL because the profiler skill still documents standalone profiling as direct `msprof python bench_<op>.py ...`.

- [ ] **Step 3: Add a failing `run-command.py profile-bench --help` test for `--case-id`**

```python
completed = subprocess.run(
    [sys.executable, str(script), "profile-bench", "--help"],
    capture_output=True,
    text=True,
    check=False,
)
self.assertEqual(completed.returncode, 0)
self.assertIn("--case-id", completed.stdout)
self.assertIn("--bench", completed.stdout)
```

Run: `uv run python -m unittest tests.test_skill_command_script.SkillCommandScriptTests.test_script_exposes_profile_bench_case_id_help -v`
Expected: FAIL because the current helper help exposes `--bench` and `--kernel-name`, but no standalone `--case-id`.

- [ ] **Step 4: Commit the contract-test changes**

```bash
git add tests/test_generation_contracts.py tests/test_skill_command_script.py
git commit -m "test: lock standalone bench contract docs"
```

### Task 2: Add failing tests for the new standalone runtime helper

**Files:**
- Create: `tests/test_standalone_bench_runtime.py`
- Modify: `tests/run_skill_test_utils.py`

- [ ] **Step 1: Add a test-loader helper for the new standalone runtime script**

```python
def load_standalone_bench_runtime_module():
    return load_operator_eval_script_module("standalone_bench_runtime")
```

Run: `uv run python -m unittest tests.test_standalone_bench_runtime -v`
Expected: FAIL because neither the loader helper nor the target runtime script exists yet.

- [ ] **Step 2: Add a failing test that loads an import-only standalone benchmark module and normalizes its cases**

```python
bench_file.write_text(
    """# bench-mode: standalone
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA, KernelB

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_standalone_bench_cases(operator_api):
    prepared = {"token": "bound"}
    def run_case():
        operator_api("case-a", prepared)
    return [{"id": "case-a", "fn": run_case, "warmup": 3, "repeats": 7}]
""",
    encoding="utf-8",
)
operator_file.write_text(
    """def build_api():
    def operator_api(name, prepared):
        return name, prepared["token"]
    return operator_api
""",
    encoding="utf-8",
)

cases, resolution = module.load_standalone_bench_cases(bench_file, operator_file)
self.assertEqual([case.case_id for case in cases], ["case-a"])
self.assertEqual(cases[0].warmup, 3)
self.assertEqual(cases[0].repeats, 7)
self.assertEqual(resolution.kernel_names, ["KernelA", "KernelB"])
```

Run: `uv run python -m unittest tests.test_standalone_bench_runtime.StandaloneBenchRuntimeTests.test_load_standalone_bench_cases_builds_hooked_cases -v`
Expected: FAIL because the helper module does not exist and the current standalone path does not import benchmark hooks.

- [ ] **Step 3: Add failing validation tests for missing hooks and duplicate case ids**

```python
with self.assertRaisesRegex(ValueError, "missing required hook 'build_operator_api'"):
    module.load_standalone_bench_cases(bench_file, operator_file)

with self.assertRaisesRegex(ValueError, "Duplicate standalone benchmark case id: case-a"):
    module.load_standalone_bench_cases(bench_file, operator_file)
```

Run: `uv run python -m unittest tests.test_standalone_bench_runtime.StandaloneBenchRuntimeTests.test_load_standalone_bench_cases_rejects_missing_hooks_and_duplicate_ids -v`
Expected: FAIL because there is no runtime helper enforcing the new contract yet.

- [ ] **Step 4: Add a failing perf-artifact test that patches profiler output and expects msprof-shaped lines**

```python
with patch.object(
    module,
    "_profile_case_with_profiler",
    return_value=(
        {
            "kernel_avg_time_us": 11.0,
            "ops": [
                {"op_type": "KernelA", "avg_time_us": 5.0},
                {"op_type": "KernelB", "avg_time_us": 6.0},
            ],
        },
        None,
    ),
):
    result, perf_path = module.run_local_standalone_bench(bench_file, operator_file)

self.assertEqual(
    perf_path.read_text(encoding="utf-8"),
    (
        'latency-case-a: 11.0\n'
        '# raw-op-statistic-case-a: {"ops":[{"op_type":"KernelA","avg_time_us":5.0},{"op_type":"KernelB","avg_time_us":6.0}]}\n'
        '# resolved-kernels-case-a: KernelA,KernelB\n'
        '# kernel-source-case-a: metadata\n'
    ),
)
```

Run: `uv run python -m unittest tests.test_standalone_bench_runtime.StandaloneBenchRuntimeTests.test_run_local_standalone_bench_writes_msprof_shaped_perf_lines -v`
Expected: FAIL because the standalone runtime helper does not exist.

- [ ] **Step 5: Commit the new runtime tests**

```bash
git add tests/run_skill_test_utils.py tests/test_standalone_bench_runtime.py
git commit -m "test: add standalone bench runtime coverage"
```

### Task 3: Add failing local wrapper tests for `bench_runner` and `profile_runner`

**Files:**
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_profile_runner.py`

- [ ] **Step 1: Add a failing local `run-bench` test that proves standalone delegation no longer executes the bench file as a script**

```python
fake_result = make_skill_result(0, "bench stdout\n", "")
perf_file = root / "kernel_perf.txt"
with patch.object(
    module,
    "run_local_standalone_bench",
    create=True,
    return_value=(fake_result, perf_file),
) as helper, patch.object(module, "run_streaming_process") as streaming:
    result, resolved_perf = module.run_local_bench(bench_file, operator_file, "standalone")

self.assertEqual(result["return_code"], 0)
self.assertEqual(resolved_perf, perf_file)
helper.assert_called_once_with(bench_file, operator_file)
streaming.assert_not_called()
```

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_run_local_bench_standalone_delegates_to_hook_runtime -v`
Expected: FAIL because the current implementation still runs `python bench_<op>.py --operator-file ...`.

- [ ] **Step 2: Add a failing local `profile-bench` test for standalone `--case-id` delegation**

```python
with patch.object(
    module,
    "profile_local_standalone_case",
    create=True,
    return_value=(make_skill_result(0, "profile stdout\n", ""), profile_dir),
) as helper:
    result, resolved_profile_dir = module.run_local_profile_bench(
        bench_file,
        operator_file,
        "standalone",
        case_id="case-b",
    )

self.assertEqual(result["return_code"], 0)
self.assertEqual(resolved_profile_dir, profile_dir)
helper.assert_called_once_with(bench_file, operator_file, "case-b")
```

Run: `uv run python -m unittest tests.test_profile_runner.ProfileRunnerTests.test_run_local_profile_bench_standalone_uses_case_id_runtime -v`
Expected: FAIL because `run_local_profile_bench(...)` does not accept `case_id` and still wraps the benchmark script with `msprof`.

- [ ] **Step 3: Add a failing local guard test that rejects `--case-id` in `msprof` mode**

```python
with self.assertRaisesRegex(ValueError, "--case-id is only valid for standalone benchmark profiling"):
    module.run_local_profile_bench(
        bench_file,
        operator_file,
        "msprof",
        bench_case=1,
        case_id="case-a",
    )
```

Run: `uv run python -m unittest tests.test_profile_runner.ProfileRunnerTests.test_run_local_profile_bench_msprof_rejects_case_id -v`
Expected: FAIL because the current profile runner has no standalone `case_id` concept.

- [ ] **Step 4: Commit the local wrapper tests**

```bash
git add tests/test_bench_runner.py tests/test_profile_runner.py
git commit -m "test: lock standalone bench wrapper delegation"
```

### Task 4: Add failing remote and IR-capture integration tests

**Files:**
- Modify: `tests/test_remote_execution.py`
- Modify: `tests/test_profile_runner.py`
- Modify: `tests/test_ascend_operator_ir_analyzer.py`

- [ ] **Step 1: Add a failing remote `run-bench` test that expects the copied helper script to drive standalone execution**

```python
copy_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
self.assertIn("standalone_bench_runtime.py", copy_targets)
self.assertEqual(
    remote_run.call_args.args[2],
    [
        "python3",
        "standalone_bench_runtime.py",
        "run-all",
        "--bench-file",
        "bench_kernel.py",
        "--operator-file",
        "kernel.py",
        "--perf-file",
        "kernel_perf.txt",
    ],
)
```

Run: `uv run python -m unittest tests.test_remote_execution.RemoteExecutionTests.test_run_remote_bench_standalone_uses_runtime_helper_and_copies_perf_back -v`
Expected: FAIL because the current remote standalone path still runs the benchmark file directly and never stages a helper script.

- [ ] **Step 2: Add a failing remote `profile-bench` test that expects `profile-one --case-id <id>`**

```python
self.assertEqual(
    remote_run.call_args.args[2],
    [
        "python3",
        "standalone_bench_runtime.py",
        "profile-one",
        "--bench-file",
        "bench_kernel.py",
        "--operator-file",
        "kernel.py",
        "--case-id",
        "case-b",
    ],
)
```

Run: `uv run python -m unittest tests.test_profile_runner.ProfileRunnerTests.test_run_remote_profile_bench_standalone_uses_case_id_runtime_helper -v`
Expected: FAIL because the current remote profile path still runs `msprof python3 bench_<op>.py --operator-file ...`.

- [ ] **Step 3: Add a failing IR-capture test that makes standalone benches use the helper script instead of direct benchmark execution**

```python
bench_file.write_text("# bench-mode: standalone\n", encoding="utf-8")
command = module.build_execution_command(
    bench_file=bench_file,
    operator_file=operator_file,
)
self.assertEqual(
    command,
    [
        sys.executable,
        "standalone_bench_runtime.py",
        "run-one",
        "--bench-file",
        "bench_matmul.py",
        "--operator-file",
        "matmul.py",
    ],
)
```

Run: `uv run python -m unittest tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_build_execution_command_uses_runtime_helper_for_standalone_benches -v`
Expected: FAIL because `capture_ir.py` still builds `python bench_<op>.py --operator-file ...`.

- [ ] **Step 4: Commit the remote and IR integration tests**

```bash
git add tests/test_remote_execution.py tests/test_profile_runner.py tests/test_ascend_operator_ir_analyzer.py
git commit -m "test: lock standalone remote and ir integration"
```

### Task 5: Implement the standalone runtime helper, wrappers, and IR integration

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/profile_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `skills/triton-npu-analyze-ir/scripts/capture_ir.py`

- [ ] **Step 1: Create the standalone runtime helper with hook loading, case validation, and a shared case dataclass**

```python
@dataclass(frozen=True)
class StandaloneBenchCase:
    case_id: str
    fn: Callable[[], object]
    warmup: int
    repeats: int


def load_standalone_bench_cases(
    bench_file: Path,
    operator_file: Path,
) -> tuple[list[StandaloneBenchCase], KernelResolution]:
    bench_module = _load_module(bench_file, "standalone_bench_module")
    operator_module = _load_module(operator_file, "standalone_operator_module")
    build_operator_api = _require_callable(bench_module, "build_operator_api")
    build_cases = _require_callable(bench_module, "build_standalone_bench_cases")
    operator_api = build_operator_api(operator_module)
    raw_cases = build_cases(operator_api)
    return _normalize_cases(raw_cases), resolve_bench_kernel_resolution(bench_file, operator_file)
```

Run: `uv run python -m unittest tests.test_standalone_bench_runtime.StandaloneBenchRuntimeTests.test_load_standalone_bench_cases_builds_hooked_cases -v`
Expected: PASS.

- [ ] **Step 2: Implement `run_local_standalone_bench`, `profile_local_standalone_case`, and helper CLI subcommands**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="standalone_bench_runtime.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--bench-file", required=True)
    run_all.add_argument("--operator-file", required=True)
    run_all.add_argument("--perf-file", required=True)

    profile_one = subparsers.add_parser("profile-one")
    profile_one.add_argument("--bench-file", required=True)
    profile_one.add_argument("--operator-file", required=True)
    profile_one.add_argument("--case-id", required=True)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--bench-file", required=True)
    run_one.add_argument("--operator-file", required=True)
    run_one.add_argument("--case-id")
```

Run: `uv run python -m unittest tests.test_standalone_bench_runtime -v`
Expected: PASS.

- [ ] **Step 3: Wire `bench_runner.py` and `profile_runner.py` to import and use the helper locally, and add `--case-id` to `run-command.py`**

```python
from standalone_bench_runtime import (
    profile_local_standalone_case,
    run_local_standalone_bench,
)


if bench_mode == "msprof":
    return _run_local_bench_msprof(bench_file, operator_file)
return run_local_standalone_bench(bench_file, operator_file)
```

```python
profile_bench.add_argument("--case-id")
```

Run: `uv run python -m unittest tests.test_bench_runner tests.test_profile_runner tests.test_skill_command_script -v`
Expected: PASS.

- [ ] **Step 4: Implement remote standalone delegation by copying the helper script into remote workspaces and invoking its CLI**

```python
helper_script = Path(__file__).resolve().with_name("standalone_bench_runtime.py")
copy_file_to_remote(
    spec,
    helper_script,
    f"{remote_workspace}/standalone_bench_runtime.py",
    verbose=verbose,
    stderr=stderr,
)
result = run_remote_command_streaming(
    spec,
    remote_workspace,
    [
        "python3",
        "standalone_bench_runtime.py",
        "run-all",
        "--bench-file",
        bench_file.name,
        "--operator-file",
        operator_file.name,
        "--perf-file",
        _perf_output_path(bench_file, operator_file).name,
    ],
    verbose=verbose,
    stderr=stderr,
)
copy_file_from_remote(
    spec,
    f"{remote_workspace}/{_perf_output_path(bench_file, operator_file).name}",
    _perf_output_path(bench_file, operator_file),
    verbose=verbose,
    stderr=stderr,
)
```

Run: `uv run python -m unittest tests.test_remote_execution tests.test_profile_runner -v`
Expected: PASS.

- [ ] **Step 5: Update `capture_ir.py` so standalone benches run through the helper script's `run-one` mode**

```python
if _resolve_bench_mode(bench_file) == "standalone":
    return [
        interpreter,
        "standalone_bench_runtime.py",
        "run-one",
        "--bench-file",
        bench_file.name,
        "--operator-file",
        operator_arg,
    ]
return [interpreter, bench_file.name, "--operator-file", operator_arg]
```

Run: `uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v`
Expected: PASS.

- [ ] **Step 6: Commit the runtime implementation**

```bash
git add \
  skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py \
  skills/triton-npu-run-eval/scripts/bench_runner.py \
  skills/triton-npu-run-eval/scripts/profile_runner.py \
  skills/triton-npu-run-eval/scripts/run-command.py \
  skills/triton-npu-analyze-ir/scripts/capture_ir.py
git commit -m "feat: add standalone bench profiler runtime"
```

### Task 6: Update docs and run verification

**Files:**
- Modify: `skills/triton-npu-gen-bench/SKILL.md`
- Modify: `skills/triton-npu-gen-bench/references/bench-standalone-spec.md`
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Modify: `skills/triton-npu-profile-operator/SKILL.md`
- Modify: `skills/triton-npu-analyze-ir/SKILL.md`
- Modify: `README.md`

- [ ] **Step 1: Rewrite the standalone benchmark spec and generation skill around hook exports instead of direct script execution**

```md
- standalone benchmark files must export `build_operator_api(operator_module)`
- standalone benchmark files must export `build_standalone_bench_cases(operator_api)`
- standalone benchmark files are import-only modules and do not need `main()`
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_benchmark_generation_specs_use_hooked_standalone_contract -v`
Expected: PASS.

- [ ] **Step 2: Update run-eval, profiler, IR-analysis, and README examples to use the new standalone semantics**

```md
- `profile-bench --bench-mode standalone` requires `--case-id <id>`
- remote standalone profiling uses the helper runtime instead of direct `msprof python bench_<op>.py ...`
- IR capture for standalone benches runs through `standalone_bench_runtime.py run-one`
```

Run: `uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_profiler_skill_documents_standalone_case_id_contract -v`
Expected: PASS.

- [ ] **Step 3: Run the focused unit-test suite**

Run: `uv run python -m unittest tests.test_generation_contracts tests.test_standalone_bench_runtime tests.test_bench_runner tests.test_profile_runner tests.test_remote_execution tests.test_ascend_operator_ir_analyzer tests.test_skill_command_script -v`
Expected: PASS.

- [ ] **Step 4: Run strict file-scoped pyright for every modified skill script**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/profile_runner.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py
```

Expected: `0 errors` for every command.

- [ ] **Step 5: Commit the doc and verification updates**

```bash
git add \
  skills/triton-npu-gen-bench/SKILL.md \
  skills/triton-npu-gen-bench/references/bench-standalone-spec.md \
  skills/triton-npu-run-eval/SKILL.md \
  skills/triton-npu-profile-operator/SKILL.md \
  skills/triton-npu-analyze-ir/SKILL.md \
  README.md
git commit -m "docs: update standalone bench workflow"
```
