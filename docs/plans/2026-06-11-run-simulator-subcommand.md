# Run-Simulator Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-only `run-simulator` CLI subcommand that executes one selected benchmark case under `msprof op simulator` without producing perf or profile artifacts.

**Architecture:** The CLI will register a new `run-simulator` command and route it to a new execution handler in `src/`. That handler will call a thin wrapper in `src/triton_agent/execution.py`, which will load a new internal helper module in `skills/triton-npu-run-eval/scripts/simulator_runner.py`. The helper will reuse the unified benchmark runtime (`bench_runtime.py`) for case selection and `bench_contract.py` for kernel resolution, then invoke `msprof op simulator` around `bench_runtime.py run-one`.

**Tech Stack:** Python 3, argparse, repository skill-loader bridge, existing run-eval benchmark runtime helpers, unittest, pyright, ruff, pytest

---

### Task 1: Add failing CLI and handler coverage

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_execution_commands.py`
- Create: `tests/test_simulator_runner.py`

- [ ] **Step 1: Add CLI parser tests for the new subcommand**

```python
def test_run_simulator_parser_accepts_required_and_optional_arguments(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-simulator",
            "--bench-file",
            "bench_kernel.py",
            "--operator-file",
            "kernel.py",
            "--case-id",
            "case-a",
            "--kernel-name",
            "KernelA",
        ]
    )

    self.assertEqual(args.command_kind, CommandKind.RUN_SIMULATOR)
    self.assertEqual(args.bench_file, "bench_kernel.py")
    self.assertEqual(args.operator_file, "kernel.py")
    self.assertEqual(args.case_id, "case-a")
    self.assertEqual(args.kernel_name, "KernelA")
```

- [ ] **Step 2: Add CLI parser tests that `run-simulator` omits unrelated run-bench options**

```python
def test_run_simulator_parser_does_not_accept_bench_mode_or_remote_options(self) -> None:
    parser = build_parser()
    with self.assertRaises(SystemExit):
        parser.parse_args(
            [
                "run-simulator",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--bench-mode",
                "msprof",
            ]
        )
```

- [ ] **Step 3: Add handler tests for successful local execution and error printing**

```python
def test_handle_run_simulator_returns_child_exit_code(self) -> None:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bench_file = root / "bench_kernel.py"
        operator = root / "kernel.py"
        bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")
        operator.write_text("print('kernel')\n", encoding="utf-8")
        args = parser.parse_args(
            [
                "run-simulator",
                "--bench-file",
                str(bench_file),
                "--operator-file",
                str(operator),
                "--case-id",
                "case-a",
                "--kernel-name",
                "KernelA",
            ]
        )

        fake_result = AgentResult(return_code=7, stdout="sim out\n", stderr="")
        with patch(
            "triton_agent.commands.execution.run_local_simulator",
            return_value=fake_result,
        ) as mocked:
            exit_code = handle_run_simulator(parser, args)

    self.assertEqual(exit_code, 7)
    mocked.assert_called_once_with(
        bench_file.resolve(),
        operator.resolve(),
        case_id="case-a",
        kernel_name="KernelA",
    )
```

- [ ] **Step 4: Add simulator helper tests for case and kernel selection**

```python
def test_run_local_simulator_defaults_single_case_and_single_kernel(self) -> None:
    module = load_simulator_runner_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bench_file = root / "bench_kernel.py"
        operator_file = root / "kernel.py"
        bench_file.write_text("# bench-mode: torch-npu-profiler\n# kernel: KernelA\n", encoding="utf-8")
        operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

        with patch.object(module, "_load_bench_runtime_module") as load_runtime, patch.object(
            module,
            "resolve_bench_kernel_resolution",
            return_value=type("_Resolution", (), {"kernel_names": ["KernelA"], "kernel_source": "metadata"})(),
        ), patch.object(
            module,
            "run_streaming_process",
            return_value=make_skill_result(0, "stdout\n", ""),
        ) as mocked:
            load_runtime.return_value = type(
                "_FakeRuntime",
                (),
                {
                    "load_bench_cases": staticmethod(lambda *_args, **_kwargs: ([type("_Case", (), {"case_id": "only-case"})()], None)),
                    "select_bench_case": staticmethod(lambda cases, _case_id: cases[0]),
                },
            )()
            result = module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name=None)

    self.assertEqual(result["return_code"], 0)
    self.assertIn("--kernel-name", mocked.call_args.args[0])
    self.assertIn("KernelA", mocked.call_args.args[0])
    self.assertIn("only-case", mocked.call_args.args[0])
```

- [ ] **Step 5: Add simulator helper tests for multi-case and multi-kernel validation failures**

```python
def test_run_local_simulator_requires_case_id_when_multiple_cases_exist(self) -> None:
    module = load_simulator_runner_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bench_file = root / "bench_kernel.py"
        operator_file = root / "kernel.py"
        bench_file.write_text("# bench-mode: msprof\n# kernels: KernelA\n", encoding="utf-8")
        operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

        with patch.object(module, "_load_bench_runtime_module") as load_runtime:
            load_runtime.return_value = type(
                "_FakeRuntime",
                (),
                {
                    "load_bench_cases": staticmethod(
                        lambda *_args, **_kwargs: (
                            [type("_Case", (), {"case_id": "case-a"})(), type("_Case", (), {"case_id": "case-b"})()],
                            None,
                        )
                    ),
                    "select_bench_case": staticmethod(lambda cases, case_id: (_ for _ in ()).throw(ValueError("Benchmark profiling requires --case-id when multiple cases exist. Available case ids: case-a, case-b"))),
                },
            )()
            with self.assertRaisesRegex(ValueError, "requires --case-id when multiple cases exist"):
                module.run_local_simulator(bench_file, operator_file, case_id=None, kernel_name="KernelA")
```

- [ ] **Step 6: Run the targeted tests to verify they fail before implementation**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py tests/test_execution_commands.py tests/test_simulator_runner.py`
Expected: FAIL because `run-simulator`, `handle_run_simulator`, `run_local_simulator`, and the simulator test loader do not exist yet.

- [ ] **Step 7: Commit the failing tests**

```bash
git add tests/test_cli.py tests/test_execution_commands.py tests/test_simulator_runner.py
git commit -m "test: add run-simulator coverage"
```

### Task 2: Implement the CLI command and runtime bridge

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/execution.py`
- Modify: `src/triton_agent/execution.py`

- [ ] **Step 1: Add the new command kind and empty skill mapping**

```python
class CommandKind(str, Enum):
    ...
    RUN_SIMULATOR = "run-simulator"
    ...


COMMAND_TO_SKILL = {
    ...
    CommandKind.RUN_SIMULATOR: "",
    ...
}
```

- [ ] **Step 2: Register the subcommand in the CLI with its own input mode**

```python
CommandKind.RUN_SIMULATOR: _CommandSpec(
    handler=handle_run_simulator,
    help_group="Execution",
    help_summary="Run one benchmark case under msprof op simulator.",
    description="Run one generated benchmark case under msprof op simulator against one operator file.",
    input_mode="run-simulator",
    has_output=False,
    has_verbose=False,
)
```

- [ ] **Step 3: Add primary argument wiring for `run-simulator`**

```python
if spec.input_mode == "run-simulator":
    subparser.add_argument("--bench-file", required=True)
    subparser.add_argument("--operator-file", required=True)
    subparser.add_argument("--case-id")
    subparser.add_argument("--kernel-name")
    return
```

- [ ] **Step 4: Add the execution handler that resolves paths and forwards to the wrapper**

```python
def handle_run_simulator(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file = resolve_run_bench_paths(parser, args)
    try:
        result = run_local_simulator(
            bench_file,
            operator_file,
            case_id=getattr(args, "case_id", None),
            kernel_name=getattr(args, "kernel_name", None),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.return_code
```

- [ ] **Step 5: Add the execution-layer protocol and wrapper for the simulator helper**

```python
class SimulatorRunnerModule(Protocol):
    def run_local_simulator(
        self,
        bench_file: Path,
        operator_file: Path,
        *,
        case_id: str | None = None,
        kernel_name: str | None = None,
    ) -> _RunSkillPayload: ...


def _load_simulator_runner() -> SimulatorRunnerModule:
    return cast(SimulatorRunnerModule, load_operator_eval_script_module("simulator_runner"))


def run_local_simulator(
    bench_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> AgentResult:
    result = _load_simulator_runner().run_local_simulator(
        bench_file,
        operator_file,
        case_id=case_id,
        kernel_name=kernel_name,
    )
    return _normalize_agent_result(result)
```

- [ ] **Step 6: Run targeted tests to verify the CLI and bridge pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py tests/test_execution_commands.py tests/test_simulator_runner.py`
Expected: FAIL only in simulator helper command-construction and validation tests because the skill-side runner still does not exist.

- [ ] **Step 7: Commit the CLI and bridge changes**

```bash
git add src/triton_agent/models.py src/triton_agent/cli.py src/triton_agent/commands/execution.py src/triton_agent/execution.py
git commit -m "feat: add run-simulator command surface"
```

### Task 3: Implement the simulator helper on top of the unified bench runtime

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/simulator_runner.py`
- Test: `tests/test_simulator_runner.py`

- [ ] **Step 1: Create the helper module with a narrow local-only API**

```python
from __future__ import annotations

import os
from pathlib import Path

from bench_contract import resolve_bench_kernel_resolution
from result_payload import ResultPayload
from run_runtime import env_int, local_python_executable, run_streaming_process


def run_local_simulator(
    bench_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> ResultPayload:
    ...
```

- [ ] **Step 2: Reuse `bench_runtime.py` case loading and selection instead of duplicating benchmark parsing**

```python
def _load_bench_runtime_module():
    script_path = Path(__file__).resolve().with_name("bench_runtime.py")
    ...


def _resolve_selected_case_id(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None,
) -> str:
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
    case = runtime.select_bench_case(cases, case_id)
    return str(case.case_id)
```

- [ ] **Step 3: Implement kernel validation and single-kernel inference**

```python
def _resolve_selected_kernel_name(
    bench_file: Path,
    operator_file: Path,
    kernel_name: str | None,
) -> str:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    kernel_names = resolution.kernel_names
    if kernel_name is not None:
        if kernel_name not in kernel_names:
            available = ", ".join(kernel_names)
            raise ValueError(f"Unknown simulator kernel '{kernel_name}'. Available kernel names: {available}")
        return kernel_name
    if len(kernel_names) == 1:
        return kernel_names[0]
    available = ", ".join(kernel_names)
    raise ValueError(
        "run-simulator requires --kernel-name when multiple kernels resolve. "
        f"Available kernel names: {available}"
    )
```

- [ ] **Step 4: Build the wrapped `msprof op simulator` command around `bench_runtime.py run-one`**

```python
def _simulator_timeout() -> int:
    return env_int("TRITON_AGENT_BENCH_TIMEOUT_SECONDS", 900)


def run_local_simulator(... ) -> ResultPayload:
    selected_case = _resolve_selected_case_id(bench_file, operator_file, case_id)
    selected_kernel = _resolve_selected_kernel_name(bench_file, operator_file, kernel_name)
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    command = [
        "msprof",
        "op",
        "simulator",
        "--soc-version=Ascend950PR_9599",
        "--kernel-name",
        selected_kernel,
        local_python_executable(),
        "bench_runtime.py",
        "run-one",
        "--bench-file",
        bench_file.name,
        "--operator-file",
        operator_arg,
        "--case-id",
        selected_case,
    ]
    return run_streaming_process(
        command,
        str(bench_file.parent),
        stall_timeout_seconds=_simulator_timeout(),
        extra_env={"TRITON_ALWAYS_COMPILE": "1"},
    )
```

- [ ] **Step 5: Make the simulator tests assert exact command construction**

```python
self.assertEqual(
    mocked.call_args.args[0],
    [
        "msprof",
        "op",
        "simulator",
        "--soc-version=Ascend950PR_9599",
        "--kernel-name",
        "KernelA",
        sys.executable,
        "bench_runtime.py",
        "run-one",
        "--bench-file",
        "bench_kernel.py",
        "--operator-file",
        "kernel.py",
        "--case-id",
        "only-case",
    ],
)
```

- [ ] **Step 6: Run the targeted tests to verify the helper passes**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py tests/test_execution_commands.py tests/test_simulator_runner.py`
Expected: PASS

- [ ] **Step 7: Run the required strict Pyright check for the new skill script**

Run: `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/simulator_runner.py`
Expected: PASS with no reported type errors.

- [ ] **Step 8: Commit the helper implementation**

```bash
git add skills/triton-npu-run-eval/scripts/simulator_runner.py tests/test_simulator_runner.py
git commit -m "feat: add simulator runner helper"
```

### Task 4: Run repository verification and keep the design doc in sync

**Files:**
- Modify: `docs/specs/2026-06-11-run-simulator-subcommand-design.md`

- [ ] **Step 1: Re-read the spec and implementation for drift**

```text
Confirm the shipped behavior still matches:
- local only
- ignores # bench-mode
- no perf/profile artifacts
- case selection from unified bench runtime
- kernel selection from benchmark kernel resolution
```

- [ ] **Step 2: Update the spec only if the implementation required a material contract change**

```markdown
If no contract changed, leave `docs/specs/2026-06-11-run-simulator-subcommand-design.md` untouched.
If a contract changed, edit the affected section so the spec remains the source of truth.
```

- [ ] **Step 3: Run focused and full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
Expected: PASS

- [ ] **Step 4: Commit the finished implementation**

```bash
git add docs/specs/2026-06-11-run-simulator-subcommand-design.md src/triton_agent/models.py src/triton_agent/cli.py src/triton_agent/commands/execution.py src/triton_agent/execution.py skills/triton-npu-run-eval/scripts/simulator_runner.py tests/test_cli.py tests/test_execution_commands.py tests/test_simulator_runner.py
git commit -m "feat: add run-simulator subcommand"
```
