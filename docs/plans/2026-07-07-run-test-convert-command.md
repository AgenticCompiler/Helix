# Run-Test Convert Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a skill-local `run-test-convert` command plus matching MCP tool so convert workflows stop borrowing baseline/optimize command names while preserving existing run-test execution behavior.

**Architecture:** Keep the repository-level `helix` CLI unchanged and implement the new surface entirely inside the staged run-eval helper and MCP server. Reuse the existing local/remote test runners, archived-result comparison helpers, and reference-result derivation flow; only the command routing, validation contract, MCP schema, and skill documentation should change.

**Tech Stack:** Python 3.11, `argparse`, `unittest`, FastMCP metadata registration, Markdown skill docs, existing `run_local_test` / `run_remote_test` helpers.

---

### Task 1: Add failing helper-CLI parser and validation tests

**Files:**
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add parser and help coverage for `run-test-convert`**

```python
def test_run_test_convert_parser_accepts_reference_flags(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "run-test-convert",
            "--test-file",
            "differential_test_kernel.py",
            "--operator-file",
            "triton_kernel.py",
            "--ref-operator-file",
            "kernel.py",
            "--test-mode",
            "differential",
        ]
    )

    self.assertEqual(args.command, "run-test-convert")
    self.assertEqual(args.ref_operator_file, "kernel.py")
    self.assertEqual(args.test_mode, "differential")


def test_script_exposes_run_test_convert_help(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "run-test-convert", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0)
    self.assertIn("usage: cli.py run-test-convert", completed.stdout)
    self.assertIn("--ref-result", completed.stdout)
    self.assertIn("--ref-operator-file", completed.stdout)
    self.assertIn("--baseline-operator-file", completed.stdout)
```

- [ ] **Step 2: Add convert-specific validation failures**

```python
def test_script_run_test_convert_requires_reference_input_in_differential_mode(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test_convert_requires_ref", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        operator = root / "triton_kernel.py"
        test_file = root / "differential_test_kernel.py"
        operator.write_text("print('x')\n", encoding="utf-8")
        test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

        stderr = StringIO()
        original_stderr = sys.stderr
        try:
            sys.stderr = stderr
            with self.assertRaises(SystemExit) as exc:
                module.main(
                    [
                        "run-test-convert",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                    ]
                )
        finally:
            sys.stderr = original_stderr

    self.assertEqual(exc.exception.code, 2)
    self.assertIn(
        "run-test-convert differential mode requires exactly one of --ref-result or --ref-operator-file",
        stderr.getvalue(),
    )


def test_script_run_test_convert_rejects_reference_inputs_in_standalone_mode(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test_convert_standalone", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        operator = root / "triton_kernel.py"
        test_file = root / "test_kernel.py"
        baseline_result = root / "kernel_result.pt"
        operator.write_text("print('x')\n", encoding="utf-8")
        test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")
        baseline_result.write_text("baseline\n", encoding="utf-8")

        stderr = StringIO()
        original_stderr = sys.stderr
        try:
            sys.stderr = stderr
            with self.assertRaises(SystemExit) as exc:
                module.main(
                    [
                        "run-test-convert",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                        "--ref-result",
                        str(baseline_result),
                    ]
                )
        finally:
            sys.stderr = original_stderr

    self.assertEqual(exc.exception.code, 2)
    self.assertIn("run-test-convert standalone mode does not accept --ref-result", stderr.getvalue())
```

- [ ] **Step 3: Run the targeted helper tests to verify they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_command_script.py -k "run_test_convert or run_test_optimize_requires_baseline_source"`

Expected: FAIL because `cli.py` does not yet register `run-test-convert`, help text does not mention it, and no convert-specific validation messages exist.

- [ ] **Step 4: Commit the failing-test scaffold once the task is later green**

```bash
git add tests/test_skill_command_script.py
git commit -m "test: cover run-test-convert parser validation"
```

### Task 2: Add failing helper-CLI runtime behavior tests

**Files:**
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add convert dispatch tests for differential comparison flows**

```python
def test_script_run_test_convert_auto_compares_when_ref_result_is_provided(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test_convert_compare", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        operator = root / "triton_kernel.py"
        test_file = root / "differential_test_kernel.py"
        archive = root / "triton_kernel_result.pt"
        baseline_result = root / "kernel_result.pt"
        operator.write_text("print('x')\n", encoding="utf-8")
        test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
        baseline_result.write_text("baseline\n", encoding="utf-8")

        def fake_run_local_test(
            test_path: Path,
            operator_path: Path,
            test_mode: str,
            verbose: bool = False,
            **_kwargs: object,
        ) -> tuple[dict[str, object], Path]:
            self.assertEqual(test_path, test_file.resolve())
            self.assertEqual(operator_path, operator.resolve())
            self.assertEqual(test_mode, "differential")
            return (
                {
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                },
                archive,
            )

        stdout = StringIO()
        stderr = StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            with patch.object(
                module,
                "_load_test_functions",
                return_value=(
                    lambda _path: {"test-mode": "differential"},
                    fake_run_local_test,
                    lambda *_args, **_kwargs: None,
                ),
            ), patch.object(
                module,
                "_load_compare_result_functions",
                return_value=(
                    lambda baseline_path, new_path: (
                        0
                        if baseline_path == baseline_result.resolve() and new_path == archive
                        else 2
                    ),
                    lambda *_args, **_kwargs: 0,
                ),
            ):
                exit_code = module.main(
                    [
                        "run-test-convert",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                        "--ref-result",
                        str(baseline_result),
                    ]
                )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    self.assertEqual(exit_code, 0)
    self.assertEqual(stdout.getvalue(), f"Return code: 0\nArchived result: {archive}\n")
    self.assertEqual(stderr.getvalue(), "")
```

- [ ] **Step 2: Add convert-only regression tests for “no optimize side effects”**

```python
def test_script_run_test_convert_does_not_append_active_round_timing_events(self) -> None:
    script = _REPO_ROOT / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
    spec = importlib.util.spec_from_file_location("run_command_test_convert_no_round_timing", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        round_dir = workspace / "opt-round-2"
        round_dir.mkdir()
        (workspace / ".helix").mkdir()
        (workspace / ".helix" / "state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "optimize-20260707-123456-abcdef",
                    "phase": "round_active",
                    "current_round": 2,
                    "baseline": {"status": "passed", "submitted_at": "2026-07-07T12:34:56Z"},
                    "rounds": {"2": {"status": "active", "round_dir": "opt-round-2"}},
                }
            ),
            encoding="utf-8",
        )
        test_file = workspace / "differential_test_kernel.py"
        operator_file = round_dir / "triton_kernel.py"
        timing_path = workspace / ".helix" / "round-timings" / "opt-round-2.jsonl"
        baseline_result = workspace / "kernel_result.pt"
        test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
        operator_file.write_text("print('kernel')\n", encoding="utf-8")
        baseline_result.write_text("baseline\n", encoding="utf-8")

        def fake_run_local_test(
            test_path: Path,
            candidate_operator_path: Path,
            test_mode: str,
            *,
            verbose: bool = False,
            **_kwargs: object,
        ) -> tuple[dict[str, object], Path]:
            self.assertEqual(test_path, test_file.resolve())
            self.assertEqual(candidate_operator_path, operator_file.resolve())
            self.assertEqual(test_mode, "differential")
            return (
                {
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                },
                workspace / "triton_kernel_result.pt",
            )

        with patch.object(
            module,
            "_load_test_functions",
            return_value=(
                lambda _path: {"test-mode": "differential"},
                fake_run_local_test,
                lambda *_args, **_kwargs: None,
            ),
        ), patch.object(
            module,
            "_load_compare_result_functions",
            return_value=(lambda *_args, **_kwargs: 0, lambda *_args, **_kwargs: 0),
        ):
            exit_code = module.main(
                [
                    "run-test-convert",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator_file),
                    "--ref-result",
                    str(baseline_result),
                ]
            )

    self.assertEqual(exit_code, 0)
    self.assertFalse(timing_path.exists())


def test_script_run_test_convert_preserves_pt_files_when_run_test_cleanup_policy_enabled(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test_convert_pt_cleanup", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        operator = root / "triton_kernel.py"
        test_file = root / "differential_test_kernel.py"
        archived_result = root / "triton_kernel_result.pt"
        baseline_result = root / "kernel_result.pt"
        operator.write_text("print('x')\n", encoding="utf-8")
        test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")
        archived_result.write_text("payload\n", encoding="utf-8")
        baseline_result.write_text("baseline\n", encoding="utf-8")

        def fake_run_local_test(
            test_path: Path,
            operator_path: Path,
            test_mode: str,
            verbose: bool = False,
            **_kwargs: object,
        ) -> tuple[dict[str, object], Path]:
            self.assertEqual(test_mode, "differential")
            return (
                {
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                },
                archived_result,
            )

        with patch.dict(
            os.environ,
            {"HELIX_OPTIMIZE_DELETE_PT_FILES": "run-test"},
            clear=False,
        ), patch.object(
            module,
            "_load_test_functions",
            return_value=(
                lambda _path: {"test-mode": "differential"},
                fake_run_local_test,
                lambda *_args, **_kwargs: None,
            ),
        ), patch.object(
            module,
            "_load_compare_result_functions",
            return_value=(lambda *_args, **_kwargs: 0, lambda *_args, **_kwargs: 0),
        ):
            exit_code = module.main(
                [
                    "run-test-convert",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                    "--ref-result",
                    str(baseline_result),
                ]
            )

    self.assertEqual(exit_code, 0)
    self.assertTrue(archived_result.exists())
```

- [ ] **Step 3: Add environment-guard coverage for the new command**

```python
def test_script_run_test_convert_guards_blocks_parallel_env(self) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "common"
        / "ascend-npu-run-eval"
        / "scripts"
        / "cli.py"
    )
    spec = importlib.util.spec_from_file_location("run_command_test_convert_guard", script)
    if spec is None or spec.loader is None:
        self.fail(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        operator = root / "triton_kernel.py"
        test_file = root / "test_kernel.py"
        operator.write_text("print('x')\n", encoding="utf-8")
        test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

        observed_env_values: list[Optional[str]] = []

        def fake_run_local_test(
            test_path: Path,
            operator_path: Path,
            test_mode: str,
            verbose: bool = False,
            **_kwargs: object,
        ) -> tuple[dict[str, object], None]:
            observed_env_values.append(os.environ.get("TRITON_ALL_BLOCKS_PARALLEL"))
            return (
                {
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                },
                None,
            )

        with patch.dict(os.environ, {"TRITON_ALL_BLOCKS_PARALLEL": "1"}, clear=False), patch.object(
            module,
            "_load_test_functions",
            return_value=(
                lambda _path: {"test-mode": "standalone"},
                fake_run_local_test,
                lambda *_args, **_kwargs: None,
            ),
        ):
            exit_code = module.main(
                [
                    "run-test-convert",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                ]
            )

    self.assertEqual(exit_code, 0)
    self.assertEqual(observed_env_values, ["0"])
```

- [ ] **Step 4: Run the focused runtime tests to verify they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_command_script.py -k "run_test_convert and not help"`

Expected: FAIL because `run-test-convert` is not dispatched, no convert-specific comparison path exists, the environment guard excludes the new command, and optimize-only timing/cleanup scoping is not yet exercised for convert.

- [ ] **Step 5: Commit the runtime test scaffold once the task is later green**

```bash
git add tests/test_skill_command_script.py
git commit -m "test: cover run-test-convert runtime behavior"
```

### Task 3: Add failing MCP and contract-doc tests

**Files:**
- Modify: `tests/test_run_eval_mcp_server.py`
- Modify: `tests/test_run_eval_mcp_server_tool_metadata.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add MCP tool-list and argument-forwarding tests**

```python
def test_server_registers_expected_tools(self) -> None:
    server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))

    async def _list_tool_names() -> list[str]:
        tools = await server.list_tools()
        return sorted(tool.name for tool in tools)

    self.assertEqual(
        asyncio.run(_list_tool_names()),
        [
            "compare-perf",
            "profile-bench",
            "profile-report",
            "run-bench",
            "run-test-baseline",
            "run-test-convert",
            "run-test-optimize",
        ],
    )


def test_run_test_convert_tool_forwards_reference_arguments(self) -> None:
    server = module.create_server(slot_pool=module.NpuDevicePool(("0",)))
    observed: dict[str, object] = {}

    def fake_run_subcommand(
        subcommand: str,
        arguments: list[str],
        *,
        leased_device: Optional[str] = None,
        workspace: Path,
    ):
        observed["subcommand"] = subcommand
        observed["arguments"] = arguments
        observed["leased_device"] = leased_device
        observed["workspace"] = workspace
        return {
            "return_code": 0,
            "stdout": "Archived result: /tmp/triton_kernel_result.pt\n",
            "stderr": "",
            "archived_result": "/tmp/triton_kernel_result.pt",
        }

    async def _call_tool() -> None:
        with (
            patch.object(module, "_run_subcommand", side_effect=fake_run_subcommand),
            patch.object(module, "current_workspace", return_value=Path("/tmp/ws")),
        ):
            await server.call_tool(
                "run-test-convert",
                {
                    "test_file": "/tmp/differential_test_kernel.py",
                    "operator_file": "/tmp/triton_kernel.py",
                    "test_mode": "differential",
                    "ref_operator_file": "/tmp/kernel.py",
                },
            )

    asyncio.run(_call_tool())

    self.assertEqual(observed["subcommand"], "run-test-convert")
    self.assertIn("--ref-operator-file", cast(list[str], observed["arguments"]))
```

- [ ] **Step 2: Add MCP metadata assertions for the new tool and narrower baseline description**

```python
self.assertEqual(
    tools["run-test-baseline"].description,
    "Run the baseline operator against a test case and return any archived differential result it produces.",
)
self.assertNotIn("ref_result", tools["run-test-baseline"].parameters["properties"])
self.assertNotIn("ref_operator_file", tools["run-test-baseline"].parameters["properties"])

self.assertEqual(
    tools["run-test-convert"].description,
    "Run the converted operator against a test case and compare it with reference evidence.",
)
self.assertEqual(
    tools["run-test-convert"].parameters["properties"]["ref_operator_file"]["description"],
    "Absolute path to the reference operator file used to produce comparison output.",
)
self.assertIn("ref_result", tools["run-test-convert"].parameters["properties"])
```

- [ ] **Step 3: Add contract-doc assertions for the new command wording**

```python
self.assertIn("run-test-convert", run_test)
self.assertIn("run-test-convert", skill)
self.assertIn("run-test-convert", _read("skills/common/ascend-npu-run-eval-mcp/SKILL.md"))

convert_skill = _read("skills/triton/triton-npu-convert-pytorch-operator/SKILL.md")
self.assertIn("run-test-convert", convert_skill)
self.assertIn("--ref-operator-file <original>", convert_skill)
self.assertNotIn("run-test-optimize` with `--baseline-operator-file <original>`", convert_skill)
```

- [ ] **Step 4: Run the focused MCP and contract tests to verify they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_run_eval_mcp_server.py tests/test_run_eval_mcp_server_tool_metadata.py tests/test_generation_contracts.py -k "run_test_convert or run_eval"`

Expected: FAIL because the MCP server does not yet register `run-test-convert`, metadata has no convert tool, and the run-eval / convert skill docs still only mention baseline and optimize.

- [ ] **Step 5: Commit the MCP/doc test scaffold once the task is later green**

```bash
git add tests/test_run_eval_mcp_server.py tests/test_run_eval_mcp_server_tool_metadata.py tests/test_generation_contracts.py
git commit -m "test: cover run-test-convert mcp and docs"
```

### Task 4: Implement the helper CLI command and validation rules

**Files:**
- Modify: `skills/common/ascend-npu-run-eval/scripts/cli.py`

- [ ] **Step 1: Register the new subcommand and guard it like other operator-execution commands**

```python
run_test_baseline = subparsers.add_parser("run-test-baseline")
_add_run_test_arguments(run_test_baseline)

run_test_convert = subparsers.add_parser("run-test-convert")
_add_run_test_arguments(run_test_convert)

run_test_optimize = subparsers.add_parser("run-test-optimize")
_add_run_test_arguments(run_test_optimize)


@contextlib.contextmanager
def _guard_operator_execution_env(command: str) -> Iterator[None]:
    if command not in {
        "run-test-baseline",
        "run-test-convert",
        "run-test-optimize",
        "run-bench",
        "profile-bench",
    }:
        yield
        return
```

- [ ] **Step 2: Expand run-test dispatch without leaking optimize-only side effects**

```python
if args.command in {"run-test-baseline", "run-test-convert", "run-test-optimize"}:
    _parse_test_metadata, run_local_test, run_remote_test = _load_test_functions()
    test_file = _resolve_existing_path(parser, args.test_file, "Test file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    timing_context = (
        _active_optimize_round_context(test_file, operator_file)
        if args.command == "run-test-optimize"
        else None
    )
    ref_result = _resolve_optional_existing_path(
        parser, getattr(args, "ref_result", None), "Reference result"
    )
    ref_operator_file = _resolve_optional_existing_path(
        parser, getattr(args, "ref_operator_file", None), "Reference operator file"
    )
    resolved_test_mode = args.test_mode or _resolve_test_mode_from_metadata(test_file)
    strict_reference_mode = args.command in {"run-test-convert", "run-test-optimize"}
    ref_result = _resolve_run_test_comparison_inputs(
        parser,
        args,
        resolved_test_mode,
        ref_result,
        ref_operator_file,
        test_file,
        run_local_test,
        run_remote_test,
        remote,
        remote_workdir,
        command_name=args.command,
        require_reference_input=strict_reference_mode,
    )
    remote_workspace: str | None = None
    _append_optimize_timing_event(
        timing_context,
        event="run_test_start",
        command=args.command,
        test_file=test_file,
        operator_file=operator_file,
    )
    if remote is not None:
        result, archived_result, remote_workspace = run_remote_test(
            test_file,
            operator_file,
            resolved_test_mode,
            remote,
            remote_workdir,
            keep_remote_workdir=args.keep_remote_workdir,
            verbose=args.verbose,
            stderr=sys.stderr,
        )
    else:
        result, archived_result = run_local_test(
            test_file,
            operator_file,
            resolved_test_mode,
            verbose=args.verbose,
        )
    if args.command == "run-test-optimize":
        _cleanup_run_test_pt_files((archived_result,))
```

- [ ] **Step 3: Parameterize validation and error messages by command name**

```python
def _resolve_run_test_comparison_inputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    resolved_test_mode: str,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    test_file: Path,
    run_local_test: RunLocalTestFn,
    run_remote_test: RunRemoteTestFn,
    remote: str | None,
    remote_workdir: str | None,
    *,
    command_name: str,
    require_reference_input: bool,
) -> Path | None:
    _validate_run_test_comparison_inputs(
        parser,
        command_name,
        resolved_test_mode,
        ref_result,
        ref_operator_file,
        require_reference_input=require_reference_input,
    )
    if ref_operator_file is None:
        return ref_result
    return _resolve_ref_operator_result(
        test_file,
        ref_operator_file,
        resolved_test_mode,
        run_local_test,
        run_remote_test,
        remote,
        remote_workdir,
        keep_remote_workdir=bool(args.keep_remote_workdir),
        verbose=bool(args.verbose),
    )


def _validate_run_test_comparison_inputs(
    parser: argparse.ArgumentParser,
    command_name: str,
    resolved_test_mode: str,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    *,
    require_reference_input: bool,
) -> None:
    if ref_result is not None and resolved_test_mode != "differential":
        parser.error(f"{command_name} standalone mode does not accept --ref-result")
    if ref_operator_file is not None and resolved_test_mode != "differential":
        parser.error(f"{command_name} standalone mode does not accept --ref-operator-file")
    if ref_result is not None and ref_operator_file is not None:
        if require_reference_input:
            parser.error(
                f"{command_name} differential mode requires exactly one of "
                "--ref-result or --ref-operator-file"
            )
        parser.error(
            f"{command_name} differential mode accepts at most one of "
            "--ref-result or --ref-operator-file"
        )
    if require_reference_input and resolved_test_mode == "differential" and ref_result is None and ref_operator_file is None:
        parser.error(
            f"{command_name} differential mode requires exactly one of "
            "--ref-result or --ref-operator-file"
        )
```

- [ ] **Step 4: Run the helper-CLI targeted tests to verify they pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_command_script.py -k "run_test_convert or run_test_optimize or blocks_parallel"`

Expected: PASS

- [ ] **Step 5: Commit the helper-CLI implementation**

```bash
git add skills/common/ascend-npu-run-eval/scripts/cli.py tests/test_skill_command_script.py
git commit -m "feat: add run-test-convert helper command"
```

### Task 5: Implement the MCP tool, update docs, and run verification

**Files:**
- Modify: `src/helix/eval/mcp_server.py`
- Modify: `skills/common/ascend-npu-run-eval/SKILL.md`
- Modify: `skills/common/ascend-npu-run-eval/references/run-test.md`
- Modify: `skills/common/ascend-npu-run-eval-mcp/SKILL.md`
- Modify: `skills/triton/triton-npu-convert-pytorch-operator/SKILL.md`
- Modify: `skills/tilelang/tilelang-npu-convert-pytorch-operator/SKILL.md`
- Modify: `tests/test_run_eval_mcp_server.py`
- Modify: `tests/test_run_eval_mcp_server_tool_metadata.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add the new MCP tool and align metadata wording**

```python
@server.tool(
    name="run-test-baseline",
    description="Run the baseline operator against a test case and return any archived differential result it produces.",
)
def run_test_baseline(
    test_file: Annotated[str, Field(description="Absolute path to the test entry file.")],
    operator_file: Annotated[str, Field(description="Absolute path to the operator implementation file.")],
    test_mode: Annotated[
        str | None,
        Field(description="Optional test mode override. Supported values: standalone, differential."),
    ] = None,
    remote: Annotated[str | None, Field(description="Optional remote execution target.")] = None,
    remote_workdir: Annotated[str | None, Field(description="Optional remote workspace root override.")] = None,
) -> dict[str, object]:
    workspace = current_workspace()
    arguments = _build_run_test_arguments(
        test_file=test_file,
        operator_file=operator_file,
        test_mode=test_mode,
        ref_result=None,
        ref_operator_file=None,
        remote=remote,
        remote_workdir=remote_workdir,
        keep_remote_workdir=False,
        verbose=False,
    )
    with _lease_device(pool) as leased_device:
        return _run_subcommand(
            "run-test-baseline",
            arguments,
            leased_device=leased_device,
            workspace=workspace,
        )


@server.tool(
    name="run-test-convert",
    description="Run the converted operator against a test case and compare it with reference evidence.",
)
def run_test_convert(
    test_file: Annotated[str, Field(description="Absolute path to the test entry file.")],
    operator_file: Annotated[str, Field(description="Absolute path to the converted operator implementation file.")],
    ref_operator_file: Annotated[
        str | None,
        Field(description="Absolute path to the reference operator file used to produce comparison output."),
    ] = None,
    ref_result: Annotated[
        str | None,
        Field(description="Absolute path to an archived reference result used for differential comparison."),
    ] = None,
    test_mode: Annotated[
        str | None,
        Field(description="Optional test mode override. Supported values: standalone, differential."),
    ] = None,
    remote: Annotated[str | None, Field(description="Optional remote execution target.")] = None,
    remote_workdir: Annotated[str | None, Field(description="Optional remote workspace root override.")] = None,
) -> dict[str, object]:
    workspace = current_workspace()
    arguments = _build_run_test_arguments(
        test_file=test_file,
        operator_file=operator_file,
        test_mode=test_mode,
        ref_result=ref_result,
        ref_operator_file=ref_operator_file,
        remote=remote,
        remote_workdir=remote_workdir,
        keep_remote_workdir=False,
        verbose=False,
    )
    with _lease_device(pool) as leased_device:
        return _run_subcommand(
            "run-test-convert",
            arguments,
            leased_device=leased_device,
            workspace=workspace,
        )
```

- [ ] **Step 2: Update run-eval and convert skill docs to use the new command names**

```markdown
- `run-test-baseline` / `run-test-convert` / `run-test-optimize`: [references/run-test.md](references/run-test.md)
```

```markdown
# `run-test-baseline`, `run-test-convert`, and `run-test-optimize`

Use `run-test-baseline` for baseline or generation validation, `run-test-convert` for convert validation, and `run-test-optimize` for optimize-round validation.

python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert --test-file differential_test_<operator>.py --operator-file triton_<operator>.py --test-mode differential --ref-operator-file <operator>.py

- standalone mode never accepts `--ref-result` or `--ref-operator-file`
- `run-test-baseline` differential mode may omit both reference flags to produce a reusable archived baseline result
- `run-test-convert` differential mode requires exactly one of `--ref-result` or `--ref-operator-file`
- `run-test-optimize` differential mode requires exactly one of `--ref-result` or `--ref-operator-file`
```

```markdown
- `run-test-baseline`
- `run-test-convert`
- `run-test-optimize`
```

```markdown
- **Differential mode**: `cli.py run-test-convert` with `--ref-operator-file <original>`
- **Standalone mode**: `cli.py run-test-convert`
```

- [ ] **Step 3: Run focused MCP, docs, and skill-contract verification**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_run_eval_mcp_server.py tests/test_run_eval_mcp_server_tool_metadata.py tests/test_generation_contracts.py -k "run_test_convert or run_eval"`

Expected: PASS

Run: `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/cli.py`

Expected: `0 errors`

- [ ] **Step 4: Run repository verification commands**

Run: `uv run --group dev ruff check`

Expected: PASS

Run: `uv run pyright`

Expected: PASS

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: PASS

- [ ] **Step 5: Commit the MCP/doc implementation**

```bash
git add src/helix/eval/mcp_server.py skills/common/ascend-npu-run-eval/SKILL.md skills/common/ascend-npu-run-eval/references/run-test.md skills/common/ascend-npu-run-eval-mcp/SKILL.md skills/triton/triton-npu-convert-pytorch-operator/SKILL.md skills/tilelang/tilelang-npu-convert-pytorch-operator/SKILL.md tests/test_run_eval_mcp_server.py tests/test_run_eval_mcp_server_tool_metadata.py tests/test_generation_contracts.py
git commit -m "feat: add run-test-convert validation surface"
```
