# Run-Bench Output And Analysis Hints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--output` to both `run-bench` entrypoints, print an IR-inspection hint after successful `capture-ir` runs, and print a `profile-report` hint after successful `profile-bench` runs.

**Architecture:** Keep the main CLI behavior unchanged except for stronger regression coverage, because `src/triton_agent/commands/execution.py` already threads `args.output` into the bench runner. Implement the missing behavior in the skill-local script surfaces (`skills/triton-npu-run-eval/scripts/run-command.py` and `skills/triton-npu-analyze-ir/scripts/capture_ir.py`) with small helper functions so the new hint text stays centralized and testable.

**Tech Stack:** Python 3, `argparse`, repository unit tests under `tests/`, strict skill-script `pyright`, `pytest`, `ruff`

---

## File Map

- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
  - Owns the skill-local `run-bench`, `profile-bench`, and `profile-report` command surface.
- Modify: `skills/triton-npu-analyze-ir/scripts/capture_ir.py`
  - Owns the skill-local IR capture success output.
- Modify: `tests/test_cli.py`
  - Regression coverage for top-level `triton-agent run-bench`.
- Modify: `tests/test_skill_command_script.py`
  - Coverage for `skills/triton-npu-run-eval/scripts/run-command.py`.
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
  - Coverage for `skills/triton-npu-analyze-ir/scripts/capture_ir.py`.
- Modify: `README.md`
  - Top-level user-facing CLI docs for `run-bench`.
- Modify: `skills/triton-npu-run-eval/references/run-bench.md`
  - Skill-local `run-bench` usage and examples.
- Modify: `skills/triton-npu-run-eval/references/profile-bench.md`
  - Skill-local `profile-bench` follow-up guidance.
- Modify: `skills/triton-npu-analyze-ir/SKILL.md`
  - IR capture workflow guidance.

### Task 1: Align Skill-Local `run-bench --output` And `profile-bench` Hint Behavior

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add regression tests for CLI forwarding plus failing tests for the skill-local script**

Add a top-level CLI regression test in `tests/test_cli.py` near the existing `run-bench` tests:

```python
    def test_main_run_bench_threads_output_to_local_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "custom_perf.txt"
            operator.write_text("print('x')", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("triton_agent.commands.execution.run_local_bench", return_value=(fake_result, perf_file)) as mocked:
                exit_code = main(
                    [
                        "run-bench",
                        "--bench-file",
                        str(bench_file),
                        "--operator-file",
                        str(operator),
                        "--output",
                        str(perf_file),
                    ]
                )

            self.assertEqual(exit_code, 0)
            mocked.assert_called_once_with(
                bench_file.resolve(),
                operator.resolve(),
                "torch-npu-profiler",
                None,
                verbose=False,
                output=str(perf_file),
            )
```

Add these skill-local tests in `tests/test_skill_command_script.py` near the existing `run-bench` and help tests:

```python
    def test_script_run_bench_threads_output_to_local_runner(self) -> None:
        script = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
        spec = importlib.util.spec_from_file_location("run_command_test_output", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            perf_file = root / "custom_perf.txt"
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\nprint('bench')\n", encoding="utf-8")

            observed: list[object] = []

            def fake_run_local_bench(
                bench_path: Path,
                operator_path: Path,
                bench_mode: str,
                npu_devices: Optional[str] = None,
                **kwargs: object,
            ) -> tuple[dict[str, object], Path]:
                observed.extend([bench_path, operator_path, bench_mode, npu_devices, kwargs.get("output")])
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    perf_file,
                )

            with patch.object(
                module,
                "_load_bench_functions",
                return_value=(lambda _path: {"bench-mode": "msprof"}, fake_run_local_bench, lambda *_args, **_kwargs: None),
            ):
                exit_code = module.main(
                    [
                        "run-bench",
                        "--bench-file",
                        str(bench_file),
                        "--operator-file",
                        str(operator),
                        "--output",
                        str(perf_file),
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            observed,
            [bench_file.resolve(), operator.resolve(), "msprof", None, str(perf_file)],
        )

    def test_script_profile_bench_prints_profile_report_hint(self) -> None:
        script = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
        spec = importlib.util.spec_from_file_location("run_command_test_profile_hint", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            bench_file = root / "bench_kernel.py"
            profile_dir = root / "PROF_000001"
            operator.write_text("print('x')\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: torch-npu-profiler\nprint('bench')\n", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_profile_functions",
                    return_value=(
                        lambda *_args, **_kwargs: (
                            {
                                "return_code": 0,
                                "stdout": "",
                                "stderr": "",
                                "stalled": False,
                                "session_id": None,
                            },
                            profile_dir,
                        ),
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote runner should not be used")),
                    ),
                ), patch.object(module, "_build_profile_report", return_value="profile summary"):
                    exit_code = module.main(
                        [
                            "profile-bench",
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            (
                "Return code: 0\n"
                f"Profile directory: {profile_dir}\n"
                "profile summary\n"
                f"Hint: rerun the bundled `profile-report` helper for this `--profile-dir {profile_dir}` if you need the summary again; if that is not enough, inspect the raw files in this profile directory directly.\n"
            ),
        )
        self.assertEqual(stderr.getvalue(), "")
```

Also extend the existing `test_script_exposes_profile_bench_help` assertion so it checks `--output` on `run-bench`, not just `profile-bench` help:

```python
    def test_script_exposes_run_bench_help(self) -> None:
        script = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
        completed = subprocess.run(
            [sys.executable, str(script), "run-bench", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--bench-file", completed.stdout)
        self.assertIn("--operator-file", completed.stdout)
        self.assertIn("--output", completed.stdout)
```

- [ ] **Step 2: Run the focused tests and confirm the skill-local surface fails before implementation**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_cli.py::PathResolutionTests::test_main_run_bench_threads_output_to_local_runner \
  tests/test_skill_command_script.py::SkillCommandScriptTests::test_script_run_bench_threads_output_to_local_runner \
  tests/test_skill_command_script.py::SkillCommandScriptTests::test_script_profile_bench_prints_profile_report_hint \
  tests/test_skill_command_script.py::SkillCommandScriptTests::test_script_exposes_run_bench_help
```

Expected:
- `tests/test_cli.py::PathResolutionTests::test_main_run_bench_threads_output_to_local_runner` passes immediately, proving the top-level CLI already threads `--output`
- at least one skill-local test fails with either `unrecognized arguments: --output` or the missing `profile-report` hint

- [ ] **Step 3: Implement the missing script-local parser, forwarding, and hint helpers**

Update `skills/triton-npu-run-eval/scripts/run-command.py` like this:

```python
_RUN_BENCH_HINT = "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly."
_RUN_TEST_HINT = "Hint: use `compare-result` to inspect this archived result instead of reading it directly."


def _profile_bench_hint(profile_dir: Path) -> str:
    return (
        "Hint: rerun the bundled `profile-report` helper for this "
        f"`--profile-dir {profile_dir}` if you need the summary again; "
        "if that is not enough, inspect the raw files in this profile directory directly."
    )
```

```python
    run_bench = subparsers.add_parser("run-bench")
    run_bench.add_argument("--bench-file", required=True)
    run_bench.add_argument("--operator-file", required=True)
    run_bench.add_argument("--output")
    run_bench.add_argument("--remote")
    run_bench.add_argument("--remote-workdir")
    run_bench.add_argument("--keep-remote-workdir", action="store_true")
    run_bench.add_argument("--verbose", action="store_true")
    run_bench.add_argument("--bench-mode", choices=["torch-npu-profiler", "msprof"])
    run_bench.add_argument("--npu-devices")
```

```python
    if args.command == "profile-bench":
        run_local_profile_bench, run_remote_profile_bench = _load_profile_functions()
        bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
        operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
        resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(bench_file)
        remote_workspace: str | None = None
        try:
            if remote is not None:
                result, profile_dir, remote_workspace = run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    resolved_bench_mode,
                    remote,
                    remote_workdir,
                    case_id=args.case_id,
                    kernel_name=args.kernel_name,
                    keep_remote_workdir=args.keep_remote_workdir,
                    verbose=args.verbose,
                    stderr=sys.stderr,
                )
            else:
                result, profile_dir = run_local_profile_bench(
                    bench_file,
                    operator_file,
                    resolved_bench_mode,
                    case_id=args.case_id,
                    kernel_name=args.kernel_name,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _render_result(result, show_output=True)
        print(f"Return code: {result['return_code']}")
        if profile_dir is not None:
            print(f"Profile directory: {profile_dir}")
            print(_build_profile_report(profile_dir, args.target_op))
            print(_profile_bench_hint(profile_dir))
        if remote is not None and args.keep_remote_workdir:
            print(f"Remote workspace: {remote_workspace}")
        return int(result["return_code"])
```

```python
    if remote is not None:
        result, perf_path, remote_workspace = run_remote_bench(
            bench_file,
            operator_file,
            resolved_bench_mode,
            remote,
            remote_workdir,
            args.npu_devices,
            keep_remote_workdir=args.keep_remote_workdir,
            verbose=args.verbose,
            stderr=sys.stderr,
            output=args.output,
        )
    else:
        result, perf_path = run_local_bench(
            bench_file,
            operator_file,
            resolved_bench_mode,
            args.npu_devices,
            output=args.output,
        )
```

- [ ] **Step 4: Re-run the focused tests and the required strict skill-script pyright check**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_cli.py::PathResolutionTests::test_main_run_bench_threads_output_to_local_runner \
  tests/test_skill_command_script.py::SkillCommandScriptTests::test_script_run_bench_threads_output_to_local_runner \
  tests/test_skill_command_script.py::SkillCommandScriptTests::test_script_profile_bench_prints_profile_report_hint \
  tests/test_skill_command_script.py::SkillCommandScriptTests::test_script_exposes_run_bench_help

bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py
```

Expected:
- all four tests pass
- the strict pyright wrapper reports `0 errors`

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py tests/test_skill_command_script.py skills/triton-npu-run-eval/scripts/run-command.py
git commit -m "feat: align run-bench output and profile hints"
```

### Task 2: Add The `capture-ir` Success Hint

**Files:**
- Modify: `skills/triton-npu-analyze-ir/scripts/capture_ir.py`
- Modify: `tests/test_ascend_operator_ir_analyzer.py`

- [ ] **Step 1: Add a failing `capture-ir` success-output test**

Add this test to `tests/test_ascend_operator_ir_analyzer.py` near the parser and `main()`-style behavior coverage:

```python
    def test_main_prints_inspect_ir_hint_after_local_success(self) -> None:
        module = _load_capture_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "ir"
            bench_file = root / "bench.py"
            operator_file = root / "kernel.py"
            manifest_path = archive_dir / "capture-manifest.json"
            bench_file.write_text("print('bench')\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(module, "capture_local_archive", return_value=manifest_path):
                    exit_code = module.main(
                        [
                            "--ir-dir",
                            str(archive_dir),
                            "--bench-file",
                            str(bench_file),
                            "--operator-file",
                            str(operator_file),
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            (
                f"Capture manifest: {manifest_path}\n"
                f"Hint: use the bundled `inspect_ir.py` helper with `--ir-dir {archive_dir}` to inspect this archive first; if that is not enough, inspect bishengir_stages/, triton_dump/, all-ir.txt, and capture-manifest.json directly.\n"
            ),
        )
        self.assertEqual(stderr.getvalue(), "")
```

- [ ] **Step 2: Run the targeted test and confirm it fails because the hint is missing**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_ascend_operator_ir_analyzer.py::AscendOperatorIrAnalyzerTests::test_main_prints_inspect_ir_hint_after_local_success
```

Expected:
- the test fails on the `stdout` assertion because `capture_ir.py` currently prints only the `Capture manifest:` line

- [ ] **Step 3: Implement a small hint helper and print it on both success paths**

Update `skills/triton-npu-analyze-ir/scripts/capture_ir.py` like this:

```python
def _capture_ir_hint(archive_dir: Path) -> str:
    return (
        "Hint: use the bundled `inspect_ir.py` helper with "
        f"`--ir-dir {archive_dir}` to inspect this archive first; "
        "if that is not enough, inspect bishengir_stages/, triton_dump/, "
        "all-ir.txt, and capture-manifest.json directly."
    )
```

```python
        if args.remote:
            manifest_path, remote_workspace = capture_remote_archive(
                bench_file=bench_file,
                operator_file=operator_file,
                archive_dir=archive_dir,
                remote=args.remote,
                remote_workdir=args.remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                case_id=args.case_id,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
            print(f"Capture manifest: {manifest_path}")
            print(_capture_ir_hint(archive_dir))
            if args.keep_remote_workdir:
                print(f"Remote workspace: {remote_workspace}")
            return 0

        manifest_path = capture_local_archive(
            bench_file=bench_file,
            operator_file=operator_file,
            archive_dir=archive_dir,
            case_id=args.case_id,
        )
        print(f"Capture manifest: {manifest_path}")
        print(_capture_ir_hint(archive_dir))
        return 0
```

- [ ] **Step 4: Re-run the targeted test and the required strict skill-script pyright check**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_ascend_operator_ir_analyzer.py::AscendOperatorIrAnalyzerTests::test_main_prints_inspect_ir_hint_after_local_success

bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py
```

Expected:
- the targeted test passes
- the strict pyright wrapper reports `0 errors`

- [ ] **Step 5: Commit**

```bash
git add tests/test_ascend_operator_ir_analyzer.py skills/triton-npu-analyze-ir/scripts/capture_ir.py
git commit -m "feat: add capture-ir inspection hint"
```

### Task 3: Update User-Facing Docs To Match The New Behavior

**Files:**
- Modify: `README.md`
- Modify: `skills/triton-npu-run-eval/references/run-bench.md`
- Modify: `skills/triton-npu-run-eval/references/profile-bench.md`
- Modify: `skills/triton-npu-analyze-ir/SKILL.md`

- [ ] **Step 1: Update the CLI and skill docs with the new flags and follow-up guidance**

Apply these documentation edits:

```markdown
# README.md snippet
- `--output ./artifacts/a_perf.txt`: write the perf artifact to an explicit path instead of the default `a_perf.txt`.

uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py --output ./artifacts/a_perf.txt
```

```markdown
# skills/triton-npu-run-eval/references/run-bench.md snippet
python3 ./scripts/run-command.py run-bench --bench-file bench_a.py --operator-file a.py --output ./artifacts/a_perf.txt

- Use `--output ./artifacts/a_perf.txt` when you need the perf artifact at a specific location.
```

```markdown
# skills/triton-npu-run-eval/references/profile-bench.md snippet
After a successful `profile-bench`, rerun the bundled `profile-report` helper for `--profile-dir PROF_000001` to re-render the summary later.
If the summary is still not enough, inspect the raw files inside the copied-back `PROF_*` directory directly.
```

```markdown
# skills/triton-npu-analyze-ir/SKILL.md snippet
After `capture_ir.py` finishes, start with the bundled `inspect_ir.py` helper and `--ir-dir ir`.
If the helper output is not enough, inspect `bishengir_stages/`, `triton_dump/`, `all-ir.txt`, and `capture-manifest.json` directly.
```

- [ ] **Step 2: Sanity-check that every updated doc now mentions the new behavior explicitly**

Run:

```bash
rg -n -- "--output|profile-report --profile-dir|inspect_ir.py list-stages|raw files" \
  README.md \
  skills/triton-npu-run-eval/references/run-bench.md \
  skills/triton-npu-run-eval/references/profile-bench.md \
  skills/triton-npu-analyze-ir/SKILL.md
```

Expected:
- all four files appear in the output
- the matched lines show the new `--output` example plus the “inspect raw files directly” guidance

- [ ] **Step 3: Commit**

```bash
git add README.md skills/triton-npu-run-eval/references/run-bench.md skills/triton-npu-run-eval/references/profile-bench.md skills/triton-npu-analyze-ir/SKILL.md
git commit -m "docs: document output and follow-up hints"
```

### Task 4: Run The Final Verification Sweep

**Files:**
- Test: `tests/test_cli.py`
- Test: `tests/test_skill_command_script.py`
- Test: `tests/test_ascend_operator_ir_analyzer.py`
- Verify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Verify: `skills/triton-npu-analyze-ir/scripts/capture_ir.py`

- [ ] **Step 1: Re-run the focused test files together**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_cli.py \
  tests/test_skill_command_script.py \
  tests/test_ascend_operator_ir_analyzer.py
```

Expected:
- the three focused files pass with no new failures

- [ ] **Step 2: Re-run the required strict skill-script pyright checks together**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py
bash scripts/run-skill-script-pyright.sh skills/triton-npu-analyze-ir/scripts/capture_ir.py
```

Expected:
- both commands report `0 errors`

- [ ] **Step 3: Run the repository-standard verification commands**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected:
- all three commands complete successfully
- if any unrelated pre-existing failure appears, stop and document it before making further code changes
