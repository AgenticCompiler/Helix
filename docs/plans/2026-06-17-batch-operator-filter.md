# Batch Operator Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared `--operator-filter <glob>` option to `optimize-batch`, `gen-eval-batch`, and `convert-batch` so users can further narrow batch operator file selection by basename after the existing built-in candidate exclusions.

**Architecture:** Keep the CLI thin and treat this as batch workspace-selection behavior instead of workflow runtime behavior. Add one shared basename-glob filtering hook in `src/triton_agent/batch_utils.py`, wire the parsed option only through the three batch command paths, and preserve the existing `0 / 1 / many` candidate resolution contract with clearer filter-aware errors.

**Tech Stack:** Python 3, `argparse`, `fnmatch`, `unittest`, existing batch helper modules

---

## File Structure

- Modify: `src/triton_agent/cli.py`
  - Add a command-spec capability bit for the new batch-only option and register `--operator-filter` only on the three supported batch commands.
- Modify: `src/triton_agent/batch_utils.py`
  - Add shared basename glob filtering support and filter-aware error wording to the batch operator resolver.
- Modify: `src/triton_agent/optimize/naming.py`
  - Thread the optional filter into optimize batch operator resolution.
- Modify: `src/triton_agent/generation/batch.py`
  - Thread the parsed option into batch workspace discovery and add helper-level coverage for filtered selection.
- Modify: `src/triton_agent/convert/batch.py`
  - Thread the parsed option into batch workspace discovery and preserve convert-specific candidate exclusions.
- Modify: `src/triton_agent/commands/optimize.py`
  - Forward the parsed CLI option into `run_optimize_batch(...)`.
- Modify: `src/triton_agent/commands/generation.py`
  - Forward the parsed CLI option into `run_gen_eval_batch(...)`.
- Modify: `src/triton_agent/commands/convert.py`
  - Forward the parsed CLI option into `run_convert_batch(...)`.
- Test: `tests/test_cli.py`
  - Cover parser acceptance and handler forwarding for the new flag.
- Test: `tests/test_generation_batch.py`
  - Cover helper behavior and generation-batch filtering behavior.
- Test: `tests/test_convert_commands.py`
  - Cover convert-batch filtering behavior and the convert-specific `triton_*` exclusion interaction.
- Test: `tests/test_optimize_runtime.py`
  - Cover optimize-batch filtering behavior.

### Task 1: Add CLI parser coverage for `--operator-filter`

**Files:**
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser tests**

Add parser coverage near the existing batch option tests:

```python
def test_gen_eval_batch_accepts_operator_filter(self) -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["gen-eval-batch", "-i", "kernels", "--operator-filter", "kernel*.py"]
    )

    self.assertEqual(args.operator_filter, "kernel*.py")


def test_convert_batch_accepts_operator_filter(self) -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["convert-batch", "-i", "kernels", "--operator-filter", "kernel*.py"]
    )

    self.assertEqual(args.operator_filter, "kernel*.py")


def test_optimize_batch_accepts_operator_filter(self) -> None:
    parser = build_parser()

    args = parser.parse_args(
        ["optimize-batch", "-i", "kernels", "--operator-filter", "kernel*.py"]
    )

    self.assertEqual(args.operator_filter, "kernel*.py")
```

- [ ] **Step 2: Run the focused parser tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_gen_eval_batch_accepts_operator_filter \
  tests.test_cli.CliParserTests.test_convert_batch_accepts_operator_filter \
  tests.test_cli.CliParserTests.test_optimize_batch_accepts_operator_filter -v
```

Expected: FAIL because `--operator-filter` is not registered yet.

- [ ] **Step 3: Implement the parser flag**

In `src/triton_agent/cli.py`, add one command-spec capability for the new batch-only option and register it in `build_parser()`:

```python
@dataclass(frozen=True)
class _CommandSpec:
    ...
    has_operator_filter: bool = False
```

Enable it only on:

```python
CommandKind.GEN_EVAL_BATCH: _CommandSpec(
    ...
    has_operator_filter=True,
),
CommandKind.CONVERT_BATCH: _CommandSpec(
    ...
    has_operator_filter=True,
),
CommandKind.OPTIMIZE_BATCH: _CommandSpec(
    ...
    has_operator_filter=True,
),
```

Then register the flag:

```python
if spec.has_operator_filter:
    subparser.add_argument(
        "--operator-filter",
        help="Shell-style glob matched against the selected operator filename basename after built-in exclusions.",
    )
```

- [ ] **Step 4: Run the focused parser tests and confirm they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_gen_eval_batch_accepts_operator_filter \
  tests.test_cli.CliParserTests.test_convert_batch_accepts_operator_filter \
  tests.test_cli.CliParserTests.test_optimize_batch_accepts_operator_filter -v
```

Expected: PASS.

- [ ] **Step 5: Commit the parser surface change**

```bash
git add src/triton_agent/cli.py tests/test_cli.py
git commit -m "feat: add batch operator filter cli flag"
```

### Task 2: Add shared helper tests for basename glob filtering

**Files:**
- Modify: `tests/test_generation_batch.py`
- Modify: `src/triton_agent/batch_utils.py`
- Modify: `src/triton_agent/generation/batch.py`

- [ ] **Step 1: Write the failing shared-helper tests**

Add helper-level tests in `tests/test_generation_batch.py`:

```python
def test_resolve_batch_gen_eval_operator_file_applies_operator_filter(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel.py").write_text("print('a')\n", encoding="utf-8")
        (workspace / "kernel_fp16.py").write_text("print('b')\n", encoding="utf-8")

        resolved = resolve_batch_gen_eval_operator_file(
            workspace,
            operator_filter="*_fp16.py",
        )

        self.assertEqual(resolved, workspace / "kernel_fp16.py")


def test_resolve_batch_gen_eval_operator_file_reports_no_match_after_filter(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel.py").write_text("print('a')\n", encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            r"found no candidate operator file after applying --operator-filter 'triton_\\*\\.py'",
        ):
            resolve_batch_gen_eval_operator_file(
                workspace,
                operator_filter="triton_*.py",
            )


def test_resolve_batch_gen_eval_operator_file_reports_multiple_matches_after_filter(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel_a.py").write_text("print('a')\n", encoding="utf-8")
        (workspace / "kernel_b.py").write_text("print('b')\n", encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            r"found multiple candidate operator files after applying --operator-filter 'kernel_\\*\\.py':",
        ):
            resolve_batch_gen_eval_operator_file(
                workspace,
                operator_filter="kernel_*.py",
            )
```

- [ ] **Step 2: Run the focused helper tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_resolve_batch_gen_eval_operator_file_applies_operator_filter \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_resolve_batch_gen_eval_operator_file_reports_no_match_after_filter \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_resolve_batch_gen_eval_operator_file_reports_multiple_matches_after_filter -v
```

Expected: FAIL because the resolver does not accept `operator_filter` yet.

- [ ] **Step 3: Implement shared basename filtering in the batch helper**

In `src/triton_agent/batch_utils.py`, add basename-glob filtering and filter-aware errors:

```python
from fnmatch import fnmatchcase
```

```python
def resolve_batch_operator_file(
    workspace: Path,
    *,
    is_operator_candidate: Callable[[Path], bool],
    no_candidate_message: str = NO_CANDIDATE_OPERATOR_FILE,
    operator_filter: str | None = None,
) -> Path:
    candidates = [
        path for path in sorted(workspace.iterdir()) if path.is_file() and is_operator_candidate(path)
    ]
    if operator_filter is not None:
        candidates = [path for path in candidates if fnmatchcase(path.name, operator_filter)]
    if not candidates:
        if operator_filter is not None:
            raise ValueError(
                f"found no candidate operator file after applying --operator-filter {operator_filter!r}"
            )
        raise ValueError(no_candidate_message)
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        if operator_filter is not None:
            raise ValueError(
                "found multiple candidate operator files after applying "
                f"--operator-filter {operator_filter!r}: {names}"
            )
        raise ValueError(f"found multiple candidate operator files: {names}")
    return candidates[0]
```

Then update `resolve_batch_gen_eval_operator_file(...)` in `src/triton_agent/generation/batch.py`:

```python
def resolve_batch_gen_eval_operator_file(
    workspace: Path,
    *,
    operator_filter: str | None = None,
) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_gen_eval_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
        operator_filter=operator_filter,
    )
```

- [ ] **Step 4: Run the focused helper tests and confirm they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_resolve_batch_gen_eval_operator_file_applies_operator_filter \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_resolve_batch_gen_eval_operator_file_reports_no_match_after_filter \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_resolve_batch_gen_eval_operator_file_reports_multiple_matches_after_filter -v
```

Expected: PASS.

- [ ] **Step 5: Commit the shared helper change**

```bash
git add src/triton_agent/batch_utils.py src/triton_agent/generation/batch.py tests/test_generation_batch.py
git commit -m "feat: add batch operator basename filtering"
```

### Task 3: Forward `--operator-filter` through the generation batch command

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/commands/generation.py`
- Modify: `src/triton_agent/generation/batch.py`

- [ ] **Step 1: Write the failing generation handler forwarding test**

Add a command-dispatch test in `tests/test_cli.py`:

```python
def test_main_gen_eval_batch_forwards_operator_filter(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root.mkdir(exist_ok=True)
        captured: dict[str, object] = {}

        def _fake_run(root_path, options, max_concurrency, operator_filter=None):
            del options
            captured["root"] = root_path
            captured["max_concurrency"] = max_concurrency
            captured["operator_filter"] = operator_filter
            return 0

        with patch(
            "triton_agent.commands.generation.run_gen_eval_batch",
            side_effect=_fake_run,
        ):
            exit_code = main(
                [
                    "gen-eval-batch",
                    "-i",
                    str(root),
                    "--operator-filter",
                    "kernel*.py",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["root"], root.resolve())
        self.assertEqual(captured["max_concurrency"], 1)
        self.assertEqual(captured["operator_filter"], "kernel*.py")
```

- [ ] **Step 2: Run the focused generation forwarding test and confirm it fails**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliMainTests.test_main_gen_eval_batch_forwards_operator_filter -v
```

Expected: FAIL because `run_gen_eval_batch(...)` does not accept or receive the argument yet.

- [ ] **Step 3: Implement generation batch forwarding**

Update `src/triton_agent/commands/generation.py`:

```python
return run_gen_eval_batch(
    root,
    generation_options_from_args(args),
    max_concurrency=max_concurrency,
    operator_filter=getattr(args, "operator_filter", None),
)
```

Update `src/triton_agent/generation/batch.py`:

```python
def run_gen_eval_batch(
    root: Path,
    options: GenerationOptions,
    *,
    max_concurrency: int,
    operator_filter: str | None = None,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
```

And pass it into workspace discovery:

```python
discovered, failures = discover_batch_workspaces(
    root,
    resolve_operator_file=lambda workspace: resolve_batch_gen_eval_operator_file(
        workspace,
        operator_filter=operator_filter,
    ),
    no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
)
```

- [ ] **Step 4: Run the focused generation forwarding test and one existing regression**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliMainTests.test_main_gen_eval_batch_forwards_operator_filter \
  tests.test_generation_batch.GenerationBatchHelpersTests.test_run_gen_eval_batch_applies_user_prompt_to_each_workspace_request -v
```

Expected: PASS.

- [ ] **Step 5: Commit the generation batch wiring**

```bash
git add src/triton_agent/commands/generation.py src/triton_agent/generation/batch.py tests/test_cli.py
git commit -m "feat: wire operator filter through gen-eval-batch"
```

### Task 4: Forward `--operator-filter` through the convert batch command

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `src/triton_agent/commands/convert.py`
- Modify: `src/triton_agent/convert/batch.py`

- [ ] **Step 1: Write the failing convert tests**

Add a CLI forwarding test in `tests/test_cli.py`:

```python
def test_main_convert_batch_forwards_operator_filter(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captured: dict[str, object] = {}

        def _fake_run(root_path, options, max_concurrency, operator_filter=None):
            del options
            captured["root"] = root_path
            captured["max_concurrency"] = max_concurrency
            captured["operator_filter"] = operator_filter
            return 0

        with patch(
            "triton_agent.commands.convert.run_convert_batch",
            side_effect=_fake_run,
        ):
            exit_code = main(
                [
                    "convert-batch",
                    "-i",
                    str(root),
                    "--operator-filter",
                    "kernel*.py",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["root"], root.resolve())
        self.assertEqual(captured["max_concurrency"], 1)
        self.assertEqual(captured["operator_filter"], "kernel*.py")
```

Add a convert-specific behavior test in `tests/test_convert_commands.py`:

```python
def test_run_convert_batch_operator_filter_does_not_reinclude_triton_prefixed_files(self) -> None:
    from triton_agent.convert.batch import run_convert_batch

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "kernel_workspace"
        workspace.mkdir()
        (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
        (workspace / "triton_kernel.py").write_text("print('y')\n", encoding="utf-8")

        stream = StringIO()
        exit_code = run_convert_batch(
            root,
            ConvertOptions(
                interact=False,
                verbose=False,
                stream_output=False,
                force_overwrite=False,
                agent_name="codex",
                remote=None,
                remote_workdir=None,
                output=None,
                test_mode="differential",
                prompt=None,
            ),
            max_concurrency=1,
            operator_filter="triton_*.py",
            stdout=stream,
            run_request=lambda request, stdout=None, stderr=None: AgentResult(
                return_code=0,
                stdout="ok",
                stderr="",
            ),
        )

        self.assertEqual(exit_code, 1)
        self.assertIn(
            "found no candidate operator file after applying --operator-filter 'triton_*.py'",
            stream.getvalue(),
        )
```

- [ ] **Step 2: Run the focused convert tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliMainTests.test_main_convert_batch_forwards_operator_filter \
  tests.test_convert_commands.ConvertRuntimeTests.test_run_convert_batch_operator_filter_does_not_reinclude_triton_prefixed_files -v
```

Expected: FAIL because the convert batch path does not accept or forward the option yet.

- [ ] **Step 3: Implement convert batch forwarding**

Update `src/triton_agent/commands/convert.py`:

```python
return run_convert_batch(
    root,
    convert_options_from_args(args),
    max_concurrency=max_concurrency,
    operator_filter=getattr(args, "operator_filter", None),
)
```

Update `src/triton_agent/convert/batch.py`:

```python
def run_convert_batch(
    root: Path,
    options: ConvertOptions,
    *,
    max_concurrency: int,
    operator_filter: str | None = None,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
```

```python
discovered, failures = discover_batch_workspaces(
    root,
    resolve_operator_file=lambda workspace: resolve_batch_convert_operator_file(
        workspace,
        operator_filter=operator_filter,
    ),
    no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
)
```

And update the resolver signature:

```python
def resolve_batch_convert_operator_file(
    workspace: Path,
    *,
    operator_filter: str | None = None,
) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_convert_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
        operator_filter=operator_filter,
    )
```

- [ ] **Step 4: Run the focused convert tests and a regression**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliMainTests.test_main_convert_batch_forwards_operator_filter \
  tests.test_convert_commands.ConvertRuntimeTests.test_run_convert_batch_operator_filter_does_not_reinclude_triton_prefixed_files \
  tests.test_convert_commands.ConvertRuntimeTests.test_run_convert_batch_accepts_root_as_single_workspace -v
```

Expected: PASS.

- [ ] **Step 5: Commit the convert batch wiring**

```bash
git add src/triton_agent/commands/convert.py src/triton_agent/convert/batch.py tests/test_cli.py tests/test_convert_commands.py
git commit -m "feat: wire operator filter through convert-batch"
```

### Task 5: Forward `--operator-filter` through the optimize batch command

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/batch.py`
- Modify: `src/triton_agent/optimize/naming.py`

- [ ] **Step 1: Write the failing optimize tests**

Add a CLI forwarding test in `tests/test_cli.py`:

```python
def test_main_optimize_batch_forwards_operator_filter(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        captured: dict[str, object] = {}

        def _fake_run(root_path, options, max_concurrency, operator_filter=None):
            del options
            captured["root"] = root_path
            captured["max_concurrency"] = max_concurrency
            captured["operator_filter"] = operator_filter
            return 0

        with patch(
            "triton_agent.commands.optimize.run_optimize_batch",
            side_effect=_fake_run,
        ):
            exit_code = main(
                [
                    "optimize-batch",
                    "-i",
                    str(root),
                    "--operator-filter",
                    "kernel*.py",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["root"], root.resolve())
        self.assertEqual(captured["max_concurrency"], 1)
        self.assertEqual(captured["operator_filter"], "kernel*.py")
```

Add an optimize runtime test in `tests/test_optimize_runtime.py`:

```python
def test_run_optimize_batch_operator_filter_selects_matching_candidate(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "kernel_workspace"
        workspace.mkdir()
        (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
        (workspace / "kernel_fp16.py").write_text("print('y')\n", encoding="utf-8")

        options = OptimizeRunOptions(
            agent_name="codex",
            interact=False,
            verbose=False,
            stream_output=False,
            remote=None,
            remote_workdir=None,
            min_rounds=1,
            resume_mode="auto",
            reset_optimize=False,
            no_agent_session=False,
            round_mode="checked",
            output=None,
            test_mode=None,
            bench_mode=None,
            prompt=None,
        )
        captured_requests: list[AgentRequest] = []

        def fake_run_request(request, stdout=None, stderr=None) -> AgentResult:
            del stdout, stderr
            captured_requests.append(request)
            return AgentResult(return_code=0, stdout="ok", stderr="")

        with patch(
            "triton_agent.optimize.batch.render_batch_optimize_results",
            return_value=0,
        ):
            exit_code = run_optimize_batch(
                root,
                options,
                max_concurrency=1,
                operator_filter="*_fp16.py",
                stdout=StringIO(),
                run_request=fake_run_request,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0].input_path.name, "kernel_fp16.py")
```

- [ ] **Step 2: Run the focused optimize tests and confirm they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliMainTests.test_main_optimize_batch_forwards_operator_filter \
  tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_operator_filter_selects_matching_candidate -v
```

Expected: FAIL because the optimize batch path does not accept or forward the option yet.

- [ ] **Step 3: Implement optimize batch forwarding**

Update `src/triton_agent/commands/optimize.py`:

```python
return run_optimize_batch(
    root,
    options,
    max_concurrency=max_concurrency,
    operator_filter=getattr(args, "operator_filter", None),
)
```

Update `src/triton_agent/optimize/naming.py`:

```python
def resolve_batch_optimize_operator_file(
    workspace: Path,
    *,
    operator_filter: str | None = None,
) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_optimize_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
        operator_filter=operator_filter,
    )
```

Update `src/triton_agent/optimize/batch.py`:

```python
def run_optimize_batch(
    root: Path,
    options: OptimizeRunOptions,
    *,
    max_concurrency: int,
    operator_filter: str | None = None,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
```

```python
discovered, failures = discover_batch_workspaces(
    root,
    resolve_operator_file=lambda workspace: resolve_batch_optimize_operator_file(
        workspace,
        operator_filter=operator_filter,
    ),
    no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
)
```

- [ ] **Step 4: Run the focused optimize tests and one existing regression**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliMainTests.test_main_optimize_batch_forwards_operator_filter \
  tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_operator_filter_selects_matching_candidate \
  tests.test_optimize_runtime.OptimizeRuntimeTests.test_run_optimize_batch_applies_user_prompt_to_each_workspace_request -v
```

Expected: PASS.

- [ ] **Step 5: Commit the optimize batch wiring**

```bash
git add src/triton_agent/commands/optimize.py src/triton_agent/optimize/batch.py src/triton_agent/optimize/naming.py tests/test_cli.py tests/test_optimize_runtime.py
git commit -m "feat: wire operator filter through optimize-batch"
```

### Task 6: Run the focused regression suite and full verification

**Files:**
- No source changes required unless verification finds regressions

- [ ] **Step 1: Run the focused batch regression suite**

Run:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_generation_batch \
  tests.test_convert_commands \
  tests.test_optimize_runtime -v
```

Expected: PASS.

- [ ] **Step 2: If focused tests fail, fix only operator-filter regressions and rerun**

Possible code touchpoints if needed:

```text
src/triton_agent/cli.py
src/triton_agent/batch_utils.py
src/triton_agent/generation/batch.py
src/triton_agent/convert/batch.py
src/triton_agent/optimize/batch.py
src/triton_agent/optimize/naming.py
src/triton_agent/commands/generation.py
src/triton_agent/commands/convert.py
src/triton_agent/commands/optimize.py
tests/test_cli.py
tests/test_generation_batch.py
tests/test_convert_commands.py
tests/test_optimize_runtime.py
```

Rerun:

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_generation_batch \
  tests.test_convert_commands \
  tests.test_optimize_runtime -v
```

Expected: PASS.

- [ ] **Step 3: Run repository verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS.

- [ ] **Step 4: Commit the finished implementation**

```bash
git add \
  src/triton_agent/cli.py \
  src/triton_agent/batch_utils.py \
  src/triton_agent/generation/batch.py \
  src/triton_agent/convert/batch.py \
  src/triton_agent/optimize/batch.py \
  src/triton_agent/optimize/naming.py \
  src/triton_agent/commands/generation.py \
  src/triton_agent/commands/convert.py \
  src/triton_agent/commands/optimize.py \
  tests/test_cli.py \
  tests/test_generation_batch.py \
  tests/test_convert_commands.py \
  tests/test_optimize_runtime.py \
  docs/plans/2026-06-17-batch-operator-filter.md
git commit -m "feat: add batch operator filter"
```
