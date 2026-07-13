# Convert Batch And Package Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `gen-convert` to `convert`, add `convert-batch`, and move convert runtime ownership into a dedicated `convert/` package.

**Architecture:** Treat this as one coherent rename-plus-boundary correction. The CLI surface moves to `convert` and `convert-batch`, convert-specific orchestration and batch runtime move into `src/helix/convert/`, and `generation/` stays focused on test, benchmark, and eval flows. Batch convert should reuse the existing batch workspace-discovery primitives and mirror the existing prefixed streaming summary style.

**Tech Stack:** Python 3, `argparse`, `unittest`, existing batch workspace helpers, Markdown docs and skills

---

## File Map

- Create: `src/helix/commands/convert.py`
- Create: `src/helix/convert/__init__.py`
- Create: `src/helix/convert/batch.py`
- Create: `src/helix/convert/orchestration.py`
- Create: `src/helix/convert/outputs.py`
- Create: `tests/test_convert_commands.py`
- Modify: `README.md`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `src/helix/generation/outputs.py`
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/models.py`
- Modify: `src/helix/paths.py`
- Modify: `src/helix/prompts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_generation_contracts.py`

## Task 1: Lock The New CLI Surface With Failing Tests

**Files:**
- Modify: `tests/test_cli.py`
- Create: `tests/test_convert_commands.py`

- [ ] **Step 1: Add parser tests that require `convert` and `convert-batch`, and reject `gen-convert`**

```python
def test_convert_maps_to_command_kind(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["convert", "-i", "kernel.py"])
    self.assertEqual(args.command, "convert")
    self.assertEqual(args.command_kind, CommandKind.CONVERT)
    self.assertEqual(args.test_mode, "differential")
    self.assertFalse(hasattr(args, "bench_mode"))


def test_convert_batch_maps_to_command_kind(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["convert-batch", "-i", "kernels"])
    self.assertEqual(args.command, "convert-batch")
    self.assertEqual(args.command_kind, CommandKind.CONVERT_BATCH)
    self.assertEqual(args.max_concurrency, 2)
    self.assertEqual(args.test_mode, "differential")
    self.assertFalse(hasattr(args, "interact"))
    self.assertFalse(hasattr(args, "output"))


def test_gen_convert_is_no_longer_a_valid_command(self) -> None:
    parser = build_parser()
    with self.assertRaises(SystemExit) as exc:
        parser.parse_args(["gen-convert", "-i", "kernel.py"])
    self.assertEqual(exc.exception.code, 2)
```

- [ ] **Step 2: Add help-text assertions that canonical command listings use `convert` and `convert-batch`**

```python
def test_help_keeps_only_canonical_convert_commands(self) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    self.assertIn("convert", help_text)
    self.assertIn("convert-batch", help_text)
    self.assertNotIn("gen-convert", help_text)
    self.assertNotIn("gen_convert", help_text)
```

- [ ] **Step 3: Add convert-command tests that expect a dedicated `helix.commands.convert` module**

```python
def test_convert_command_module_exists(self) -> None:
    self.assertIsNotNone(importlib.util.find_spec("helix.commands.convert"))


def test_generation_command_module_no_longer_exports_convert_handlers(self) -> None:
    import helix.commands.generation as generation_commands

    self.assertFalse(hasattr(generation_commands, "handle_gen_convert"))
```

- [ ] **Step 4: Run the new parser and module tests to verify RED**

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_convert_batch_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_gen_convert_is_no_longer_a_valid_command \
  tests.test_cli.CliParserTests.test_help_keeps_only_canonical_convert_commands \
  tests.test_convert_commands.ConvertCommandModuleTests.test_convert_command_module_exists \
  tests.test_convert_commands.ConvertCommandModuleTests.test_generation_command_module_no_longer_exports_convert_handlers
```

Expected:
- parser tests fail because `convert` and `convert-batch` do not exist yet
- `gen-convert` still parses
- `helix.commands.convert` does not exist yet

## Task 2: Rename The Command Surface And Routing

**Files:**
- Create: `src/helix/commands/convert.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `src/helix/models.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Replace convert command kinds in `src/helix/models.py`**

```python
class CommandKind(str, Enum):
    GEN_EVAL = "gen-eval"
    GEN_EVAL_BATCH = "gen-eval-batch"
    CONVERT = "convert"
    CONVERT_BATCH = "convert-batch"
    GEN_TEST = "gen-test"
```

```python
COMMAND_TO_SKILL = {
    CommandKind.GEN_EVAL: "triton-npu-gen-eval-suite",
    CommandKind.GEN_EVAL_BATCH: "",
    CommandKind.CONVERT: "triton-npu-convert-pytorch-operator",
    CommandKind.CONVERT_BATCH: "",
```

- [ ] **Step 2: Create `src/helix/commands/convert.py` with dedicated single and batch handlers**

```python
from helix.convert.batch import run_convert_batch
from helix.convert.orchestration import build_convert_request, run_convert_request
from helix.generation.models import GenerationOptions


def handle_convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_convert_command(parser, args)


def handle_convert_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    return run_convert_batch(root, convert_options_from_args(args), max_concurrency=args.max_concurrency)
```

- [ ] **Step 3: Update `src/helix/cli.py` to register `convert` and `convert-batch` and drop `gen-convert`**

```python
from helix.commands.convert import handle_convert, handle_convert_batch
from helix.commands.generation import handle_gen_bench, handle_gen_eval, handle_gen_eval_batch, handle_gen_test
```

```python
CommandKind.CONVERT: _CommandSpec(
    handler=handle_convert,
    help_group="Conversion",
    help_summary="Convert one PyTorch operator into a Triton NPU-backed PyTorch operator.",
    description="Convert one PyTorch operator file into a Triton NPU-backed PyTorch operator.",
    has_remote=True,
    has_agent=True,
    has_interact=True,
    has_show_output=True,
    has_test_mode=True,
    test_mode_default="differential",
    test_mode_choices=("differential",),
    has_force_overwrite=True,
),
CommandKind.CONVERT_BATCH: _CommandSpec(
    handler=handle_convert_batch,
    help_group="Conversion",
    help_summary="Convert multiple operator workspaces.",
    description="Convert multiple operator workspaces through the convert workflow.",
    has_output=False,
    has_remote=True,
    has_agent=True,
    has_show_output=True,
    has_test_mode=True,
    test_mode_default="differential",
    test_mode_choices=("differential",),
    max_concurrency_default=2,
),
```

- [ ] **Step 4: Remove convert routing from `src/helix/commands/generation.py`**

```python
def handle_gen_eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_generation_command(parser, args, CommandKind.GEN_EVAL)


def handle_gen_test(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_generation_command(parser, args, CommandKind.GEN_TEST)
```

- [ ] **Step 5: Run the Task 1 tests again to verify GREEN**

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_convert_batch_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_gen_convert_is_no_longer_a_valid_command \
  tests.test_cli.CliParserTests.test_help_keeps_only_canonical_convert_commands \
  tests.test_convert_commands.ConvertCommandModuleTests.test_convert_command_module_exists \
  tests.test_convert_commands.ConvertCommandModuleTests.test_generation_command_module_no_longer_exports_convert_handlers
```

Expected:
- all six tests pass

- [ ] **Step 6: Commit the command-surface rename**

```bash
git add \
  src/helix/models.py \
  src/helix/cli.py \
  src/helix/commands/convert.py \
  src/helix/commands/generation.py \
  tests/test_cli.py \
  tests/test_convert_commands.py
git commit -m "feat: rename convert commands"
```

## Task 3: Move Single-Workspace Convert Runtime Into `convert/`

**Files:**
- Create: `src/helix/convert/__init__.py`
- Create: `src/helix/convert/orchestration.py`
- Create: `src/helix/convert/outputs.py`
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/generation/outputs.py`
- Modify: `src/helix/paths.py`
- Modify: `src/helix/prompts.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_convert_commands.py`

- [ ] **Step 1: Add failing tests that require convert-specific output and orchestration modules**

```python
def test_convert_orchestration_module_exists(self) -> None:
    self.assertIsNotNone(importlib.util.find_spec("helix.convert.orchestration"))


def test_convert_outputs_module_exists(self) -> None:
    self.assertIsNotNone(importlib.util.find_spec("helix.convert.outputs"))
```

```python
def test_build_convert_request_uses_convert_only_skills(self) -> None:
    request = build_convert_request(
        Path("/tmp/kernel.py"),
        Path("/tmp/kernel.py"),
        Path("/tmp"),
        GenerationOptions(
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            remote=None,
            remote_workdir=None,
            min_rounds=None,
            continue_optimize=False,
            output=None,
            test_mode="differential",
            bench_mode=None,
        ),
    )
    self.assertEqual(request.command_kind, CommandKind.CONVERT)
    self.assertEqual(
        request.staged_skill_names,
        (
            "triton-npu-convert-pytorch-operator",
            "triton-npu-gen-test",
            "triton-npu-run-eval",
            "triton-npu-repair-guide",
        ),
    )
```

- [ ] **Step 2: Create `src/helix/convert/outputs.py` and move convert output resolution there**

```python
def resolve_convert_output_path(input_path: Path, *, explicit_output: str | None) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    return input_path.with_name(f"triton_{input_path.stem}.py")


def prepare_convert_target(output_path: Path, *, force_overwrite: bool) -> list[str]:
    if not output_path.exists():
        return []
    if output_path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {output_path}. Choose a file path instead.")
    if not force_overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --force-overwrite to replace it.")
    output_path.unlink()
    return [f"removed existing output file {output_path}"]
```

- [ ] **Step 3: Create `src/helix/convert/orchestration.py` and move convert request building there**

```python
CONVERT_STAGED_SKILLS = (
    "triton-npu-convert-pytorch-operator",
    "triton-npu-gen-test",
    "triton-npu-run-eval",
    "triton-npu-repair-guide",
)


def build_convert_request(
    input_path: Path,
    operator_path: Path,
    workdir: Path,
    options: GenerationOptions,
) -> AgentRequest:
    output_path = resolve_convert_output_path(input_path, explicit_output=options.output)
    prompt = build_prompt(
        CommandKind.CONVERT,
        input_path,
        operator_path,
        output_path,
        options.test_mode,
        options.bench_mode,
        options.force_overwrite,
        options.remote,
        options.remote_workdir,
    )
    return AgentRequest(
        command_kind=CommandKind.CONVERT,
        input_path=input_path,
        operator_path=operator_path,
        output_path=output_path,
        test_mode=options.test_mode,
        bench_mode=options.bench_mode,
        interact=options.interact,
        verbose=options.verbose,
        show_output=options.show_output,
        force_overwrite=options.force_overwrite,
        agent_name=options.agent_name,
        skill_name=COMMAND_TO_SKILL[CommandKind.CONVERT],
        prompt=prompt,
        workdir=workdir,
        staged_skill_names=CONVERT_STAGED_SKILLS,
    )
```

- [ ] **Step 4: Remove convert branches from generation-owned helpers**

```python
if command_kind in {
    CommandKind.GEN_TEST,
    CommandKind.GEN_BENCH,
    CommandKind.OPTIMIZE,
}:
    return default_generated_output_path(command_kind, input_path, test_mode=test_mode)
```

```python
if command_kind not in {
    CommandKind.GEN_TEST,
    CommandKind.GEN_BENCH,
}:
    return []
```

- [ ] **Step 5: Update convert prompt handling to use `CommandKind.CONVERT`**

```python
PROMPT_INTROS = {
    CommandKind.GEN_EVAL: "Repair the operator when needed, then generate correctness tests and a benchmark.",
    CommandKind.CONVERT: "Convert the PyTorch operator into a Triton NPU-backed PyTorch operator and validate it with differential correctness testing.",
```

- [ ] **Step 6: Run focused convert-runtime tests**

```bash
uv run python -m unittest \
  tests.test_convert_commands \
  tests.test_generation_commands.GenerationHelpersTests.test_generation_orchestration_module_replaces_runtime_module \
  tests.test_generation_commands.GenerationHelpersTests.test_prepare_generation_targets_rejects_existing_gen_eval_artifacts_without_overwrite
```

Expected:
- convert-specific tests pass
- generation tests remain green after convert branches are removed

- [ ] **Step 7: Commit the package split for single-workspace convert**

```bash
git add \
  src/helix/convert/__init__.py \
  src/helix/convert/orchestration.py \
  src/helix/convert/outputs.py \
  src/helix/paths.py \
  src/helix/prompts.py \
  src/helix/generation/orchestration.py \
  src/helix/generation/outputs.py \
  tests/test_generation_commands.py \
  tests/test_convert_commands.py
git commit -m "refactor: move convert runtime into convert package"
```

## Task 4: Add `convert-batch`

**Files:**
- Create: `src/helix/convert/batch.py`
- Modify: `src/helix/commands/convert.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing tests for batch workspace discovery, candidate filtering, and summary rendering**

```python
def test_convert_batch_excludes_triton_prefixed_candidates(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "triton_kernel.py").write_text("converted\n", encoding="utf-8")
        (workspace / "kernel.py").write_text("source\n", encoding="utf-8")
        resolved = resolve_batch_convert_operator_file(workspace)
        self.assertEqual(resolved, workspace / "kernel.py")
```

```python
def test_convert_batch_renders_summary(self) -> None:
    stream = StringIO()
    exit_code = render_batch_convert_results(
        [
            BatchConvertResult(Path("/tmp/a"), True, "converted a.py"),
            BatchConvertResult(Path("/tmp/b"), False, "boom"),
        ],
        stdout=stream,
    )
    self.assertEqual(exit_code, 1)
    output = stream.getvalue()
    self.assertIn("[OK] a: converted a.py", output)
    self.assertIn("[FAIL] b: boom", output)
    self.assertIn("Summary: 1 succeeded, 1 failed", output)
```

- [ ] **Step 2: Implement `src/helix/convert/batch.py` using existing batch helpers**

```python
_BATCH_CONVERT_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_", "triton_")
_BATCH_CONVERT_EXCLUDED_NAMES = {"__init__.py"}


def is_batch_convert_operator_candidate(path: Path) -> bool:
    return is_batch_operator_candidate(
        path,
        excluded_names=_BATCH_CONVERT_EXCLUDED_NAMES,
        excluded_prefixes=_BATCH_CONVERT_EXCLUDED_PREFIXES,
    )
```

```python
def run_convert_batch(
    root: Path,
    options: GenerationOptions,
    *,
    max_concurrency: int,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    ...
    request = build_convert_request(item.operator_file, item.operator_file, item.workspace, options)
    ...
    results.append(BatchConvertResult(workspace=item.workspace, succeeded=True, message=f"converted {item.operator_file.name}"))
```

- [ ] **Step 3: Wire `handle_convert_batch()` to the new batch runtime**

```python
return run_convert_batch(
    root,
    convert_options_from_args(args),
    max_concurrency=max_concurrency,
)
```

- [ ] **Step 4: Add README coverage for `convert-batch`**

```md
- `convert-batch`: convert many operator workspaces.
```

```md
uv run helix convert-batch --input operators_root
```

- [ ] **Step 5: Run focused batch tests**

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_batch_maps_to_command_kind \
  tests.test_convert_commands.ConvertBatchTests
```

Expected:
- parser and batch tests pass

- [ ] **Step 6: Commit batch convert support**

```bash
git add \
  src/helix/convert/batch.py \
  src/helix/commands/convert.py \
  tests/test_convert_commands.py \
  tests/test_cli.py \
  README.md
git commit -m "feat: add convert-batch command"
```

## Task 5: Update Prompt, README, And Contract Tests For The Rename

**Files:**
- Modify: `README.md`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add doc-contract tests that require `convert` and `convert-batch` wording**

```python
def test_readme_uses_convert_command_names(self) -> None:
    readme = _read("README.md")
    self.assertIn("`convert`", readme)
    self.assertIn("`convert-batch`", readme)
    self.assertNotIn("`gen-convert`", readme)
```

- [ ] **Step 2: Update README command map, quick start, convert section, and batch section**

```md
- `convert`: convert one PyTorch operator into a Triton NPU-backed PyTorch operator and validate it with differential testing.
- `convert-batch`: convert many operator workspaces.
```

```md
uv run helix convert --input a.py
uv run helix convert-batch --input operators_root
```

- [ ] **Step 3: Update top-level CLI examples to remove `gen-convert`**

```python
_TOP_LEVEL_EXAMPLES = (
    "helix gen-test -i kernel.py",
    "helix convert -i kernel.py",
    "helix convert-batch -i kernels",
```

- [ ] **Step 4: Run contract and help-text verification**

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_help_keeps_only_canonical_convert_commands \
  tests.test_generation_contracts
```

Expected:
- all convert naming assertions pass

- [ ] **Step 5: Commit the rename documentation sweep**

```bash
git add README.md tests/test_generation_contracts.py tests/test_cli.py
git commit -m "docs: rename convert commands"
```

## Task 6: Final Verification

**Files:**
- Modify any touched files above as needed

- [ ] **Step 1: Run the full touched test slice**

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_convert_commands \
  tests.test_generation_commands \
  tests.test_generation_contracts
```

Expected:
- all tests pass

- [ ] **Step 2: If any regression appears, make the smallest follow-up fix and rerun the same command**

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_convert_commands \
  tests.test_generation_commands \
  tests.test_generation_contracts
```

Expected:
- green test run with no remaining `gen-convert` references in parser, routing, or docs

- [ ] **Step 3: Run a focused smoke suite for renamed single and batch convert**

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_convert_batch_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_gen_convert_is_no_longer_a_valid_command \
  tests.test_convert_commands
```

Expected:
- PASS

- [ ] **Step 4: Create the final integration commit**

```bash
git add README.md src/helix tests
git commit -m "feat: split convert workflow and add batch support"
```
