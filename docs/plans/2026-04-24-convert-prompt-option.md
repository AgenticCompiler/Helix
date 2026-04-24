# Convert Prompt Option Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--prompt` support to `convert` and `convert-batch` with the same prompt-append behavior already used by `optimize` and `optimize-batch`.

**Architecture:** Extend the shared convert option payload so parser-level `--prompt` values flow through existing request construction. Reuse the shared prompt append helper in convert orchestration so single and batch convert both get identical `Additional user instructions:` semantics without inventing a convert-specific format.

**Tech Stack:** Python 3, `argparse`, `unittest`, existing CLI/request builder modules

---

### Task 1: Cover Convert Prompt Parsing And Prompt Composition

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_convert_commands.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/generation/models.py`
- Modify: `src/triton_agent/commands/convert.py`
- Modify: `src/triton_agent/convert/orchestration.py`

- [ ] **Step 1: Write the failing parser and orchestration tests**

```python
def test_convert_command_accepts_user_prompt(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["convert", "-i", "kernel.py", "--prompt", "Keep the API shape."])
    self.assertEqual(args.prompt, "Keep the API shape.")
    options = convert_options_from_args(args)
    self.assertEqual(options.prompt, "Keep the API shape.")


def test_convert_batch_accepts_user_prompt(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["convert-batch", "-i", "kernels", "--prompt", "Avoid numerics changes."]
    )
    self.assertEqual(args.prompt, "Avoid numerics changes.")
    options = convert_options_from_args(args)
    self.assertEqual(options.prompt, "Avoid numerics changes.")


def test_build_convert_request_appends_user_prompt(self) -> None:
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
            prompt="Keep the exported function name.",
        ),
    )

    self.assertIn("Additional user instructions:", request.prompt)
    self.assertIn("Keep the exported function name.", request.prompt)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_command_accepts_user_prompt \
  tests.test_cli.CliParserTests.test_convert_batch_accepts_user_prompt \
  tests.test_convert_commands.ConvertRuntimeTests.test_build_convert_request_appends_user_prompt
```

Expected: FAIL because `convert` commands do not register `--prompt`, `GenerationOptions` has no `prompt` field, and convert request building does not append additional user instructions.

- [ ] **Step 3: Implement the minimal parser and request wiring**

```python
@dataclass(frozen=True)
class GenerationOptions:
    ...
    bench_mode: str | None
    prompt: str | None
```

```python
if spec.has_agent and command_kind in {CommandKind.CONVERT, CommandKind.CONVERT_BATCH}:
    subparser.add_argument("--prompt")
```

```python
return GenerationOptions(
    ...
    bench_mode=getattr(args, "bench_mode", None),
    prompt=getattr(args, "prompt", None),
)
```

```python
prompt = append_additional_user_instructions(
    build_prompt(...),
    options.prompt,
)
```

- [ ] **Step 4: Run the focused tests to verify they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_command_accepts_user_prompt \
  tests.test_cli.CliParserTests.test_convert_batch_accepts_user_prompt \
  tests.test_convert_commands.ConvertRuntimeTests.test_build_convert_request_appends_user_prompt
```

Expected: PASS for all three tests.

### Task 2: Prove Batch Convert Reuses The Same Prompt For Every Workspace

**Files:**
- Modify: `tests/test_convert_commands.py`
- Modify: `src/triton_agent/convert/batch.py` only if batch execution needs prompt-specific handling beyond request construction

- [ ] **Step 1: Write the failing batch propagation test**

```python
def test_run_convert_batch_applies_user_prompt_to_each_workspace_request(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("kernel_a", "kernel_b"):
            workspace = root / name
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

        options = GenerationOptions(
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
            prompt="Avoid changing numerics.",
        )

        prompts: list[str] = []

        def _fake_run(request, stdout=None, stderr=None):
            del stdout, stderr
            prompts.append(request.prompt)
            return AgentResult(return_code=0, stdout="ok", stderr="")

        exit_code = run_convert_batch(root, options, max_concurrency=1, run_request=_fake_run)

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("Additional user instructions:", prompt)
            self.assertIn("Avoid changing numerics.", prompt)
```

- [ ] **Step 2: Run the focused batch test to verify it fails**

Run:

```bash
uv run python -m unittest \
  tests.test_convert_commands.ConvertBatchTests.test_run_convert_batch_applies_user_prompt_to_each_workspace_request
```

Expected: FAIL because the built convert requests do not yet include appended user instructions.

- [ ] **Step 3: Keep batch flow on shared request construction only**

```python
request = build_convert_request(
    item.operator_file,
    item.operator_file,
    item.workspace,
    options,
)
```

No new batch-specific prompt branch should be added if Task 1 already makes request construction carry `options.prompt`.

- [ ] **Step 4: Run the focused batch test to verify it passes**

Run:

```bash
uv run python -m unittest \
  tests.test_convert_commands.ConvertBatchTests.test_run_convert_batch_applies_user_prompt_to_each_workspace_request
```

Expected: PASS.

- [ ] **Step 5: Run the final regression slice**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_convert_batch_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_convert_command_accepts_user_prompt \
  tests.test_cli.CliParserTests.test_convert_batch_accepts_user_prompt \
  tests.test_convert_commands.ConvertRuntimeTests.test_build_convert_request_uses_convert_only_skills \
  tests.test_convert_commands.ConvertRuntimeTests.test_build_convert_request_appends_user_prompt \
  tests.test_convert_commands.ConvertBatchTests.test_run_convert_batch_accepts_root_as_single_workspace \
  tests.test_convert_commands.ConvertBatchTests.test_run_convert_batch_applies_user_prompt_to_each_workspace_request
```

Expected: PASS with no convert prompt regressions.
