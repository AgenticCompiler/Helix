# Gen Convert Differential-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redefine `gen-convert` so it only converts one PyTorch operator into a Triton NPU-backed operator and proves correctness through differential testing against the original operator.

**Architecture:** Keep `gen-convert` on the existing generation command path, but shrink its contract. The CLI, prompt, staged-skill allowlist, and docs should all describe a differential-only workflow. Baseline and benchmark behavior stays exclusive to `optimize`.

**Tech Stack:** Python 3, `argparse`, `unittest`, Markdown skills, existing generation orchestration and prompt builders

---

## File Map

- Modify: `README.md`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/generation/orchestration.py`
- Modify: `src/triton_agent/prompts.py`
- Modify: `skills/triton-npu-convert-pytorch-operator/SKILL.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_generation_contracts.py`

## Task 1: Lock The New CLI Contract With Failing Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_commands.py`

- [ ] Add a parser test showing `gen-convert` no longer exposes `bench_mode`.

```python
def test_gen_convert_maps_to_command_kind(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["gen-convert", "-i", "kernel.py"])
    self.assertEqual(args.command_kind, CommandKind.GEN_CONVERT)
    self.assertEqual(args.test_mode, "differential")
    self.assertFalse(hasattr(args, "bench_mode"))
```

- [ ] Add a parser error test showing `gen-convert --test-mode standalone` is rejected.

```python
def test_gen_convert_rejects_non_differential_test_mode(self) -> None:
    parser = build_parser()
    stderr = StringIO()
    with self.assertRaises(SystemExit) as exc, redirect_stderr(stderr):
        parser.parse_args(["gen-convert", "-i", "kernel.py", "--test-mode", "standalone"])
    self.assertEqual(exc.exception.code, 2)
    self.assertIn("differential", stderr.getvalue())
```

- [ ] Add a generation-orchestration test showing `gen-convert` stages convert-plus-test skills only.

```python
def test_build_generation_request_for_gen_convert_uses_differential_only_skills(self) -> None:
    request = build_generation_request(
        CommandKind.GEN_CONVERT,
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

- [ ] Run the focused tests and confirm they fail for the expected contract reasons.

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_gen_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_gen_convert_rejects_non_differential_test_mode \
  tests.test_generation_commands.GenerationHelpersTests.test_build_generation_request_for_gen_convert_uses_differential_only_skills
```

Expected:
- `bench_mode` assertions fail because the parser still exposes benchmark options
- the `standalone` rejection test fails because the parser still accepts it
- the staged-skill assertion fails because baseline and benchmark skills are still present

## Task 2: Shrink The Runtime Surface To Differential-Only

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/generation/orchestration.py`
- Modify: `src/triton_agent/prompts.py`

- [ ] Update the `gen-convert` CLI spec so it does not register `--bench-mode`.

```python
CommandSpec(
    command="gen-convert",
    kind=CommandKind.GEN_CONVERT,
    handler=handle_gen_convert,
    test_mode_default="differential",
    bench_mode_default=None,
)
```

- [ ] Enforce differential-only `gen-convert` validation in parser or command validation logic.

```python
if args.command_kind == CommandKind.GEN_CONVERT and args.test_mode != "differential":
    parser.error("gen-convert supports only --test-mode differential")
```

- [ ] Replace the `GEN_CONVERT` staged-skill allowlist with convert-plus-test skills only.

```python
GEN_CONVERT_STAGED_SKILLS = (
    "triton-npu-convert-pytorch-operator",
    "triton-npu-gen-test",
    "triton-npu-run-eval",
    "triton-npu-repair-guide",
)
```

- [ ] Rewrite the `GEN_CONVERT` prompt block so it requires differential validation against the original operator and forbids baseline or benchmark work.

```python
if command_kind == CommandKind.GEN_CONVERT:
    lines.extend(
        [
            "Treat the input operator file as source material and the differential correctness oracle.",
            "Do not benchmark this workflow.",
            "Do not create `baseline/`.",
            "Generate a differential test for the converted output and execute it.",
            "Validate the converted output by comparing it against the original operator behavior.",
        ]
    )
```

- [ ] Run the focused tests again and confirm the new runtime contract passes.

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_gen_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_gen_convert_rejects_non_differential_test_mode \
  tests.test_generation_commands.GenerationHelpersTests.test_build_generation_request_for_gen_convert_uses_differential_only_skills
```

Expected:
- all three tests pass

## Task 3: Rewrite The Skill And README Contract

**Files:**
- Modify: `skills/triton-npu-convert-pytorch-operator/SKILL.md`
- Modify: `README.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] Add a doc-contract test that requires differential-only convert wording and forbids baseline or benchmark wording.

```python
def test_convert_skill_and_readme_document_differential_only_conversion(self) -> None:
    convert_skill = _read("skills/triton-npu-convert-pytorch-operator/SKILL.md")
    readme = _read("README.md")
    self.assertIn("correctness oracle", convert_skill)
    self.assertIn("differential test", convert_skill)
    self.assertNotIn("triton-npu-prepare-optimize-baseline", convert_skill)
    self.assertNotIn("baseline/", convert_skill)
    self.assertIn("differential correctness validation", readme)
    self.assertNotIn("preparing `baseline/`", readme)
```

- [ ] Rewrite the convert skill so completion means converted artifact plus differential correctness pass.

```md
## Outputs

- one converted operator file
- one generated differential test file
- a short summary of conversion and validation results

## Required Workflow

11. Use the original input operator as the differential correctness oracle.
12. Generate and run a differential test for the converted output.
13. Finish only after the differential test passes or a clear blocker is reported.
```

- [ ] Rewrite the README `gen-convert` section to remove baseline and benchmark claims.

```md
What it is for:

- converting one source PyTorch operator into a Triton NPU-backed PyTorch operator
- preserving the input file's trailing input-helper block in the converted output
- validating the converted operator through differential testing against the original operator
```

- [ ] Run the doc-contract tests and confirm they fail before the doc edits, then pass after the edits.

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_convert_skill_and_readme_document_differential_only_conversion
```

Expected before doc edits:
- failure because the skill and README still mention baseline preparation

Expected after doc edits:
- PASS

## Task 4: Run The Touched Test Slice And Fix Regressions

**Files:**
- Modify any touched files above as needed

- [ ] Run the full touched test slice for CLI, generation helpers, and doc contracts.

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_generation_commands \
  tests.test_generation_contracts
```

Expected:
- all tests pass

- [ ] If a failure appears outside the new contract, make the smallest follow-up edit needed and rerun the same test command.

```bash
uv run python -m unittest \
  tests.test_cli \
  tests.test_generation_commands \
  tests.test_generation_contracts
```

Expected:
- green test run with no convert baseline or benchmark assumptions left

- [ ] Run one final focused smoke command for the changed convert contract.

```bash
uv run python -m unittest \
  tests.test_cli.CliParserTests.test_gen_convert_maps_to_command_kind \
  tests.test_cli.CliParserTests.test_gen_convert_rejects_non_differential_test_mode \
  tests.test_generation_commands.GenerationHelpersTests.test_build_generation_request_for_gen_convert_uses_differential_only_skills \
  tests.test_generation_contracts.GenerationContractTests.test_convert_skill_and_readme_document_differential_only_conversion
```

Expected:
- PASS
