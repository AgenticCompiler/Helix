# Optimize Layered Analysis Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make layered analysis the default `optimize` workflow and remove `--require-analysis` plus all related plumbing.

**Architecture:** Keep the behavior change centered on optimize contracts, prompts, and guidance. First remove the CLI/model flag surface with failing parser and model tests, then make worker, unsupervised, resume, and supervisor prompts describe the layered escalation order by default, and finally update optimize skill docs plus README so the user-facing contract matches the new default.

**Tech Stack:** Python `argparse`, Python dataclasses, Markdown skill docs, Python `unittest`

---

### Task 1: Remove The `--require-analysis` CLI And Model Surface

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/models.py`
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `src/helix/optimize/run_loop.py`
- Modify: `src/helix/backends/base.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Rewrite parser tests so the flag is rejected**

Replace the existing parser-acceptance checks in `tests/test_cli.py` with rejection tests shaped like:

```python
    def test_optimize_command_rejects_require_analysis(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["optimize", "-i", "kernel.py", "--require-analysis"])

    def test_optimize_batch_rejects_require_analysis(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["optimize-batch", "-i", "kernels", "--require-analysis"])
```

Also remove assertions that expect `args.require_analysis` or `captured["require_analysis"]`.

- [ ] **Step 2: Update model tests to stop constructing requests with `require_analysis`**

Rewrite the optimize request/model fixtures in `tests/test_models.py` so they no longer pass or assert `require_analysis`.

Use request construction shaped like:

```python
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-optimize",
            prompt="Prompt",
            workdir=Path("/tmp"),
        )
```

- [ ] **Step 3: Run the CLI and model tests to verify RED**

Run:

```bash
uv run python -m unittest tests.test_cli tests.test_models -v
```

Expected: FAIL because the parser still accepts `--require-analysis`, and the runtime code still expects `require_analysis` fields in optimize option and request plumbing.

- [ ] **Step 4: Remove the flag and field plumbing**

Apply the minimal code changes:

```python
# src/helix/cli.py
        if spec.has_optimize_options:
            subparser.add_argument("--min-rounds", type=int)
            subparser.add_argument("--resume", default="auto", choices=_RESUME_CHOICES)
            subparser.add_argument("--reset-optimize", action="store_true")
            subparser.add_argument("--enable-compiler-source-analysis", action="store_true")
```

```python
# src/helix/models.py
    min_rounds: Optional[int] = None
    continue_optimize: bool = False
    no_agent_session: bool = False
```

```python
# src/helix/optimize/models.py
    resume_mode: str
    reset_optimize: bool
    no_agent_session: bool
```

Then remove the deleted field from:

- `optimize_run_options_from_args()`
- `build_optimize_request()`
- `build_optimize_supervisor_prompt()` call sites
- `OptimizeRunLoop` and execution adapters
- `AgentRunner.resume()`

- [ ] **Step 5: Re-run the CLI and model tests**

Run:

```bash
uv run python -m unittest tests.test_cli tests.test_models -v
```

Expected: PASS.

### Task 2: Make Layered Analysis The Default Prompt And Guidance Contract

**Files:**
- Modify: `src/helix/prompts.py`
- Modify: `src/helix/optimize/guidance.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_backends_base.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_codex_runner.py`
- Modify: `tests/test_opencode_runner.py`
- Modify: `tests/test_pi_runner.py`
- Modify: `tests/test_claude_runner.py`
- Modify: `tests/test_traecli_runner.py`

- [ ] **Step 1: Rewrite prompt and guidance tests around the new default**

Replace the old strict-analysis wording checks with default layered-analysis checks.

Use assertions shaped like:

```python
    def test_optimize_prompt_defaults_to_layered_analysis(self) -> None:
        prompt = build_prompt(
            CommandKind.OPTIMIZE,
            Path("/tmp/op.py"),
            Path("/tmp/op.py"),
            Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            force_overwrite=False,
            supervise="on",
        )
        self.assertIn("Choose the analysis level for the round before editing code.", prompt)
        self.assertIn(
            "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
            prompt,
        )
        self.assertIn("Do not begin with blind tiling or launch-parameter search.", prompt)
```

Update resume and guidance tests to check that the layered wording appears without passing any `require_analysis=True` fixture field.

- [ ] **Step 2: Run the prompt/guidance suite to verify RED**

Run:

```bash
uv run python -m unittest tests.test_cli tests.test_optimize_guidance tests.test_backends_base tests.test_supervisor tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner tests.test_traecli_runner -v
```

Expected: FAIL because optimize prompts and guidance still gate the stronger wording behind the deleted `require_analysis` parameter and do not yet mention the layered escalation order by default.

- [ ] **Step 3: Rewrite optimize prompts around the layered default**

In `src/helix/prompts.py`, add a shared optimize prompt block instead of the old conditional branch. Use wording shaped like:

```python
def layered_analysis_lines() -> list[str]:
    return [
        "Choose the analysis level for the round before editing code.",
        "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
        "Use pattern triage only to decide whether a strong pattern-backed hypothesis already exists.",
        "If pattern triage is not enough, use profiling diagnosis as the default deeper entrypoint.",
        "Use IR attribution only after profiler-backed symptoms need explanation.",
        "Use compiler-source escalation only when profiler and IR evidence have already narrowed the issue.",
        "When starting from a deeper level, cite the reused evidence path and explain why the shallower level is already established or insufficient.",
        "Do not begin with blind tiling or launch-parameter search.",
    ]
```

Thread that block into:

- `build_optimize_worker_prompt()`
- `build_optimize_unsupervised_prompt()`
- `build_optimize_supervisor_prompt()`
- `build_optimize_resume_prompt()`

without any `require_analysis` parameter.

- [ ] **Step 4: Rewrite optimize guidance to match the same default**

In `src/helix/optimize/guidance.py`, remove the conditional `analysis_block` and replace it with a default block such as:

```python
        analysis_block = (
            "- Choose the analysis level for each round before editing code.\n"
            "- Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.\n"
            "- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.\n"
            "- Record why a deeper escalation is needed when a round moves past its current level.\n"
            "- Do not begin with blind tiling or launch-parameter search.\n"
        )
```

Keep compiler-source wording separate and continue to present it as the last escalation in the evidence ladder.

- [ ] **Step 5: Re-run the prompt/guidance suite**

Run:

```bash
uv run python -m unittest tests.test_cli tests.test_optimize_guidance tests.test_backends_base tests.test_supervisor tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner tests.test_traecli_runner -v
```

Expected: PASS.

### Task 3: Update Optimize Skill Docs And README To Match The New Default

**Files:**
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/workflow.md`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Modify: `README.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add a failing doc-contract test for the layered workflow**

Extend `tests/test_generation_contracts.py` with a contract test shaped like:

```python
    def test_optimize_docs_default_to_layered_analysis_and_do_not_reference_require_analysis(self) -> None:
        optimize = _read("skills/triton/triton-npu-optimize/SKILL.md")
        workflow = _read("skills/triton/triton-npu-optimize/references/workflow.md")
        readme = _read("README.md")

        self.assertIn("pattern triage", optimize)
        self.assertIn("profiling diagnosis", optimize)
        self.assertIn("IR attribution", optimize)
        self.assertIn("compiler-source escalation", optimize)
        self.assertIn("Use profiling diagnosis as the default deeper entrypoint", workflow)
        self.assertNotIn("--require-analysis", readme)
        self.assertNotIn("--require-analysis", optimize)
        self.assertNotIn("--require-analysis", workflow)
```

- [ ] **Step 2: Run the optimize-doc contract suite to verify RED**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: FAIL because the docs and README still mention `--require-analysis` and do not yet describe the layered default workflow.

- [ ] **Step 3: Update the optimize skill and workflow docs**

Rewrite the optimize docs so they explicitly describe:

- the default analysis order
- pattern triage as shallow screening rather than blind pattern search
- profiling as the default deeper entrypoint
- IR as explanation and attribution
- compiler source as the final escalation
- the requirement that each round record its chosen level and escalation reason

Use text shaped like:

```md
- Choose the round's analysis level before editing code.
- Escalate in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.
- If a round starts from a deeper level, cite the reused evidence path and explain why the shallower level is already established or insufficient.
```

- [ ] **Step 4: Remove `--require-analysis` from the README**

Delete the option bullets and examples that mention the removed flag, and rewrite the optimize section so layered analysis is described as the default behavior.

- [ ] **Step 5: Re-run the generation-contract suite**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts -v
```

Expected: PASS.

### Task 4: Final Verification And Review

**Files:**
- All files changed above

- [ ] **Step 1: Run the combined verification command**

Run:

```bash
uv run python -m unittest tests.test_cli tests.test_models tests.test_optimize_guidance tests.test_backends_base tests.test_supervisor tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner tests.test_traecli_runner tests.test_generation_contracts -v
```

Expected: PASS.

- [ ] **Step 2: Review the final diff**

Run:

```bash
git diff -- docs/specs/2026-04-22-optimize-layered-analysis-default-design.md docs/plans/2026-04-22-optimize-layered-analysis-default.md README.md src/helix/cli.py src/helix/commands/optimize.py src/helix/models.py src/helix/prompts.py src/helix/optimize/models.py src/helix/optimize/orchestration.py src/helix/optimize/execution.py src/helix/optimize/run_loop.py src/helix/optimize/guidance.py src/helix/backends/base.py skills/triton/triton-npu-optimize/SKILL.md skills/triton/triton-npu-optimize/references/workflow.md skills/triton/triton-npu-optimize/references/artifacts.md tests/test_cli.py tests/test_models.py tests/test_optimize_guidance.py tests/test_backends_base.py tests/test_supervisor.py tests/test_codex_runner.py tests/test_opencode_runner.py tests/test_pi_runner.py tests/test_claude_runner.py tests/test_traecli_runner.py tests/test_generation_contracts.py
```

Expected: only the approved layered-analysis workflow, `--require-analysis` removal, matching docs, and aligned tests changed.
