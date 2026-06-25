# Optimize Analysis-Driven Implementation Plan

> **Superseded:** The `--require-analysis` flag described in this plan was subsequently removed by `docs/specs/2026-04-22-optimize-layered-analysis-default-design.md` (layered analysis became the default, and the flag was deleted). This plan is retained as implementation history only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten `optimize` into a diagnosis-first workflow, add a lightweight `--require-analysis` flag that only strengthens prompts and guidance, and document that existing test and benchmark harnesses should be reused when present.

**Architecture:** Keep a single optimize skill workflow. Update the optimize skill docs and references so rounds must explain their hypothesis and evidence, then thread a boolean `require_analysis` flag through CLI options, request building, prompt generation, temporary workspace guidance, and resume prompts. Do not add runtime artifact enforcement or workflow branching in the supervisor.

**Tech Stack:** Python `argparse`, `dataclasses`, existing optimize prompt/guidance plumbing, Markdown skill docs, Python `unittest`

---

### Task 1: Add Failing Tests For The New CLI And Prompt Surface

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/prompts.py`

- [ ] **Step 1: Write failing parser tests for `--require-analysis`**

Add tests that lock the new CLI option on both optimize commands:

```python
def test_optimize_command_accepts_require_analysis(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize", "-i", "kernel.py", "--require-analysis"])
    self.assertTrue(args.require_analysis)

def test_optimize_batch_accepts_require_analysis(self) -> None:
    parser = build_parser()
    args = parser.parse_args(["optimize-batch", "-i", "kernels", "--require-analysis"])
    self.assertTrue(args.require_analysis)
```

- [ ] **Step 2: Write failing prompt tests for default and strict optimize wording**

Add prompt tests that require:

- default optimize prompts to mention harness reuse, justified hypotheses, and evidence
- strict optimize prompts to mention analysis before the first code-changing round

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_cli.PromptTests -v`
Expected: FAIL because the parser, option plumbing, and prompt text do not yet expose the new behavior

- [ ] **Step 4: Implement the minimal CLI/model/prompt plumbing**

Add `require_analysis: bool` through optimize CLI options and request models, then teach `build_prompt()` to emit the new optimize wording.

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_cli.PromptTests -v`
Expected: PASS

### Task 2: Add Failing Tests For Temporary Optimize Guidance And Resume Wording

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/optimize/guidance.py`
- Modify: `src/triton_agent/optimize/run_loop.py`
- Modify: `src/triton_agent/backends/codex.py`
- Modify: `src/triton_agent/backends/opencode.py`
- Modify: `src/triton_agent/backends/claude.py`
- Modify: `src/triton_agent/backends/pi.py`

- [ ] **Step 1: Write failing guidance tests**

Add tests that require temporary optimize guidance to mention:

- checking for existing tests and benchmark harnesses before generating new ones
- recording a hypothesis and rationale before code edits
- explaining why profiling or IR capture is skipped
- stricter analysis wording when `require_analysis=True`

- [ ] **Step 2: Write failing resume wording tests**

Add tests that require resumed optimize prompts to preserve the new analysis-driven language, especially in strict mode.

- [ ] **Step 3: Run the targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests tests.test_cli.SupervisorTests -v`
Expected: FAIL because optimize guidance and resume text still only mention generic continuation behavior

- [ ] **Step 4: Implement the minimal guidance and resume changes**

Thread `require_analysis` into temporary optimize guidance generation and resume prompt construction so resumed sessions keep the same expectations.

- [ ] **Step 5: Run the targeted tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests tests.test_cli.SupervisorTests -v`
Expected: PASS

### Task 3: Update Optimize Skill And Reference Docs

**Files:**
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/workflow.md`
- Modify: `skills/triton/triton-npu-optimize/references/pattern_index.md`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Modify: `skills/triton/triton-npu-optimize/references/opt-note-format.md`

- [ ] **Step 1: Update the optimize skill workflow**

Document that optimize should:

- reuse existing harnesses when present
- generate missing harnesses only when needed
- write a diagnosis summary before the first code-changing round
- justify every round with hypothesis plus evidence

- [ ] **Step 2: Update workflow and artifact references**

Make `attempts.md`, `summary.md`, and `opt-note.md` expectations match the new analysis-driven behavior.

- [ ] **Step 3: Tighten the pattern index language**

Adjust pattern selection guidance so tiling, autotune, and launch-parameter exploration are evidence-driven rather than default first moves.

### Task 4: Update User-Facing Documentation And Verify Everything

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/specs/2026-04-09-optimize-analysis-driven-design.md`
- Modify: `docs/plans/2026-04-09-optimize-analysis-driven.md`

- [ ] **Step 1: Update README and AGENTS semantics**

Document:

- the new analysis-driven optimize expectations
- harness reuse before regeneration
- the optional `--require-analysis` flag

- [ ] **Step 2: Run lint**

Run: `uv run --group dev ruff check`
Expected: PASS

- [ ] **Step 3: Run type checking**

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Run the full unit suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
