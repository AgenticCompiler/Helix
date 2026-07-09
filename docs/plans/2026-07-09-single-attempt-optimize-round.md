# Single-Attempt Optimize Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each optimize round stop after one code-changing optimization attempt and one canonical benchmark conclusion.

**Architecture:** Keep this change in the guidance and contract layers instead of redesigning optimize workflow state. Align the worker prompt, temporary workspace guidance, CLI follow-up prompt, optimize skill docs, and `start-round` hard rules so they all describe the same boundary: one round owns one code-changing optimization attempt, then closes after the first canonical `run-bench` plus `compare-perf` conclusion for that attempt.

**Tech Stack:** Python `unittest`, existing optimize prompt builders, temporary optimize guidance rendering, staged skill markdown, one skill-side Python helper checked with the repository's strict skill-script `pyright` wrapper

---

## File Structure

- `src/triton_agent/optimize/prompts.py`
  Owns the checked/supervised worker prompt contract. This is the primary place to tell the agent that one worker-owned round gets one code-changing optimization attempt.
- `src/triton_agent/optimize/memory_file.py`
  Owns the temporary `AGENTS.md` / `CLAUDE.md` guidance injected into optimize workspaces. This must mirror the worker prompt so backend differences do not reopen same-round iteration.
- `src/triton_agent/optimize/execution.py`
  Owns the CLI-generated follow-up prompt when a later invocation needs to repair prior batch issues. This wording must forbid reusing an already-benchmarked round for a second optimization attempt.
- `skills/triton/triton-npu-optimize/SKILL.md`
  Owns the human-readable optimize workflow contract. It must define a round as one attempt plus canonical validation.
- `skills/triton/triton-npu-optimize/references/round-failure-handling.md`
  Owns round-local regression and failure guidance. It currently contains the same-round iteration loophole and must be rewritten to close the round instead.
- `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`
  Owns runtime `hard_rules` returned by `start-round`. It must remind the agent that a round is not a container for multiple post-benchmark optimization edits.
- `tests/test_cli.py`
  Verifies worker prompt contract.
- `tests/test_optimize_guidance.py`
  Verifies temporary round-loop guidance contract.
- `tests/test_optimize_runtime.py`
  Verifies later CLI follow-up prompts do not authorize a second optimization attempt inside an already-benchmarked round.
- `tests/test_generation_contracts.py`
  Verifies the staged optimize skill markdown contract.
- `tests/test_skill_command_script.py`
  Verifies `start-round` runtime `hard_rules`.

### Task 1: Constrain The Worker Prompt

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/optimize/prompts.py`

- [ ] **Step 1: Extend the existing worker prompt test with the one-attempt assertions**

Update `tests/test_cli.py` inside `PromptTests.test_build_optimize_round_prompt_mentions_current_and_final_round` so it also checks for the new round boundary:

```python
def test_build_optimize_round_prompt_mentions_current_and_final_round(self) -> None:
    prompt = build_optimize_round_prompt(
        Path("/tmp/op.py"),
        Path("/tmp/opt_op.py"),
        test_mode="differential",
        bench_mode="torch-npu-profiler",
        round_mode="checked",
        current_round=2,
        final_round=4,
        round_batch_size=3,
    )
    self.assertIn("This invocation owns rounds 2 through 4.", prompt)
    self.assertIn("Execute those rounds strictly one at a time.", prompt)
    self.assertIn(
        "Each round in this invocation gets exactly one code-changing optimization attempt.",
        prompt,
    )
    self.assertIn(
        "After the first canonical `run-bench` plus `compare-perf` conclusion for that attempt, stop editing that round and record the outcome.",
        prompt,
    )
    self.assertIn(
        "If the result is slower, inconclusive, or not worth promoting, carry the next optimization idea into a new round instead of revising the current round again.",
        prompt,
    )
    self.assertIn("Do not pre-plan the full batch before acting.", prompt)
    self.assertIn(
        "When a round in this invocation is complete, run `submit-round --round-dir opt-round-N --current-round N --final-round M` with the actual round numbers from this worker batch.",
        prompt,
    )
```

- [ ] **Step 2: Run the focused prompt test and verify it fails for the new assertions**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.PromptTests.test_build_optimize_round_prompt_mentions_current_and_final_round \
  -v
```

Expected: FAIL because `build_optimize_round_prompt()` does not yet mention the one-attempt round boundary.

- [ ] **Step 3: Add the minimal worker-prompt lines in `build_optimize_round_prompt()`**

Update `src/triton_agent/optimize/prompts.py` so the leading `lines` list in `build_optimize_round_prompt()` becomes:

```python
lines = [
    f"This invocation owns rounds {current_round} through {final_round}.",
    "Execute those rounds strictly one at a time.",
    "Each round in this invocation gets exactly one code-changing optimization attempt.",
    "After the first canonical `run-bench` plus `compare-perf` conclusion for that attempt, stop editing that round and record the outcome.",
    "If the result is slower, inconclusive, or not worth promoting, carry the next optimization idea into a new round instead of revising the current round again.",
    "Do not pre-plan the full batch before acting.",
    "Produce all required round artifacts before stopping.",
    "The CLI will validate the completed batch after the invocation exits.",
]
```

- [ ] **Step 4: Re-run the focused prompt test and verify it passes**

Run:

```bash
uv run python -m unittest \
  tests.test_cli.PromptTests.test_build_optimize_round_prompt_mentions_current_and_final_round \
  -v
```

Expected: PASS.

- [ ] **Step 5: Commit the prompt-only change**

```bash
git add tests/test_cli.py src/triton_agent/optimize/prompts.py
git commit -m "fix: constrain optimize worker prompt to one attempt per round"
```

### Task 2: Mirror The Rule In Workspace Guidance And CLI Follow-Up

**Files:**
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/triton_agent/optimize/memory_file.py`
- Modify: `src/triton_agent/optimize/execution.py`

- [ ] **Step 1: Extend the checked-session guidance test with the same one-attempt assertions**

Update `tests/test_optimize_guidance.py` inside `OptimizeSessionArtifactsManagerTests.test_prepare_checked_session_creates_round_loop_guidance_file_only`:

```python
self.assertIn(
    "Each optimize round is one code-changing optimization attempt plus its canonical validation.",
    guidance_content,
)
self.assertIn(
    "After the first canonical `run-bench` plus `compare-perf` conclusion for a round, stop editing that round and record the outcome.",
    guidance_content,
)
self.assertIn(
    "If the result is slower, inconclusive, or not worth promoting, move the next optimization idea into a new round instead of reusing the current round.",
    guidance_content,
)
```

- [ ] **Step 2: Extend the repair-follow-up runtime test so it forbids a second same-round attempt**

Update `tests/test_optimize_runtime.py` inside `OptimizeRuntimeTests.test_multi_invocation_controller_checked_batch_carries_failures_to_next_batch`:

```python
self.assertIn("CLI batch follow-up from the previous worker batch:", runner.requests[1].prompt)
self.assertIn("opt-round-2", runner.requests[1].prompt)
self.assertIn("opt-round-3", runner.requests[1].prompt)
self.assertIn(
    "Do not use an already-benchmarked round for another code-changing optimization attempt.",
    runner.requests[1].prompt,
)
self.assertIn(
    "Carry any next optimization idea into the new round range owned by this invocation.",
    runner.requests[1].prompt,
)
self.assertNotIn("not yet accepted as session progress", runner.requests[1].prompt)
```

- [ ] **Step 3: Run the guidance and runtime tests and verify they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests.test_prepare_checked_session_creates_round_loop_guidance_file_only \
  tests.test_optimize_runtime.OptimizeRuntimeTests.test_multi_invocation_controller_checked_batch_carries_failures_to_next_batch \
  -v
```

Expected: FAIL because neither the temporary guidance file nor the repair follow-up prompt contains the new wording yet.

- [ ] **Step 4: Add the one-attempt wording to the round-loop guidance template**

Update `_ROUND_GATED_GUIDANCE_TEMPLATE` in `src/triton_agent/optimize/memory_file.py` so the static text block includes:

```python
This workspace is under an optimize round loop.
Each optimize round is one code-changing optimization attempt plus its canonical validation.
After the first canonical `run-bench` plus `compare-perf` conclusion for a round, stop editing that round and record the outcome.
If the result is slower, inconclusive, or not worth promoting, move the next optimization idea into a new round instead of reusing the current round.
```

- [ ] **Step 5: Narrow the CLI repair follow-up wording**

Update `_request_with_fresh_batch_prompt()` in `src/triton_agent/optimize/execution.py` so `repair_lines` becomes:

```python
repair_lines = [
    f"This invocation needs to complete rounds {batch_start} through {batch_end}, "
    "but before that, fix the previous batch issues.",
    "CLI batch follow-up from the previous worker batch:",
    issues,
    "Repair those issues first using the existing round directories and artifacts.",
    "Do not use an already-benchmarked round for another code-changing optimization attempt.",
    "Carry any next optimization idea into the new round range owned by this invocation.",
]
```

- [ ] **Step 6: Re-run the guidance and runtime tests and verify they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_guidance.OptimizeSessionArtifactsManagerTests.test_prepare_checked_session_creates_round_loop_guidance_file_only \
  tests.test_optimize_runtime.OptimizeRuntimeTests.test_multi_invocation_controller_checked_batch_carries_failures_to_next_batch \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit the orchestration-guidance change**

```bash
git add tests/test_optimize_guidance.py tests/test_optimize_runtime.py src/triton_agent/optimize/memory_file.py src/triton_agent/optimize/execution.py
git commit -m "fix: keep optimize follow-up prompts round-bounded"
```

### Task 3: Align The Skill Contract And `start-round` Hard Rules

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_skill_command_script.py`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/round-failure-handling.md`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`

- [ ] **Step 1: Add a dedicated generation-contract test for the one-attempt round boundary**

Add this test to `tests/test_generation_contracts.py`:

```python
def test_optimize_round_contract_limits_rounds_to_one_code_change(self) -> None:
    optimize = _read("skills/triton/triton-npu-optimize/SKILL.md")
    failure_handling = _read("skills/triton/triton-npu-optimize/references/round-failure-handling.md")

    self.assertIn("one code-changing optimization attempt", optimize)
    self.assertIn(
        "After the first canonical `run-bench` plus `compare-perf` conclusion for that attempt, stop editing the current round.",
        optimize,
    )
    self.assertIn("move the next optimization idea into a new round", failure_handling)
    self.assertNotIn("if yes, keep iterating within the same round", failure_handling)
```

- [ ] **Step 2: Extend the successful `start-round` script test with the new `hard_rules` assertions**

Update `tests/test_skill_command_script.py` inside `SkillCommandScriptTests.test_optimize_state_start_round_success_returns_strategy_state`:

```python
self.assertIn(
    "Treat each round as one code-changing optimization attempt followed by canonical validation.",
    payload["hard_rules"],
)
self.assertIn(
    "After the first canonical `run-bench` plus `compare-perf` conclusion for a round, do not keep editing that round.",
    payload["hard_rules"],
)
```

- [ ] **Step 3: Run the contract tests and verify they fail**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_round_contract_limits_rounds_to_one_code_change \
  tests.test_skill_command_script.SkillCommandScriptTests.test_optimize_state_start_round_success_returns_strategy_state \
  -v
```

Expected: FAIL because the optimize skill markdown still allows same-round iteration and `start-round` does not yet emit the new `hard_rules`.

- [ ] **Step 4: Update the optimize skill and regression reference to close the round after the first canonical benchmark conclusion**

Make the following exact wording changes.

In `skills/triton/triton-npu-optimize/SKILL.md`, update the core loop bullets to include:

```markdown
- treat each round as one code-changing optimization attempt followed by canonical validation
- make one coherent optimization attempt
- optionally screen the direction cheaply with `probe-bench` when the available run-eval surface exposes it
- validate correctness and benchmark performance
- after the first canonical `run-bench` plus `compare-perf` conclusion for that attempt, stop editing the current round and record the round outcome
```

Also add this rule near Stage 3:

```markdown
- If the result is slower, inconclusive, or not worth promoting, close the round and carry the next optimization idea into a new round instead of revising the current round again.
```

In `skills/triton/triton-npu-optimize/references/round-failure-handling.md`, replace the regression section with:

```markdown
## Regression Handling

If a round is correct but slower:

- preserve the round summary as a failed or non-promoted branch
- do not reopen the current round for another optimization edit after the first canonical benchmark conclusion
- move any next optimization idea into a new round that starts from the best validated parent for that next idea

Do not promote a slower round to current best status.
```

- [ ] **Step 5: Add the same rule to `start-round` runtime `hard_rules`**

Update `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py` so `HARD_RULES` includes:

```python
HARD_RULES = (
    "Only one optimize round may be active at a time.",
    "Treat each round as one code-changing optimization attempt followed by canonical validation.",
    "After the first canonical `run-bench` plus `compare-perf` conclusion for a round, do not keep editing that round.",
    "Do not use a script to create multiple optimize rounds where each round only adjusts parameters in order to speed up the optimization process. This is cheating behavior and is strictly prohibited.",
    "Do not use agents or subagents to advance multiple rounds in parallel while the current round is still in flight.",
    "Do not treat the next round as a blind parameter sweep. If you need to tune parameters, prefer the `autotune` optimization pattern.",
    "Do not burn rounds on hand-tuned launch or tile sweeps unless existing evidence clearly justifies that direction.",
    "Before editing code, decide which operator, kernel path, or wrapper bottleneck should anchor the next round.",
    "Before editing code, decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first.",
    "Keep the round goal narrow: one coherent hypothesis, one active round, one evidence-backed change direction.",
)
```

- [ ] **Step 6: Run the skill-script strict `pyright` check and the focused contract tests**

Run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_optimize_round_contract_limits_rounds_to_one_code_change \
  tests.test_skill_command_script.SkillCommandScriptTests.test_optimize_state_start_round_success_returns_strategy_state \
  -v
```

Expected: the strict `pyright` check reports success, and both focused tests PASS.

- [ ] **Step 7: Commit the contract-layer change**

```bash
git add tests/test_generation_contracts.py tests/test_skill_command_script.py skills/triton/triton-npu-optimize/SKILL.md skills/triton/triton-npu-optimize/references/round-failure-handling.md skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py
git commit -m "docs: align optimize round contract with single-attempt rule"
```

### Task 4: Run Full Repository Verification

**Files:**
- Verify: `tests/test_cli.py`
- Verify: `tests/test_optimize_guidance.py`
- Verify: `tests/test_optimize_runtime.py`
- Verify: `tests/test_generation_contracts.py`
- Verify: `tests/test_skill_command_script.py`
- Verify: `src/triton_agent/optimize/prompts.py`
- Verify: `src/triton_agent/optimize/memory_file.py`
- Verify: `src/triton_agent/optimize/execution.py`
- Verify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`
- Verify: `skills/triton/triton-npu-optimize/SKILL.md`
- Verify: `skills/triton/triton-npu-optimize/references/round-failure-handling.md`

- [ ] **Step 1: Run repository lint**

Run:

```bash
uv run --group dev ruff check
```

Expected: PASS with no new lint findings.

- [ ] **Step 2: Run repository type checking**

Run:

```bash
uv run pyright
```

Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS.

- [ ] **Step 4: Confirm the worktree is in the expected state before handoff**

Run:

```bash
git status --short
```

Expected: only the intended tracked changes remain if this task is being reviewed before the final commit, or nothing remains if the commits from Tasks 1-3 are already completed and no follow-up fixes were needed.
