# Single-Attempt Optimize Round Design

## Summary

- Tighten optimize guidance so one round owns one code-changing optimization attempt.
- After the first canonical `run-bench` plus `compare-perf` conclusion for that attempt, stop editing the current round.
- Carry any follow-up optimization idea into a new round instead of looping inside the already-benchmarked round.

## Problem

The current optimize contract leaves room for same-round iteration after a candidate has already reached canonical benchmark evaluation.

Two places encourage that behavior:

- round guidance talks about a round as a hypothesis-driven unit, but it does not explicitly say the round must stop after its first canonical benchmark conclusion
- `skills/triton/triton-npu-optimize/references/round-failure-handling.md` explicitly says a slower-but-promising round may keep iterating within the same round

That produces confusing round histories: one `opt-round-N/` can contain multiple optimization edits, multiple benchmark conclusions, and an unclear round boundary.

## Goals

- Define one optimize round as one code-changing optimization attempt plus its canonical validation.
- Make the worker prompt, temporary workspace guidance, and optimize skill agree on that rule.
- Preserve small validation reruns or metadata repairs that are needed to finish and submit the round.

## Non-Goals

- Do not redesign round-state JSON fields or optimize workflow phases.
- Do not change `submit-round` result kinds or min-round counting.
- Do not turn technical artifact repair into a new orchestration mode.

## Design

### Prompt And Workspace Guidance

Update optimize worker guidance so it states:

- each round gets exactly one code-changing optimization attempt
- after the first canonical `run-bench` plus `compare-perf` conclusion, the round is over
- if the result is slower, inconclusive, or not worth promoting, record that outcome and move the next idea into a new round
- later follow-up prompts may repair round artifacts, but they must not use an already-benchmarked round as permission for a second optimization attempt

This belongs in:

- `src/triton_agent/optimize/prompts.py`
- `src/triton_agent/optimize/memory_file.py`
- `src/triton_agent/optimize/execution.py` follow-up wording

### Skill Contract

Update optimize skill wording so the round boundary is explicit:

- the core loop should say a round makes one coherent optimization attempt
- regression handling should close the round instead of iterating again inside it
- start-round hard rules should remind the agent that a finished round is not reopened for another optimization edit

This belongs in:

- `skills/triton/triton-npu-optimize/SKILL.md`
- `skills/triton/triton-npu-optimize/references/round-failure-handling.md`
- `skills/common/ascend-npu-optimize-state/scripts/state_manage/start_round.py`

## Verification

- prompt tests prove the worker prompt mentions the one-attempt rule
- guidance tests prove temporary `AGENTS.md` mentions the same rule
- generation-contract tests prove the optimize skill and failure-handling reference no longer allow same-round repeated optimization
