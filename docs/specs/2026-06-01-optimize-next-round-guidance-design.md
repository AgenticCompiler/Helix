# Optimize Next-Round Guidance Design

## Summary

- Strengthen the optimize workflow guidance that appears after a successful `check-round`.
- Tell the agent which optimization round should open next when the session must continue.
- Require a short pre-round reflection before starting the next round: choose the right operator or bottleneck focus first, then decide whether profiling, IR, or compiler-source analysis is needed.

## Problem

The current workflow already tells the agent whether it may stop after a round, but it does not strongly direct how to begin the next round.

This leaves three gaps:

- In `continuous` mode, the most immediate post-`check-round` signal is the success summary from the optimize-check script, and that summary does not name the next round or enforce a pause for analysis selection.
- In `checked` and `supervised` multi-invocation flows, the CLI follow-up summary exposes the latest round and continue requirement, but does not explicitly state the next round name.
- The shared continue prompt tells the agent to keep going, but it does not clearly require a deliberate pre-round reflection on the starting operator and evidence level before more code edits.

## Goals

- When a round passes but the session must continue, explicitly name the next round as `opt-round-N+1`.
- Make the immediate success summary in `continuous` mode tell the agent not to rush into code changes.
- Make the CLI handoff summary in `checked` and `supervised` modes expose both the latest round and the next round.
- Strengthen the continue prompt so the next round begins with bottleneck selection and evidence-level selection, not reflexive code editing.

## Non-Goals

- Do not redesign general resume semantics outside the post-`check-round` continue path.
- Do not add new structured result fields such as `next_round_name`.
- Do not change round gating decisions, minimum-round counting, or best-round selection.

## Design

### 1. Optimize-check success summary

Update the pass summary in `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py` when `min_rounds` is provided and more rounds are still required.

The summary should:

- keep the existing round-progress signal
- explicitly name the next round as `opt-round-N+1`
- tell the agent not to rush into the next code edit
- tell the agent to first choose the right operator or bottleneck entrypoint
- tell the agent to decide whether existing evidence is enough or whether deeper profiling, IR, or compiler-source analysis is needed
- tell the agent not to use agents or subagents to optimize multiple rounds in parallel
- tell the agent not to spend the next round on a parameter-tuning-only sweep

This is the most important direct signal for `continuous` mode because the agent sees it immediately after running `check-round`.

### 2. CLI follow-up summary

Update `src/triton_agent/optimize/execution.py` so the CLI-generated follow-up summary includes:

- `Latest round: opt-round-N`
- `Next round: opt-round-N+1`
- the existing continue requirement and issues

This makes the next-round identity explicit when the CLI launches a fresh worker in `checked` and `supervised` flows.

### 3. Continue prompt guidance

Update `src/triton_agent/optimize/prompts.py` so the shared continue prompt requires the next round to begin with a short reflection step before editing code.

That reflection should require the agent to decide:

- which operator, kernel path, or wrapper bottleneck should anchor the next round
- whether current benchmark and comparison evidence is already sufficient
- whether profiling is needed first
- whether IR is needed after profiler evidence narrows but does not explain the bottleneck
- whether compiler-source analysis is justified only after profiler and IR evidence have narrowed a concrete compiler-side question
- whether the next round is driven by a bottleneck-backed hypothesis rather than a pure tuning sweep

This keeps the optimize loop deliberate instead of reactive.

## Files

- Modify `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`
- Modify `src/triton_agent/optimize/execution.py`
- Modify `src/triton_agent/optimize/prompts.py`
- Modify targeted tests under `tests/`

## Verification

- Add or update tests that assert the `check-round` pass summary names the next round and pre-round reflection guidance.
- Add or update tests that assert CLI follow-up summaries include `Next round`.
- Add or update tests that assert continue prompts mention the pre-round reflection and evidence-ladder choices.
