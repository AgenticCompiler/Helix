---
name: triton-npu-optimize-start-round
description: Start the next optimize round carefully by enforcing one-round-at-a-time workflow constraints.
---

# Optimize Start Round

This skill must be used right before beginning a new `opt-round-N/`.

IMPORTANT guidance: make sure the next round starts from a deliberate bottleneck-backed plan instead of a rushed or parallelized workflow.

## Hard Rules

- Only one optimize round may be active at a time.
- Do not optimize multiple rounds in parallel.
- Do not use agents or subagents to advance a different optimize round while the current round is still in flight.
- Do not treat the next round as a blind parameter sweep.
- Do not burn rounds on hand-tuned launch or tile sweeps unless existing evidence clearly justifies that direction.
- Before editing code, decide which operator, kernel path, or wrapper bottleneck should anchor the next round.
- Before editing code, decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first.
- Keep the round goal narrow: one coherent hypothesis, one active round, one evidence-backed change direction.
