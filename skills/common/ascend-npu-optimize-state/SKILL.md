---
name: ascend-npu-optimize-state
description: Use when baseline validation, round-start gating, or completed-round validation is needed in an Ascend NPU optimize workspace.
---

# Optimize State

Use this skill to manage optimize workflow state through one structured CLI entrypoint:

```bash
python3 scripts/cli.py submit-baseline --baseline-dir baseline
python3 scripts/cli.py start-round --round-dir opt-round-1
python3 scripts/cli.py submit-round --round-dir opt-round-1
python3 scripts/cli.py submit-round --round-dir opt-round-2 --current-round 2 --final-round 4
```

## When To Use

- Use `submit-baseline` when `baseline/` needs to be accepted before any optimize round may begin or continue.
- Use `start-round` immediately before beginning work on a new `opt-round-N/`.
- Use `submit-round` after one round is complete and before the workflow may continue or stop.

## Subcommands

### `submit-baseline`

- Validates canonical baseline artifacts and `baseline/state.json`.
- Prints JSON only; read the `guideline` field for the pass/fix instruction.
- Treat returned `issues` as the baseline repair checklist.
- Baseline preparation still belongs to `ascend-npu-prepare-optimize-baseline`.

### `start-round`

- Enforces the runner-managed `.triton-agent/state.json` workflow gate before a round begins.
- Prints JSON only; read the `guideline` field and keep the returned `hard_rules` in force for the active round.
- Use this to bridge temporary runner-managed workflow state with the durable `opt-round-N/` you are about to work in.

### `submit-round`

- Validates one completed `opt-round-N/` directory against the baseline contract and round-state contract.
- Prints JSON only; read the `guideline` field for the pass/fix instruction, and read `next_option` when it is present.
- When `--current-round` and `--final-round` are provided, next-step guidance is relative to the current worker batch instead of deciding whether the whole optimize session stops.

## Hard Rules

- Only one optimize round may be active at a time.
- Do not use a script to create multiple optimize rounds where each round only adjusts parameters in order to speed up the optimization process. This is cheating behavior and is strictly prohibited.
- Do not use agents or subagents to advance multiple rounds in parallel while the current round is still in flight.
- Do not treat the next round as a blind parameter sweep. If you need to tune parameters, prefer the `autotune` optimization pattern.
- Do not burn rounds on hand-tuned launch or tile sweeps unless existing evidence clearly justifies that direction.
- Before editing code, decide which operator, kernel path, or wrapper bottleneck should anchor the next round.
- Before editing code, decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first.
- Keep the round goal narrow: one coherent hypothesis, one active round, one evidence-backed change direction.
