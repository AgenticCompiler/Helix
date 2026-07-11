---
title: Check-Round Min Rounds Exit Signal
created: 2026-05-15
summary: During check-round, tell the agent whether it may exit the optimize session by comparing completed rounds against min_rounds.
---

# Check-Round Min Rounds Exit Signal Design

## Summary

- Add an optional `--min-rounds N` parameter to the `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill's `check-round` script.
- When `--min-rounds` is provided and all existing checks pass, enhance the `summary` field with a clear exit/continue signal based on `opt-round-*` directory count.
- Keep the `decision` field unchanged — min_rounds is an informational signal, not a gate.
- Update the optimize workflow and prompts so agents know to pass `--min-rounds` when invoking `check-round`.

## Problem

The current optimize workflow has inconsistent round-requirement enforcement:

| Mode | How min_rounds is enforced |
|------|---------------------------|
| **Unsupervised** | Prompt-level instruction: `"Complete at least N rounds before deciding the session should stop"`. The agent reads this in the prompt but the `check-round` script itself knows nothing about it. |
| **Supervised** | CLI overrides supervisor's `PASS_STOP` to `PASS_CONTINUE` in `execution.py` (lines 131-141) when round count < min_rounds. The worker agent per round never sees the limit. |

Neither mode surfaces the exit signal through the `check-round` flow — the one validation every round must pass. This means:

- In **unsupervised** mode, the agent relies on remembering prompt instructions. If it loses context (long session, mid-session resume), the min_rounds instruction may be forgotten.
- In **supervised** mode, the worker per round has no awareness of progress toward min_rounds, making it harder for the worker to calibrate effort (e.g., shallow round when N-1 rounds already done, deep round when this is only round 1 of 5 required).
- There is no single source of truth for "can I stop now?" — the answer depends on which layer you ask (prompt vs CLI vs skill).

## Goals

- Make `check-round` the authoritative place where the agent learns whether min_rounds is satisfied.
- Keep the signal human-readable so it works in both supervised and unsupervised contexts without requiring structured parsing.
- Do not change the `decision` semantics — a `pass` round is still a valid round regardless of round count.
- Keep the feature backwards-compatible: when `--min-rounds` is not passed, behavior is identical to today.

## Non-Goals

- Do not change the CLI run-loop's mechanical enforcement of min_rounds (`run_loop.py`, `execution.py`). Those remain as safety nets.
- Do not add `min_rounds` awareness to the `check-baseline` flow.
- Do not introduce a new `decision` value for round-count gating.
- Do not move the round-counting logic out of `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` into the CLI layer.

## Design

### 1. Extend `OptimizeCheckResult` summary (no schema change)

The `OptimizeCheckResult` dataclass and its JSON serialization remain unchanged. The signal is carried in the `summary` field, which is already human-facing text that agents read.

Pros:
- Zero breaking change to any consumer (CLI, tests, subagent prompts).
- Agent sees the exit signal in the same output it's already reading.
- No new fields to synchronize between Python runtime and skill scripts.

Cons:
- Agents must read the summary text (they already do).
- No structured field for programmatic consumption (not needed — the CLI already has its own enforcement).

### 2. Modify `check_round()` in `optimize_check_contract.py`

After the existing pass path (line 406, after kernel continuity check and pt cleanup), add:

```python
def check_round(round_dir: Path, *, min_rounds: int | None = None) -> OptimizeCheckResult:
    # ... existing checks ...
    result = _build_result(kind="round", decision="pass", issues=issues)

    if min_rounds is not None:
        completed = _count_round_directories(round_dir.parent)
        if completed >= min_rounds:
            result = _build_result(
                kind="round",
                decision="pass",
                issues=issues,
                summary=(
                    f"round check passed. "
                    f"Minimum round requirement satisfied ({completed}/{min_rounds}) — "
                    f"the optimize session may stop after this round."
                ),
            )
        else:
            result = _build_result(
                kind="round",
                decision="pass",
                issues=issues,
                summary=(
                    f"round check passed. "
                    f"Round {completed}/{min_rounds} complete — "
                    f"at least {min_rounds - completed} more round(s) required before stopping."
                ),
            )

    return result
```

Key points:
- The `decision` stays `"pass"` — the round artifacts are valid.
- The `summary` now carries the round-progress signal.
- `_count_round_directories` counts `opt-round-*` dirs in the workspace (same logic as `run_loop.py:176-177`).
- When `min_rounds is None`, the summary is unchanged (backwards-compatible).

### 3. Add `--min-rounds` to the `check-round` CLI

In `optimize_check.py`:

```python
round_parser.add_argument("--min-rounds", type=int, default=None)

# In main():
result = check_round(Path(args.round_dir).expanduser().resolve(), min_rounds=args.min_rounds)
```

### 4. Update `_build_result()` to accept optional summary override

Currently `_build_result()` always generates the summary from `kind` and `decision`. Add an optional `summary` parameter:

```python
def _build_result(
    *,
    kind: Literal["baseline", "round"],
    decision: Literal["pass", "revise-required", "hard-fail"],
    issues: tuple[str, ...],
    summary: str | None = None,
) -> OptimizeCheckResult:
    ok = decision == "pass"
    if summary is None:
        summary = f"{kind} check passed" if ok else f"{kind} check requires fixes: {'; '.join(issues)}"
    return OptimizeCheckResult(ok=ok, kind=kind, decision=decision, issues=issues, summary=summary)
```

This is a purely internal helper change — callers that don't pass `summary` get identical behavior.

### 5. Update the CLI check wrapper

In `src/helix/optimize/checks.py`:

```python
def check_round(round_dir: Path, *, min_rounds: int | None = None) -> OptimizeCheckResult:
    module = load_skill_script_module("triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round", "optimize_check")
    return _normalize_result(module.check_round(round_dir, min_rounds=min_rounds))
```

### 6. Update agent-facing instructions

#### 6a. Unsupervised prompt (`build_optimize_unsupervised_prompt`)

The prompt already embeds min_rounds instructions (lines 190-198). Add to the check-round instruction:

```
"After finishing each round, use the staged `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill to run 
`check-round --round-dir opt-round-N --min-rounds {min_rounds}` and repair the round 
until it passes. Read the summary for the exit signal."
```

The `{min_rounds}` value is already available in the prompt builder since it's the same code that inserts lines 190-197.

#### 6b. Optimize SKILL.md (Stage 3: Validate And Record)

Update the check-round instruction (currently around line 130):

```markdown
- Use the sibling `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` skill to run `check-round` 
  with `--min-rounds <N>` (where N is the session's minimum round requirement) 
  and repair the current round until it passes before continuing or stopping.
- After `check-round` passes, read the summary output for the exit signal:
  if minimum rounds are satisfied, the session may stop after this round.
```

### 7. No changes to CLI enforcement

The existing enforcement points remain as safety nets:

- **`run_loop.py:_resume_until_round_requirement_satisfied()`** (unsupervised): Auto-resumes agent when round count < min_rounds.
- **`execution.py:SupervisedOptimizeAdapter.run_supervisor()`** (supervised): Overrides `PASS_STOP` to `PASS_CONTINUE` when round count < min_rounds.

These are mechanical guards — the skill-level signal is the soft instruction that guides agent behavior within those bounds.

## Flow Diagram

```
Agent completes round N
        │
        ▼
Agent runs: check-round --round-dir opt-round-N --min-rounds M
        │
        ▼
  ┌─────────────────────────────┐
  │ check_round() checks:       │
  │ 1. Artifact completeness    │
  │ 2. Round state validity     │
  │ 3. Baseline gate issues     │
  │ 4. Semantic state checks    │
  │ 5. Kernel continuity        │
  │ 6. pt file cleanup          │
  └─────────────┬───────────────┘
                │
        ┌───────┴───────┐
        │               │
     fail             pass
        │               │
        ▼               ▼
  revise-required   ┌──────────────────────────┐
  / hard-fail       │ count = opt-round-* dirs │
  (agent fixes)     └──────────┬───────────────┘
                               │
                       ┌───────┴───────┐
                       │               │
                  count >= M      count < M
                       │               │
                       ▼               ▼
               summary:           summary:
               "satisfied         "N/M complete
               (N/M) —            — M-N more
               may stop"          required"
                       │               │
                       ▼               ▼
                  Agent may       Agent must
                  stop session    continue
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| `--min-rounds` not passed | Summary unchanged: `"round check passed"`. No exit signal. |
| `min_rounds=0` | Should not happen — CLI validation rejects `min_rounds < 1`. |
| Resume session with existing rounds | Count reflects actual dirs on disk. If 3 dirs exist and min=2, summary says satisfied. |
| Round N passes but earlier round N-1 was incomplete | Only completed `opt-round-N/` dirs are counted. If N-1 is missing, count is 1 not 2. |
| Agent ignores exit signal | CLI enforcement (`run_loop.py` / `execution.py`) catches this mechanically. |

## Rollout

1. Add `summary` parameter to `_build_result()` (internal, no breaking change).
2. Add `_count_round_directories()` helper and update `check_round()` in `optimize_check_contract.py`.
3. Add `--min-rounds` argument to `check-round` subparser in `optimize_check.py`.
4. Update `src/helix/optimize/checks.py` to plumb `min_rounds` through.
5. Update `build_optimize_unsupervised_prompt()` to include `--min-rounds` in check-round instruction.
6. Update `skills/triton/triton-npu-optimize/SKILL.md` Stage 3 to document the `--min-rounds` flag.
7. Run existing tests (`test_optimize_checks.py`) to verify no regression.
