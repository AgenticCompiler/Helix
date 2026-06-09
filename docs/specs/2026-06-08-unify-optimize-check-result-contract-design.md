# Unify Optimize Check Result Contract Design

## Summary

- Keep the optimize baseline and optimize round skill helpers independently loadable and self-contained.
- Align the public `OptimizeCheckResult` dataclass shape across both helpers.
- Avoid introducing a new cross-skill Python import dependency just to share one dataclass.

## Problem

- `optimize_submit_baseline_contract.py` currently narrows `OptimizeCheckResult.kind` to `Literal["baseline"]`.
- `optimize_submit_round_contract.py` exposes `OptimizeCheckResult.kind` as `Literal["baseline", "round"]`.
- That difference is valid locally, but it is confusing when reading the codebase because both classes represent the same public result payload shape.

## Decision

- Keep two contract modules, one per staged skill, so each skill remains self-contained.
- Make both `OptimizeCheckResult` definitions expose the same public field shape:
  - `kind: Literal["baseline", "round"]`
  - `status: Literal["pass", "fail"]`
  - `issues: tuple[str, ...]`
  - `summary: str`
  - `next_option: str | None`

## Non-Goals

- Do not add a new shared skill only to host this dataclass.
- Do not make skill-side helpers import `triton_agent`.
- Do not change runtime behavior or optimize check semantics.

## Verification

- Add a regression test that baseline and round check-result contracts expose the same public field names and `kind` annotation.
- Run targeted tests plus the standard repository verification commands.
