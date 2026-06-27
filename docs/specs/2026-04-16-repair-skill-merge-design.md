# Repair Skill Merge Design

## Summary

Merge `triton-npu-log-repair` into `triton-npu-repair-guide` so operator repair guidance and post-fix repair logging live in one skill. The standalone log-only skill adds very little independent value and makes the repair workflow harder to discover.

## Goals

- Keep one repair-focused skill entrypoint.
- Preserve the existing repair-experience reference workflow.
- Preserve the repair-log behavior after successful novel fixes.
- Store the repair log under the surviving repair skill directory.

## Non-Goals

- Do not change the repair heuristics themselves.
- Do not change generation or optimize CLI behavior.
- Do not redesign the repair log format.

## Design

- Keep `skills/triton/triton-npu-repair-guide/` as the surviving skill.
- Move `skills/triton-npu-log-repair/output.md` to `skills/triton/triton-npu-repair-guide/output.md`.
- Extend `skills/triton/triton-npu-repair-guide/SKILL.md` with append-only logging instructions for novel successful repairs.
- Delete `skills/triton-npu-log-repair/`.
- Update repository references so flows that previously pointed to `triton-npu-log-repair` now point to `triton-npu-repair-guide` and `../triton-npu-repair-guide/output.md`.

## Verification

- Add or update contract tests to assert that repair-guide documents the append-only logging path.
- Run focused tests covering skill contracts and staging assumptions.
