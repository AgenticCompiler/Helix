# TileLang Optimize Analysis Ladder Simplification Design

## Summary

Simplify the TileLang optimize workflow so its documented and injected analysis ladder stops at profiling diagnosis.

Today the TileLang optimize skill and shared optimize prompt helpers still describe `IR attribution` and `compiler-source escalation`, but the TileLang workflow does not actually support IR capture or compiler-source analysis. That mismatch makes the workflow over-promise capabilities that are not available.

## Goals

- Remove `IR attribution` and `compiler-source escalation` from the TileLang optimize skill contract.
- Make TileLang optimize prompts and memory-file guidance stop at `pattern triage -> profiling diagnosis`.
- Remove TileLang-only prompt guidance that tells workers or subagents to use IR or compiler-source evidence.
- Add regression tests that lock the TileLang-specific behavior in place.

## Non-Goals

- Do not change the Triton optimize workflow or its four-level analysis ladder.
- Do not remove TileLang IR or compiler-source skills from the catalog or staging tables in this change.
- Do not change generic round-validation or profiling skills.

## Proposed Changes

### 1. Simplify The TileLang Skill Contract

Update `skills/tilelang/tilelang-npu-optimize/SKILL.md` so:

- the default escalation order becomes `pattern triage -> profiling diagnosis`
- the `### IR attribution` section is removed
- the `### compiler-source escalation` section is removed
- output and artifact wording no longer promises round-local `ir/` or `compiler-analysis.md`

The round record structure should still keep `Primary analysis level`, `Supporting evidence`, and escalation tracking, but only for the remaining supported levels.

### 2. Make Shared Prompt Builders Respect TileLang Limits

Update `src/triton_agent/optimize/prompts.py` so TileLang prompt construction:

- emits the shortened analysis ladder
- does not tell the worker to escalate to IR or compiler-source analysis
- does not inject TileLang IR companion guidance into round prompts
- does not inject compiler-source-enabled guidance for TileLang runs
- makes TileLang supervisor auditing use the shortened ladder

The prompt helpers should stay shared, but the analysis-depth wording should branch by language.

### 3. Keep Memory Guidance And Diagnosis Subagents Consistent

Update the optimize memory-file and perf-diagnosis subagent helpers so TileLang runs no longer:

- recommend collecting IR evidence
- preload TileLang IR analysis as part of the diagnosis workflow
- grant TileLang IR helper-script permissions in the Opencode subagent rendering

This keeps workspace-level guidance aligned with the main TileLang optimize prompt.

## Test Impact

Update tests to cover:

- the TileLang optimize skill contract no longer contains IR or compiler-source sections
- TileLang optimize prompts use the shortened ladder while Triton retains the original one
- TileLang optimize guidance and subagent rendering no longer advertise IR collection or compiler-source analysis
