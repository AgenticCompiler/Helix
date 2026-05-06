# Run-Eval Skill Doc Split Design

## Summary

Split `skills/triton-npu-run-eval/SKILL.md` into a short router plus command-specific reference docs so agents can load only the usage guidance they need for the current run.

## Goals

- Keep the public `triton-npu-run-eval` skill name and helper-script entrypoint unchanged.
- Shrink `SKILL.md` into a small routing contract that tells the agent which focused doc to read next.
- Move `run-test`, `run-bench`, `profile-bench`, `compare-result`, and `compare-perf` usage details into separate Markdown files.
- Preserve the existing command semantics, required flags, mode-override rules, and representative remote examples.

## Non-Goals

- Do not change CLI behavior, helper-script behavior, or remote execution behavior.
- Do not rename subcommands or split the runtime into multiple skills.
- Do not rewrite the Python scripts under `skills/triton-npu-run-eval/scripts/`.

## Design

### 1. Turn `SKILL.md` Into A Router

`skills/triton-npu-run-eval/SKILL.md` should keep only:

- the skill purpose
- the shared helper-script entrypoint
- a small routing table from subcommand to focused doc
- an explicit instruction to avoid reading unrelated command guides or `scripts/*.py` during normal use

This keeps the top-level skill cheap to load while preserving a single public entrypoint for cross-skill references.

### 2. Move Usage Details Into Focused Docs

Add one Markdown file per subcommand under `skills/triton-npu-run-eval/references/`:

- `references/run-test.md`
- `references/run-bench.md`
- `references/profile-bench.md`
- `references/compare-result.md`
- `references/compare-perf.md`

Each file should cover only one command and include:

- the minimal invocation shape
- required flags
- when embedded metadata is used versus when a mode override is appropriate
- remote execution notes when supported
- a small number of representative examples

### 3. Lock The Structure With One Contract Test

Update `tests/test_generation_contracts.py` to protect the new boundary:

- `SKILL.md` stays a router and names the focused docs
- the old long per-command sections do not return to `SKILL.md`
- the focused docs exist and contain the key command-specific guidance that used to live in the monolithic skill file

## Validation

- Run the new targeted generation-contract test for the run-eval doc split.
- Run the full `tests.test_generation_contracts` suite to confirm the documentation refactor does not break other skill contracts.
