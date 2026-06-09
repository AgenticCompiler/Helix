# Agent-Facing Generation Contract Boundary Design

## Summary

Keep agent-facing generation skills and reference specs focused on the generated artifact contract, not on runner-private execution details. For import-only generated files, the prompt should describe metadata, exported hooks, determinism, device rules, and result semantics, while internal runner flags, parser technology, and invocation shape stay in repository design docs and runtime implementation.

## Goals

- Remove runner-private benchmark and differential-test CLI details from agent-facing generation prompts.
- Keep generated artifact requirements clear enough for the agent to produce the right file shape.
- Preserve concrete CLI details only where the generated artifact itself must expose that CLI.

## Non-Goals

- Do not hide or weaken internal runtime design documentation.
- Do not rewrite execution skills whose job is explicitly to run repository commands.
- Do not remove concrete CLI requirements from standalone generated files that are intentionally self-executing.

## Decision

### Agent-facing generation docs

Agent-facing generation skills and referenced spec files should describe:

- required metadata headers
- exported hook names or direct script behavior that the generated file must implement
- deterministic input and case-declaration requirements
- device, correctness, profiling, and artifact semantics
- public-entrypoint resolution rules such as supported `api-kind` values

For import-only generated files, these docs should avoid runner-private details such as:

- repository-specific flag names like `--operator-file`
- parser implementation details such as `argparse`
- internal command rendering or invocation sequences

Instead, they should describe the contract generically, for example:

- external execution tooling provides the imported operator module object
- external execution tooling consumes exported hooks and owns execution or artifact writing

### Internal design docs

Internal repository docs under `docs/specs/` and runtime implementation may continue to describe:

- exact CLI flags
- command rendering
- staging behavior
- parser choices
- remote and local execution details

### Validation guidance

Top-level generation skills may still point Codex to the repository's validation commands, but that guidance should stay separate from the generated-file contract itself.

## Scope

Apply this boundary immediately to agent-facing import-only benchmark specs and differential test specs. Keep standalone test specs unchanged where the generated file is intentionally a directly executable CLI artifact.
