# Optimize Session Artifacts Refactor Design

## Context

`src/triton_agent/optimize/guidance.py` currently mixes several different responsibilities behind `OptimizeGuidanceManager`:

- rendering temporary optimize memory files (`AGENTS.md` / `CLAUDE.md`)
- backing up, writing, restoring, and deleting those memory files
- creating and cleaning supervised runtime handoff files under `.triton-agent/`
- archiving optimize session outputs under `optimize-logs/`
- recording agent session ids into `agent-sessions.jsonl`

The implementation still works, but the current shape is hard to reason about because one file and one manager own several artifact domains with different lifecycles.

The word `guidance` is also now too narrow for the actual scope. The current module manages far more than top-level guidance text.

## Decision

Refactor optimize artifact management around artifact domains instead of keeping all responsibilities in one `guidance.py` manager.

Rename the top-level facade to `OptimizeSessionArtifactsManager` and split the current behavior into focused modules:

- `memory_file`
- `runtime_handoff`
- `archive`

The facade should remain thin and orchestration-only. It should compose the domain modules and preserve the current caller experience as much as practical.

## Module Boundaries

### `memory_file`

Owns:

- choosing `AGENTS.md` vs `CLAUDE.md`
- rendering temporary memory-file content
- backing up existing workspace memory files
- writing temporary memory files
- restoring or deleting memory files during cleanup

Does not own:

- `.triton-agent/` runtime files
- `optimize-logs/`
- session recording

### `runtime_handoff`

Owns:

- `.triton-agent/round-brief.md`
- `.triton-agent/supervisor-report.md`
- `.triton-agent/history/`
- supervised runtime-tree cleanup

Does not own:

- top-level `AGENTS.md` / `CLAUDE.md`
- optimize archive layout
- session recording

### `archive`

Owns:

- `optimize-logs/triton-agent/<run-id>/`
- `shared-guidance.md` snapshots
- `final/`
- `history/`
- `agent-sessions.jsonl`

Does not own:

- memory-file rendering
- runtime handoff file creation

### `session_artifacts`

Owns:

- the `OptimizeSessionArtifactsManager` facade
- composition of the three domain modules
- session-level state assembly for supervised and unsupervised flows
- compatibility for current callers in `execution.py`

Does not own:

- detailed rendering logic
- archive file I/O details
- runtime handoff tree logic

## State Design

Replace the current mixed state objects with domain-aligned state:

- `MemoryFileState`
- `ArchiveState`
- `RuntimeHandoffState`

Then compose them into facade-level session state objects instead of one dataclass owning unrelated fields.

The unsupervised session state should include:

- memory-file state
- archive state

The supervised session state should include:

- memory-file state
- archive state
- runtime handoff state

## Compatibility

This refactor should preserve current optimize behavior:

- same memory-file contents
- same backup and restore semantics
- same `.triton-agent/` runtime files
- same archive layout
- same session-recording behavior

This is a structural refactor, not a workflow redesign.

## Testing Strategy

Refactor tests to follow the same artifact boundaries:

- memory-file tests
- runtime-handoff tests
- archive/session-record tests
- a small facade integration layer for optimize execution entry points

Existing optimize runtime behavior should continue to pass without semantic changes.

## Non-Goals

- changing optimize prompts or skills
- changing optimize archive contents or naming
- changing supervised vs unsupervised behavior
- redesigning optimize execution flow
- deduplicating all optimize instruction text across prompts and memory files
