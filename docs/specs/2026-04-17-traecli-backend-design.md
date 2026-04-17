# TraeCLI Backend Design

## Summary

- Add `traecli` as a new built-in code-agent backend alongside `codex`, `opencode`, `pi`, `claude`, and `openhands`.
- Support both interactive and non-interactive execution in the first phase.
- Keep the existing CLI surface unchanged apart from allowing `--agent traecli`.
- Refactor skill staging so backend-specific target directories are data-driven instead of implemented as near-identical per-backend methods.
- Stage repository skills for TraeCLI under `.traecli/skills`.

## Goals

- Reuse the existing generation and optimize orchestration flow without moving workflow logic out of `skills/`.
- Keep backend-specific behavior isolated to a dedicated runner module plus a small backend mapping update.
- Preserve the repository rule that workspace-local copied skills remain the source of truth.
- Reduce duplication in `SkillLinkManager` while keeping current behavior and cleanup guarantees unchanged.
- Make non-interactive TraeCLI runs script-friendly by default.

## Non-Goals

- Do not add TraeCLI-specific CLI flags in this change.
- Do not change prompt construction, optimize resume prompts, or output rendering semantics.
- Do not invent TraeCLI session-disabling behavior that is not documented by the installed CLI.
- Do not migrate existing backend behavior away from copy-based skill staging.
- Do not move project rules from `AGENTS.md` or workflow logic from `skills/` into the CLI.

## Current Context

The repository already exposes backend selection through `--agent` and isolates command construction in backend-specific runner modules. Existing backends stage repository skills into backend-native workspace directories such as `.codex/skills` and `.claude/skills`.

The current skill staging implementation in `src/triton_agent/skills.py` duplicates nearly the same copy, symlink-protection, and cleanup logic once per backend. Adding another backend by copying that pattern again would continue the redundancy without adding user-visible value.

Local inspection of the installed TraeCLI binary and help output on 2026-04-17 shows:

- executable name: `traecli`
- interactive invocation shape: `traecli <prompt>`
- non-interactive invocation shape: `traecli --print <prompt>`
- script-friendly approval bypass flag: `--yolo`
- session-related flags such as `--resume` and `--session-id`
- no documented flag for disabling session persistence

The same local inspection also surfaced references to `AGENTS.md` and `**/SKILL.md`, which is sufficient for a first-phase design that stages skills into the workspace and runs TraeCLI from that workspace root.

## User-Facing Behavior

### Backend Selection

- Users can select `--agent traecli` on all existing agent-backed commands.
- No other CLI flags change in the first phase.

### Interactive Execution

- Interactive commands launch TraeCLI directly in the target workspace:

```text
traecli <prompt>
```

- The generated prompt remains the initial task contract, matching the current backend model.

### Non-Interactive Execution

- Non-interactive commands launch TraeCLI in print mode with approval bypass enabled:

```text
traecli --print --yolo <prompt>
```

- This keeps TraeCLI aligned with the repository's existing expectation that non-interactive agent-backed commands are script-friendly and should not stop for tool confirmations.

### Optimize Session Behavior

- `optimize --no-agent-session` is accepted because it is part of the shared CLI contract.
- The TraeCLI backend ignores that option in the first phase because the installed CLI does not expose a documented no-persistence or no-session flag.
- This mirrors the repository's existing approach for backends that cannot map every shared session option.

### Workspace Context And Skills

- TraeCLI runs with `request.workdir` as the process working directory.
- Repository skills are staged into `.traecli/skills` inside that workspace before launch.
- The backend does not copy this repository's top-level `AGENTS.md` into the workspace; it only stages skills and otherwise respects whatever repository-owned guidance files the workspace already contains.

### Output And Diagnostics

- TraeCLI output continues to flow through the shared process runner.
- Any startup warning emitted by TraeCLI itself, such as the currently observed keyring warning, is passed through unchanged in the first phase.
- The backend does not introduce custom output filtering or session-id extraction initially.

## Approaches Considered

### Recommended: Add TraeCLI Backend And Refactor Skill Staging Together

- Add a dedicated `TraeCLIRunner`.
- Extend backend selection and factory wiring to include `traecli`.
- Replace duplicated skill staging methods with one generic implementation driven by backend-to-directory mappings.

Why this is the best fit:

- It delivers the requested backend without copying another block of near-identical skill staging code.
- It keeps the CLI thin and preserves the current backend module boundaries.
- It reduces future backend-addition cost to a mapping change plus a runner module.

### Alternative: Add Only The Runner And Reuse Another Backend's Skill Directory

- Treat TraeCLI as a thin subprocess wrapper but do not add `.traecli/skills`.

Why not choose this:

- It would violate the repository's existing backend-specific workspace staging pattern.
- It would make the workspace layout misleading because staged skills would appear owned by a different tool.

### Alternative: Add TraeCLI By Copying Another `prepare_*_skills()` Implementation

- Add a new `prepare_traecli_skills()` method by duplicating the existing code shape.

Why not choose this:

- It would continue the redundancy the user explicitly called out.
- It would make `SkillLinkManager` harder to maintain for no semantic benefit.

## Proposed Design

### CLI Surface

- Extend `_AGENT_CHOICES` in `src/triton_agent/cli.py` to include `traecli`.
- Do not add TraeCLI-specific CLI flags in this change.

### Backend Module

Add a new module:

- `src/triton_agent/backends/traecli.py`

Responsibilities:

- build TraeCLI command lines from `AgentRequest`
- rely on the shared `AgentRunner.run()` and `AgentRunner.resume()` flow
- keep backend-specific behavior limited to TraeCLI command construction

Command construction:

- interactive:
  - `traecli <prompt>`
- non-interactive:
  - `traecli --print --yolo <prompt>`

The backend should not override shared mode selection, resume behavior, or output filtering in the first phase.

### Backend Factory

- Update `src/triton_agent/backends/factory.py` so `create_runner("traecli")` returns `TraeCLIRunner`.

### Skill Staging Refactor

Refactor `src/triton_agent/skills.py` so backend-specific skill staging uses a shared implementation plus a target-directory mapping.

Recommended mapping shape:

```python
{
    "codex": (".codex", "skills"),
    "opencode": (".opencode", "skills"),
    "pi": (".pi", "skills"),
    "claude": (".claude", "skills"),
    "openhands": (".openhands", "skills"),
    "traecli": (".traecli", "skills"),
}
```

Behavior must stay the same as today:

- copy directories instead of creating symlinks
- fail if the target root or staged skill path already exists as a symlink where that would be unsafe
- support staging all skills when no specific skill list is requested
- support staging only selected skill directories when a skill list is provided
- only clean up paths created by the current run
- never delete unrelated user-owned directories

The refactor should preserve the public `prepare_skills()` entrypoint so callers outside `skills.py` do not need architectural changes.

### Testing Strategy

Update or add tests for:

- parser coverage showing `--agent traecli` is accepted on agent-backed commands
- backend factory coverage showing `create_runner("traecli")` returns `TraeCLIRunner`
- TraeCLI runner command construction:
  - interactive command
  - non-interactive command
  - `no_agent_session` ignored
  - verbose launch logging
  - shared process-runner dispatch
- skill staging refactor behavior:
  - target directory mapping still stages existing backends correctly
  - `.traecli/skills` staging works
  - symlink rejection still works
  - cleanup only removes paths created in the current run

### Documentation

- Update `README.md` to mention `traecli` in all existing backend lists for agent-backed commands.
- Keep durable project rules in `AGENTS.md` unchanged because this change follows existing backend conventions rather than changing them.

## Risks And Mitigations

- Risk: the generic skill-staging refactor could accidentally change existing backend behavior.
  - Mitigation: preserve current semantics exactly and broaden tests around staging, symlink rejection, and cleanup.
- Risk: TraeCLI may expect a different skill-discovery path than `.traecli/skills`.
  - Mitigation: this path is explicitly user-directed and consistent with the repository's backend-specific staging model; keep the behavior isolated so it is easy to adjust later if TraeCLI documents a stronger contract.
- Risk: TraeCLI's session model may differ from other backends during long optimize workflows.
  - Mitigation: rely on the existing shared resume prompt flow and explicitly ignore `--no-agent-session` until a documented TraeCLI mapping exists.
- Risk: TraeCLI may emit environment-specific warnings before normal output.
  - Mitigation: pass them through unchanged in the first phase and avoid brittle output parsing.

## Verification

- Run focused regression tests while implementing:
  - `uv run python -m unittest tests.test_cli tests.test_backends_factory tests.test_skills tests.test_traecli_runner -v`
- Run repository verification before completion:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
