# OpenHands Backend Design

## Summary

- Add `openhands` as a new built-in agent backend alongside `codex`, `opencode`, `pi`, and `claude`.
- Keep CLI changes minimal in the first phase: users select `--agent openhands`, and no new OpenHands-specific flags are added yet.
- Implement the backend in-process with the OpenHands Python SDK instead of shelling out to an `openhands` executable.
- Support non-interactive runs only in the first phase; `--agent openhands --interact` should fail fast with a short actionable error.
- Use the OpenHands SDK's built-in coding tools and `LLMSummarizingCondenser` so the backend supports file editing, bash execution, and automatic context compression.

## Goals

- Reuse the existing CLI command surface and orchestration flow as much as possible.
- Keep backend-specific logic isolated from prompt construction, skill staging, and command parsing.
- Preserve the repository rule that workspace-local copied skills remain the source of truth.
- Make OpenHands configuration script-friendly through environment variables only in the first phase.
- Return short user-facing configuration errors instead of raw Python tracebacks when OpenHands setup is incomplete.

## Non-Goals

- Do not add new CLI flags such as `--llm-model` or `--llm-base-url` in the first phase.
- Do not support `--interact` for the OpenHands backend in the first phase.
- Do not introduce remote agent-server execution, Docker sandboxing, or Kubernetes deployment for OpenHands.
- Do not move repository workflow logic from `skills/` into the CLI.
- Do not add OpenHands-specific session persistence or resumable conversation state in the first phase.

## OpenHands SDK Usage

The intended backend design follows the documented standalone SDK flow:

1. Create an `LLM` from environment variables.
2. Create an agent with coding tools.
3. Create a `Conversation` bound to the target workspace.
4. Send the generated prompt and run the conversation.

The documentation also confirms the three capabilities this backend needs:

- Coding tools:
  - The Hello World guide shows either `get_default_agent(llm=llm, cli_mode=True)` or an explicit `Agent(..., tools=[TerminalTool, FileEditorTool, TaskTrackerTool])` pattern for code-oriented tasks.
  - Source: [Hello World](https://docs.openhands.dev/sdk/guides/hello-world)
- Automatic context compression:
  - The SDK guide documents `LLMSummarizingCondenser` as the default LLM-based conversation condenser.
  - Source: [Context Condenser](https://docs.openhands.dev/sdk/guides/context-condenser)
- Project rules and skills:
  - `load_project_skills(work_dir=...)` loads any workspace-owned `AGENTS.md`-style always-on repository context that already exists in the target workspace.
  - In practice, `load_project_skills(work_dir=...)` also discovers staged workspace skills under `.openhands/skills`, so the backend does not need a second explicit `load_skills_from_dir(...)` step.
  - Source: [Agent Skills & Context](https://docs.openhands.dev/sdk/guides/skill)

The SDK getting-started guide also shows that `LLM_MODEL`, `LLM_API_KEY`, and optional `LLM_BASE_URL` are standard environment-variable inputs for local usage.
Source: [Getting Started](https://docs.openhands.dev/sdk/getting-started)

## Approaches Considered

### Recommended: In-Process SDK Backend

- Add an `OpenHandsRunner` that constructs and runs the OpenHands SDK conversation directly in Python.
- Keep the existing `AgentRequest` and `AgentResult` interface so generation and optimize runtime code stay mostly unchanged.
- Treat OpenHands as a first-class backend implementation, not as an external CLI wrapper.

Why this is the best fit:

- OpenHands is fundamentally an SDK-first runtime, so an in-process backend matches the upstream product model.
- It avoids an extra shim script or subprocess protocol layer.
- It keeps the CLI thin and preserves the current repository architecture where backend modules only own backend-specific launch behavior.

### Alternative: Subprocess Wrapper Around an Internal Python Worker

- Add `openhands` as a pseudo-external backend that actually launches a repository-local Python entrypoint.

Why not choose this now:

- It adds a second protocol layer for stdout, stderr, exit codes, and error propagation.
- It gives up one of the main advantages of using the SDK directly.
- It makes testing and configuration harder without delivering user-visible value.

### Alternative: Full OpenHands Integration Including Interactive Sessions

- Implement OpenHands as a fully interactive backend from day one.

Why not choose this now:

- The current repository's `--interact` semantics are built around attaching to live external agent CLIs.
- The OpenHands SDK documentation focuses on programmatic `Conversation` execution, not on a ready-made interactive terminal flow equivalent to the existing backend model.
- The user explicitly approved a first phase where `--interact` is unsupported.

## User-Facing Behavior

### Backend Selection

- Users can select `--agent openhands` on all existing agent-backed commands.
- No other CLI flags change in the first phase.

### Non-Interactive Execution

- OpenHands runs in-process against the target workspace.
- The generated prompt remains the primary task contract.
- The backend can edit files and run bash commands through OpenHands tools.
- Context compression is enabled by default through `LLMSummarizingCondenser`.

### Interactive Execution

- `--agent openhands --interact` is rejected immediately.
- The CLI should surface a short message such as:

```text
OpenHands backend does not support --interact yet.
```

### Configuration

- The first phase reads:
  - `LLM_MODEL`
  - `LLM_API_KEY`
  - `LLM_BASE_URL` (optional)
- Missing required configuration should fail with short actionable messages rather than Python tracebacks.

Examples:

```text
OpenHands backend requires LLM_API_KEY to be set.
OpenHands backend requires LLM_MODEL to be set.
```

## Proposed Design

### CLI Surface

- Extend `_AGENT_CHOICES` in `src/triton_agent/cli.py` to include `openhands`.
- Do not add backend-specific CLI flags in this change.
- Validate the unsupported interactive case in command-handling code before attempting to launch OpenHands.

This keeps the first-phase CLI change intentionally small and reversible.

### Backend Module

Add a new module:

- `src/triton_agent/backends/openhands.py`

Responsibilities:

- Validate environment-based OpenHands configuration.
- Construct the OpenHands `LLM`.
- Construct the OpenHands agent and conversation.
- Load workspace-local project context and staged skills.
- Drive the conversation from the existing `AgentRequest.prompt`.
- Return a normal `AgentResult`.

The backend should continue to support the existing shared `resume()` flow by reusing `AgentRunner.resume()`, which rebuilds a continuation prompt. The first phase should not attempt to preserve OpenHands conversation state across separate CLI invocations.

### Agent Construction

The backend should use a conservative explicit-tool construction rather than depending on a larger preset surface.

Recommended tool set:

- terminal tool
- file editor tool
- task tracker tool

Recommended conversation additions:

- `LLMSummarizingCondenser`
- `NeverConfirm()` confirmation policy for script-friendly non-interactive behavior

Rationale:

- The repository only requires code-editing and shell-execution capabilities for this phase.
- An explicit tool list is easier to reason about and test than inheriting every default tool that may change upstream.
- The task tracker tool is already shown in the OpenHands documentation examples and helps long-running coding tasks without widening the CLI surface.

### Workspace Context And Skills

OpenHands should run against `request.workdir` as the conversation workspace.

Context loading should follow this order:

1. Load any workspace-owned project rules and staged workspace skills from the workspace root through `load_project_skills(work_dir=...)`.
2. Use that returned skill list directly as the OpenHands agent context.

This preserves the repository's current rule that:

- repository `skills/` are copied into the workspace before launch
- the copied workspace-local skill tree is what the backend should read
- cleanup removes only the copies created by the current run

### Skill Staging

Extend `SkillLinkManager` with OpenHands-specific staging under:

- `.openhands/skills`

Behavior should mirror the existing backend-specific skill staging rules:

- copy skill directories rather than symlink
- fail if the target skill path already exists as a symlink
- only remove copies created by the current run
- never delete unrelated user-owned directories

The OpenHands backend should then load skills from that staged path rather than from the repository source directory.

OpenHands should not auto-copy this repository's top-level `AGENTS.md` into the target workspace. To stay aligned with the existing backends, it should stage skills only and otherwise respect whatever guidance files the workspace already owns.

### Output And Result Mapping

OpenHands runs in-process, so stdout and stderr are not naturally identical to the external-CLI backends.

First-phase result behavior:

- Capture a textual summary of the final OpenHands conversation output into `AgentResult.stdout`.
- Surface backend errors as `AgentResult.stderr`.
- Return `session_id=None`.
- Keep exit codes aligned with the existing convention:
  - `0` on success
  - non-zero on configuration or runtime failure

For `--show-output`, the backend may optionally stream simple event text during execution, but the first phase does not need to reproduce a full OpenHands TUI or event renderer.

### Dependencies

Add runtime dependencies in `pyproject.toml`:

- `openhands-sdk`
- `openhands-tools`

Do not add the optional remote/server packages in this phase:

- `openhands-workspace`
- `openhands-agent-server`

### Error Handling

The first phase should prefer explicit friendly failures for these cases:

- unsupported `--interact`
- missing `LLM_API_KEY`
- missing `LLM_MODEL`
- OpenHands package import failure
- skill loading failure from the staged workspace path

The CLI should keep its existing "short actionable error" style and avoid raw tracebacks for expected setup problems.

## Testing Strategy

### Parser Coverage

Update `tests/test_cli.py` to verify that agent-backed commands accept `--agent openhands`.

### Factory Coverage

Update `tests/test_backends_factory.py` so `create_runner("openhands")` returns `OpenHandsRunner`.

### Skill Staging Coverage

Update `tests/test_skills.py` with coverage for:

- copying `.openhands/skills` when absent
- rejecting existing symlink targets
- preserving unrelated user-owned content during cleanup

### Backend Coverage

Add `tests/test_openhands_runner.py` covering:

- environment validation failures
- rejection of `--interact`
- successful request execution via mocked OpenHands SDK objects
- `AgentResult` mapping from SDK execution outcome
- `resume()` reuse of the shared continuation prompt path

## Risks And Mitigations

- Risk: OpenHands SDK object construction may differ slightly from the current docs.
  - Mitigation: keep the implementation isolated in one backend module and write tests against the adapter boundary, not deep SDK internals.
- Risk: an upstream preset helper like `get_default_agent()` may not expose all context-injection hooks cleanly.
  - Mitigation: prefer explicit `Agent` plus explicit tools and context loading in the backend.
- Risk: in-process execution may produce different output formatting than external CLI backends.
  - Mitigation: keep the first phase focused on correctness and friendly error handling, not on perfect parity of streamed output.
- Risk: unsupported interactive semantics may surprise users.
  - Mitigation: fail fast with a short message before any workspace mutation or backend startup.

## Verification

Run at least:

- `uv run python -m unittest tests.test_cli tests.test_backends_factory tests.test_skills tests.test_openhands_runner -v`
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`
