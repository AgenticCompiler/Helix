# Optimize Enable Subagent Design

## Summary

- Add one `--enable-subagent` option to `optimize` and `optimize-batch`.
- Support this option only for the `codex`, `opencode`, and `claude` backends.
- Stage one built-in performance diagnosis subagent into the backend-native workspace config area for the current run.
- Keep the first change generic enough to support more built-in subagents later without adding a new staging mechanism per agent.
- Treat subagent use as strongly recommended guidance for the main optimize agent, not as a CLI-enforced hard gate.
- Keep the staged diagnosis subagent strictly analysis-only: it may read existing operator files, generated harnesses, and recorded evidence, and it may collect fresh benchmark, profile, or IR evidence for diagnosis, but it must not perform optimization work itself.
- Ensure staged subagents can use the same staged skill tree safely, without requiring skill-script source reads or backend-specific manual setup from the user.

## Problem

The optimize workflow already stages backend-local skills and temporary guidance files, but it does not stage backend-native subagents. That leaves two gaps:

- there is no backend-native specialist that the main agent can delegate diagnosis work to when optimize evidence is still ambiguous
- there is no shared subagent staging abstraction that can later host more built-in subagents

The motivating subagent for this change is a performance diagnosis advisor. Its job is to analyze the current optimize workspace, read the staged optimize knowledge, and suggest plausible optimization directions based on patterns, symptoms, and existing evidence.

This advisor should not become a parallel optimize worker. The existing optimize round contract must remain intact:

- continuous mode still completes rounds strictly one at a time
- checked and supervised modes still keep round ownership with the main worker invocation
- supervisor behavior does not depend on any separate subagent artifact

## Goals

- Add `--enable-subagent` to `optimize`.
- Add `--enable-subagent` to `optimize-batch`.
- Thread the parsed value through `OptimizeRunOptions` and optimize request construction.
- Fail explicitly when `--enable-subagent` is used with a backend other than `codex`, `opencode`, or `claude`.
- Stage one built-in subagent named `triton-agent-perf-diagnosis-advisor` for supported backends.
- Ensure that staged diagnosis subagent runs may inspect existing test and benchmark harnesses, may collect fresh benchmark, profile, or IR evidence for diagnosis, and cannot perform optimize edits against operator source, generated optimized operators, or round artifacts.
- Keep subagent registration generic so future built-in agents can be added through a registry instead of a new special-case code path.
- Strongly recommend subagent use in optimize prompts and workspace guidance when pattern triage or diagnosis is unclear.
- Make subagent instructions explicitly compatible with staged skills under `.codex/skills/`, `.opencode/skills/`, and `.claude/skills/`.
- Clean up only the files and directories created by the current run.

## Non-Goals

- Do not add a user-facing option to select which subagents are enabled in this change.
- Do not require the main agent to invoke the staged subagent.
- Do not create a required `subagent-advice.md` or other dedicated subagent output file.
- Do not make supervisor auditing depend on whether a subagent was invoked.
- Do not support `pi`, `openhands`, or `traecli` in this change.
- Do not allow subagents to advance multiple optimize rounds in parallel.
- Do not add a new skill staging layout.
- Do not let fresh benchmark, profiler, or IR collection become a backdoor for optimization edits or open-ended workflow ownership.

## Design

### CLI Surface

Both optimize commands should accept:

```text
--enable-subagent
```

Behavior:

- default: disabled
- when enabled, only `--agent codex`, `--agent opencode`, and `--agent claude` are valid
- unsupported backend use should fail before the agent process launches, with a short explicit error

`OptimizeRunOptions` should carry this as:

```python
enable_subagent: bool = False
```

### Built-In Subagent Contract

This change registers one built-in subagent:

- id: `triton-agent-perf-diagnosis-advisor`

Its role is advisory diagnosis only. The rendered backend-specific definitions should instruct it to:

- read the current optimize context before making suggestions
- read existing generated test and benchmark harnesses when they help explain how the operator is exercised
- prioritize the staged optimize knowledge `references/pattern_index.md`
- use symptom cards and specific pattern cards only when the index is not enough
- use existing benchmark, profiler, IR, and round artifacts as supporting evidence when they already exist
- collect fresh benchmark, profiler, or IR evidence when the current diagnosis is blocked by missing measurement data
- use documented helper-script entrypoints and workflow commands for benchmark, profiler, or IR collection instead of inventing ad hoc execution paths
- propose likely bottlenecks, candidate pattern directions, and concrete next validation steps
- return analysis through the backend's native subagent response channel instead of writing files
- never edit operator code, optimized candidates, generated harnesses, or optimize round artifacts
- never run code-generation, patching, or file-editing actions
- never produce or apply candidate optimization patches itself
- avoid advancing a separate optimize round
- avoid reading staged skill implementation files under `skills/*/scripts/` unless the parent explicitly needs helper behavior verification

The instructions should also mention optional staged skills that may appear in some optimize runs:

- `triton-npu-optimize-knowledge`
- `torch-npu-optimize-knowledge` for operator-target runs
- `triton-npu-cann-ext-api-patterns` when CANN extension API guidance is enabled

### Generic Subagent Registry

Introduce a small runtime abstraction for built-in subagent definitions instead of wiring the diagnosis advisor directly into one backend.

Representative shape:

```python
@dataclass(frozen=True)
class SubagentDefinition:
    id: str
    supported_backends: tuple[str, ...]
    description: str
    ...
```

The staging layer should own:

- backend-specific target path resolution
- backend-specific content rendering
- collision checks
- cleanup bookkeeping

This keeps future additions additive. A later change should be able to register another built-in agent by adding one more `SubagentDefinition` and updating prompt guidance if needed.

### Backend-Native Staging

Subagents must be staged into the workspace using each backend's native project-level mechanism.

#### Codex

Stage the custom agent under `.codex/agents/` using the current Codex project-level custom-agent file format.

The rendered Codex config should stay minimal:

- include the agent name/description and diagnosis-focused system prompt
- do not force a read-only sandbox, because benchmark, profiler, and IR collection can write analysis artifacts into the workspace
- do not override worktree, model, approval, or `skills.config`

The important compatibility constraint is inheritance. When the staged Codex custom agent omits those overrides, it can inherit the parent session's worktree and skill configuration instead of creating a second, potentially incompatible environment. That keeps the staged `.codex/skills/` tree available without adding a separate Codex-specific skill override. Because the diagnosis advisor is now allowed to collect fresh analysis artifacts, the no-optimize rule must come from the agent instructions and workflow contract rather than a blanket read-only sandbox.

#### Claude

Stage the subagent under `.claude/agents/` using Claude Code's project subagent format.

The Claude definition should similarly avoid restrictive extra configuration:

- explicitly allow the analysis tools it needs, such as `Read`, `Glob`, `Grep`, and `Bash`
- exclude `Write` and `Edit`
- do not assume the parent has already loaded any particular skill into the current context window
- point the subagent directly at the staged `.claude/skills/.../SKILL.md` and knowledge references it should consult
- use the subagent's `skills` frontmatter to preload the staged optimize knowledge skill when available for the run

Claude subagents inherit the thread's broader environment by default, but this design should not depend on previously loaded skill state. The subagent instructions must therefore be self-sufficient about where the staged optimize knowledge lives. `Bash` is needed for analysis commands that collect fresh evidence, while the prompt contract still forbids code edits and optimization work.

#### OpenCode

Stage the subagent under `.opencode/agents/` using OpenCode's project agent format.

The staged agent definition should:

- use `mode: subagent`
- set agent permissions to allow read-oriented actions plus the minimum command execution needed for diagnosis
- deny `edit`
- allow `bash` only for documented benchmark, profiler, IR, and read-only shell command patterns
- allow `skill` so the agent can use the staged skill tree if needed

This repository already stages `.opencode/opencode.json` to deny the built-in `general` subagent during optimize. That behavior should remain unchanged. `--enable-subagent` should not disable the repository's own staged diagnosis subagent, and the staged OpenCode config must not accidentally deny any tool the diagnosis advisor needs to read staged skills or follow their documented command interface.

### Staged Skill Compatibility

Subagent compatibility with staged skills is a first-class design constraint.

The repository already stages skills per backend into:

- `.codex/skills/`
- `.opencode/skills/`
- `.claude/skills/`

The new subagent layer must preserve access to those trees instead of assuming backend-global skills.

Rules:

- subagent prompts must refer to the staged skill path for the current backend
- rendered definitions should prefer inheriting the parent agent's workspace and tool setup rather than replacing it
- subagents should read `SKILL.md` and checked-in references such as `pattern_index.md` or symptom cards
- subagents should not inspect `skills/*/scripts/` source directly unless explicitly required
- documented helper-script entrypoints for benchmark, profiler, or IR collection are allowed for this diagnosis subagent
- the diagnosis subagent may write analysis artifacts that those commands normally produce, but it must not mutate operator implementations or generated optimize candidates

This means no new skill staging mechanism is needed. The main work is to avoid backend-specific subagent definitions that accidentally hide, override, or restrict the existing staged skill tree.

### Prompt And Guidance Integration

When `--enable-subagent` is active, the optimize worker prompt and temporary workspace guidance should add a short recommendation block.

Representative behavior:

- tell the main agent that `triton-agent-perf-diagnosis-advisor` is available in this workspace
- recommend using it before deep diagnosis or code edits when the bottleneck hypothesis is still unclear
- remind the main agent that subagents may help with supporting analysis only
- remind the main agent that this subagent is diagnosis-only, may read existing harnesses and evidence, may collect fresh benchmark/profile/IR artifacts, and cannot perform optimization edits
- keep the existing round-serialization wording unchanged

This wording should be additive. Not invoking the subagent must not make the run invalid.

### Lifecycle And Cleanup

Stage subagents only for the current run and remove them afterward.

Cleanup requirements:

- remove only the files and directories created by the current run
- do not delete user-owned subagent files
- do not replace user-owned agent directories
- fail explicitly when the exact staged file path already exists or is a symlink

Parent directories such as `.codex/agents/`, `.claude/agents/`, or `.opencode/agents/` may already exist for unrelated user-owned agents. That should be allowed as long as the target staged file path for `triton-agent-perf-diagnosis-advisor` does not already exist.

### Interaction With Existing OpenCode Config Staging

OpenCode already has backend-owned workspace config staging in `OpenCodeRunner.run()`. This change should not introduce a second unrelated writer for `.opencode/opencode.json`.

The design should keep responsibilities separated:

- the subagent staging layer owns `.opencode/agents/...`
- the OpenCode runner continues to own `.opencode/opencode.json`

If the implementation later needs an OpenCode config adjustment specifically for staged custom subagents, it should extend the existing config renderer instead of creating a second config file owner.

## Recommendation

Use a generic `--enable-subagent` flag plus one built-in advisory diagnosis subagent.

This gives optimize a backend-native specialist immediately while keeping the implementation aligned with the repository's long-term need to add more built-in agents later.

## Testing

Add or update tests in these areas:

- `tests/test_cli.py`
  Verify `optimize` and `optimize-batch` parse `--enable-subagent`, default it to `False`, and pass it through `optimize_run_options_from_args()`.
- optimize option validation tests
  Verify unsupported backends fail explicitly when `--enable-subagent` is enabled.
- subagent staging tests
  Verify backend-native paths are rendered under `.codex/agents/`, `.claude/agents/`, and `.opencode/agents/`, and only the current run's files are cleaned up.
- collision tests
  Verify an existing exact subagent file path fails explicitly without overwriting user-owned content.
- OpenCode config tests
  Verify existing `.opencode/opencode.json` staging behavior still denies the built-in `general` subagent and does not regress when subagent support is enabled.
- analysis-only permission tests
  Verify the rendered backend-specific subagent definitions disallow edit tools while still permitting the analysis command surface needed for benchmark, profiler, or IR collection.
- optimize guidance tests
  Verify prompt or memory-file text mentions the staged diagnosis advisor only when `--enable-subagent` is active.

## Scope Boundaries

- Do not add non-optimize uses of `--enable-subagent` in this change.
- Do not change skill staging paths.
- Do not make supervisor require or inspect a dedicated subagent report file.
- Do not redesign backend-wide config ownership beyond the minimal additions needed for staged project subagents.
