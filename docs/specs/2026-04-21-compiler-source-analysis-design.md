# Compiler Source Analysis Design

## Summary

- Add an optional compiler-source analysis capability for `optimize` and `optimize-batch`.
- Keep compiler source provisioning in the CLI instead of asking the launched code agent to clone or update repositories.
- Cache a shallow local checkout of AscendNPU-IR under the user's Triton Agent home directory.
- Add a dedicated `triton-npu-analyze-compiler-source` skill that explains the compiler project, guides read-only source analysis, and produces round-local compiler analysis evidence.
- Defer detailed compiler source indexing design to a later spec.

## Goals

- Give optimize agents a controlled way to inspect Triton Ascend compiler source when profiler and IR evidence are not enough to explain a round.
- Preserve a clear boundary between orchestration and analysis:
  - the CLI prepares and validates local compiler source resources
  - skills own the analysis workflow and compiler-project guidance
  - agents inspect the provided checkout as read-only evidence
- Avoid exposing repository clone URLs or update commands to the launched agent.
- Avoid silently changing compiler source between optimize runs.
- Keep compiler source analysis opt-in so normal optimize runs do not pay clone or prompt costs.

## Non-Goals

- Do not make compiler source analysis part of the default optimize workflow.
- Do not replace profiler or IR analysis with compiler source reading.
- Do not require compiler source evidence for every round when the feature is enabled.
- Do not ask the launched agent to run `git clone`, `git fetch`, `git pull`, or otherwise update the compiler source checkout.
- Do not copy compiler source into the operator workspace or backend-specific skill staging directories.
- Do not design the detailed source index format yet.
- Do not modify or patch the compiler source checkout during operator optimization.

## User-Facing Behavior

### Enabling The Capability

`optimize` and `optimize-batch` accept a new option:

```bash
uv run triton-agent optimize -i a.py --enable-compiler-source-analysis
uv run triton-agent optimize-batch -i operators --enable-compiler-source-analysis
```

When enabled, the CLI prepares the compiler source checkout before launching the code agent.

The default compiler source location is:

```text
~/.triton-agent/compiler-sources/AscendNPU-IR/
```

The default source repository is fixed in CLI code:

```text
https://gitcode.com/Ascend/AscendNPU-IR.git
```

The launched agent should not receive this repository URL. It should receive only the prepared local source path and resolved commit.

### Existing Local Source

Advanced users may point the CLI at an existing checkout:

```bash
uv run triton-agent optimize -i a.py \
  --enable-compiler-source-analysis \
  --compiler-source-path /path/to/AscendNPU-IR
```

The supplied path must exist and be a usable git checkout. The CLI validates it before launching the agent.

`--compiler-source-path` is only valid when compiler source analysis is enabled.

### Provisioning Rules

When compiler source analysis is enabled and no explicit source path is provided:

- If the default checkout does not exist, the CLI creates its parent directory and runs a shallow clone with `--depth 1`.
- If the default checkout already exists and is a git checkout, the CLI reuses it.
- If the default checkout exists but is not a directory, is not a git checkout, or appears incompatible, the CLI fails with a short actionable error.
- The CLI does not run `git pull`, `git fetch`, or automatic refresh during normal optimize startup.
- Future refresh behavior should use a separate explicit option such as `--refresh-compiler-source`, not implicit updates.

The CLI records the resolved source path, current commit, and dirty state. The agent prompt and workspace guidance should include those values.

## Agent Contract

When compiler source analysis is enabled, prompts and temporary workspace guidance should tell the launched agent:

- Compiler source analysis is enabled for this optimize run.
- The local compiler source checkout path is `<compiler_source_path>`.
- The compiler source commit is `<compiler_source_commit>`.
- Treat the checkout as read-only.
- Do not run `git clone`, `git fetch`, `git pull`, or modify files in the compiler source checkout.
- Use the staged `triton-npu-analyze-compiler-source` skill only when compiler source evidence is needed.
- Prefer the normal evidence ladder first: benchmark and correctness results, then profiler evidence, then IR evidence, then compiler source.

Compiler source analysis should be used as an escalation path, not as the first analysis step.

## Skill Contract

Add a new skill:

```text
skills/triton-npu-analyze-compiler-source/
```

The skill owns compiler-source analysis guidance. It should include:

- a concise English overview of AscendNPU-IR and its role in Triton Ascend lowering
- the expected relationship between benchmark/profile symptoms, IR stages, compiler passes, and source files
- rules for using the CLI-provided checkout path instead of cloning or updating source
- instructions for tracing a concrete IR stage, pass name, compiler error, op name, or lowering symptom into the source tree
- requirements for distinguishing facts from inference
- requirements for recording the source commit and version-match uncertainty
- the output contract for `opt-round-N/compiler-analysis.md`

The skill should not depend on a detailed index in its first version. It may reserve space for future helper scripts and index artifacts, but index structure is a later design.

## Triggering Rules

Compiler source analysis is allowed only when the capability is enabled.

Even when enabled, the agent should escalate to compiler source only when at least one of these conditions is true:

- A compiler, lowering, or legality failure blocks the round and cannot be explained from the operator code or error text alone.
- IR inspection shows a suspicious pass transition, layout conversion, copy insertion, synchronization, vectorization loss, fusion loss, or buffer expansion that needs source-level explanation.
- Profiling and IR evidence identify a symptom but do not explain why the compiler produced the observed lowering.
- Multiple attempts in the same optimization direction have stalled because the current evidence does not identify a next concrete operator change.
- Chip-specific behavior differs in a way that architecture notes and IR evidence do not explain well enough.

The agent should not use compiler source analysis when:

- benchmark, profile, or IR evidence already supports a concrete next change
- no compiler error, IR stage, pass name, op name, or lowering symptom has narrowed the source search
- the only purpose is broad background reading

## Output Artifact

When used in an optimize round, the compiler source analysis skill writes:

```text
opt-round-N/compiler-analysis.md
```

The document should contain:

- `# Compiler Source Analysis`
- `## Executive Summary`
- `## Trigger`
- `## Compiler Source Context`
- `## IR Or Error Evidence`
- `## Source Files Inspected`
- `## Source-Backed Explanation`
- `## Impact On Current Operator`
- `## Recommended Next Change`
- `## Confidence And Evidence Gaps`

The analysis must cite local source paths and the compiler source commit. If the source commit cannot be matched to the installed compiler/toolchain version, the analysis must state that uncertainty and avoid presenting source-derived explanations as proven facts.

Round summaries and `perf-analysis.md` may link to `compiler-analysis.md` when it exists, but they should not duplicate the full analysis.

## CLI And Model Shape

Add an internal option for compiler source analysis mode:

```text
compiler_source_analysis: off|auto
```

For the first user-facing version, expose only:

```text
--enable-compiler-source-analysis
```

The flag maps to `compiler_source_analysis="auto"`. The default is `off`.

Add an optional path override:

```text
--compiler-source-path <path>
```

The request model should carry:

- `compiler_source_analysis`
- `compiler_source_path`
- `compiler_source_commit`
- `compiler_source_dirty`

These values should be available to prompt construction, temporary workspace guidance, supervised resume prompts, and optimize-batch child requests.

## Skill Staging

`optimize` and `optimize-batch` should stage every skill directory under the repository `skills/` root.

This keeps the optimize workflow aligned with its role as a multi-skill orchestration flow. The optimize skill may call sibling skills for generation, validation, profiling, IR inspection, round analysis, repair guidance, and future compiler-source analysis. Staging all repository skills avoids a brittle dependency allowlist that must be updated every time a workflow skill gains a new sibling dependency.

Compiler source analysis options do not control skill visibility. They control only:

- whether the CLI provisions or validates the external compiler source checkout
- whether prompts and temporary workspace guidance include compiler-source path, commit, and read-only instructions
- whether the agent is allowed to use compiler-source analysis as an evidence escalation path

Other commands may continue using smaller command-specific staged skill sets when they do not need the full optimize workflow.

## Implementation Shape

Add a small compiler source provisioning module under `src/triton_agent/optimize/` or another feature-local package. It should:

- resolve the Triton Agent home directory, defaulting to `~/.triton-agent`
- resolve the default source checkout path
- clone the fixed repository with `--depth 1` when the checkout is missing
- validate explicit and default checkouts
- read the current commit with non-interactive git commands
- detect dirty state without modifying the checkout
- return a compact result object for prompt and guidance construction
- raise concise user-facing errors for invalid paths, clone failures, or git inspection failures

Prompt and guidance generation should append compiler-source instructions only when analysis is enabled and provisioning succeeded.

The implementation should not add runtime artifact enforcement in the supervisor for `compiler-analysis.md`. The artifact is required only when the agent actually uses compiler source analysis in a round.

## Testing

Add tests for:

- parser support for `--enable-compiler-source-analysis` on `optimize` and `optimize-batch`
- parser rejection or command validation for `--compiler-source-path` without `--enable-compiler-source-analysis`
- default path resolution under a fake home or configured Triton Agent home
- clone command construction uses the fixed URL and `--depth 1`
- existing git checkout reuse does not run pull or fetch
- invalid existing checkout fails with a concise error
- request plumbing carries compiler source path, commit, dirty state, and mode
- prompt and guidance text include local path and read-only rules when enabled
- prompt and guidance text omit compiler source details when disabled
- optimize and optimize-batch stage all repository skills regardless of compiler source analysis mode
- compiler source analysis mode controls provisioning and prompt/guidance activation, not skill staging
- optimize-batch passes the effective compiler source settings through to each workspace run

Run the standard repository verification commands documented in `README.md` after implementation.
