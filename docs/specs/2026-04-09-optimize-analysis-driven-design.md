# Optimize Analysis-Driven Design

## Summary

- Make `optimize` workflows hypothesis-driven instead of parameter-thrashing by default.
- Require each optimization round to explain why the chosen change is likely to help before editing code.
- Reuse existing test and benchmark harnesses when they already exist; generate them only when they are missing.
- Add an optional `--require-analysis` flag for `optimize` and `optimize-batch` that strengthens analysis requirements through prompt and guidance text without creating a second optimize skill workflow.

## Goals

- Push code agents to diagnose likely bottlenecks before repeatedly changing tiling or launch parameters.
- Keep the optimize skill as the single workflow source of truth while tightening its default reasoning discipline.
- Make every round record both the optimization hypothesis and the evidence behind it.
- Preserve the current optimize wrapper shape: the CLI orchestrates, while skills own the workflow guidance.
- Keep strict analysis mode lightweight by implementing it as prompt and guidance reinforcement only.

## Non-Goals

- Do not add a second optimize skill or a flag-controlled skill variant.
- Do not require both profiling and IR capture for every round.
- Do not make harness generation mandatory when reusable harnesses already exist.
- Do not add new optimize subcommands or hard runtime artifact enforcement in the supervisor.

## User-Facing Behavior

### Default Optimize Workflow

- `optimize` first checks whether correctness tests and benchmark cases already exist in the workspace.
- If reusable harnesses already exist, the agent should use them instead of regenerating them.
- If a required harness is missing, the agent may generate it through the existing helper flow before optimization starts.
- Before each optimization round, the agent must state:
  - the optimization hypothesis
  - why that change may improve performance
  - what evidence supports the hypothesis
- Acceptable evidence may come from:
  - code inspection
  - benchmark observations
  - profiling data
  - IR inspection
  - a combination of the above
- If the agent skips profiling or IR capture for a round, it must explain why the available evidence is already sufficient.

### Strict Analysis Flag

- `optimize` accepts `--require-analysis`.
- `optimize-batch` accepts the same flag and passes it through to each workspace run.
- The flag does not select a different skill or different orchestration path.
- Instead, it strengthens optimize prompts and temporary workspace guidance so the agent treats analysis as a stronger gate.
- In strict mode, the optimize guidance should explicitly tell the agent to gather profiling or IR-backed evidence before the first code-changing round unless it records a concrete reason why one analysis path is unavailable and the remaining evidence is still sufficient.

## Skill Contract Changes

The optimize skill should move from "analysis when needed" toward "diagnosis before code changes."

### Workflow Expectations

- Reuse existing validation artifacts when they are present and compatible.
- Run a baseline benchmark before evaluating optimization wins if the workspace does not already have one for the current session.
- Record a short diagnostic summary before the first code-changing round.
- Require each round to start with a justified hypothesis, not only a theme label.
- Use profiling and IR capture as first-class evidence sources for choosing and validating optimization directions.
- Allow skipping profiling or IR capture only when the round log explains why another evidence source is sufficient.

### Pattern Selection Expectations

- Pattern choice must be symptom-driven.
- `tiling`, `autotune`, and launch-parameter changes should not be the default first move simply because they are easy to try.
- Pattern references should explicitly say that benchmark, profiling, IR, or concrete code-structure evidence should justify selection.

## Prompt And Guidance Contract

### Base Optimize Prompt

Fresh and resumed optimize prompts should both tell the agent to:

- treat optimization as a long-running task
- reuse existing harnesses when available
- justify every round with a hypothesis and supporting evidence
- explain why profiling or IR capture was skipped when those tools were not used

### Temporary Workspace Guidance

The optimize-specific temporary `AGENTS.md` or `CLAUDE.md` should additionally tell the agent to:

- check for existing tests and benchmark harnesses before regenerating them
- write a diagnostic summary before the first code-changing round
- keep each round hypothesis and rationale in `attempts.md`
- use profiling and IR tools to guide optimization choices when benchmark numbers alone are not enough

### Strict Analysis Reinforcement

When `--require-analysis` is enabled, prompt and guidance text should add stricter wording:

- gather profiling or IR-backed evidence before the first code-changing round
- if one analysis tool is skipped, record why and what evidence replaced it
- do not begin with blind tiling or launch-parameter search

## Artifact Expectations

Round artifacts stay in the existing layout, but the content requirements tighten:

- `attempts.md` must record the hypothesis and its rationale before code edits
- `summary.md` must describe the evidence that motivated the round
- `opt-note.md` should stay concise but reflect why the winning round was pursued

No new required artifact filenames are introduced.

## CLI And Model Shape

- Add `--require-analysis` to `optimize` and `optimize-batch`.
- Extend optimize option plumbing and agent requests with a boolean `require_analysis`.
- Feed that value into prompt construction, temporary optimize guidance generation, and optimize resume prompts.

## Implementation Shape

- Update optimize skill and references to make diagnosis-first behavior explicit.
- Add CLI plumbing for `--require-analysis`.
- Update prompt generation and temporary optimize guidance rendering to mention the new reasoning rules.
- Preserve the existing supervisor lifecycle while ensuring resumed optimize prompts keep the same analysis expectations.

## Testing

- Parser tests for `--require-analysis` on `optimize` and `optimize-batch`
- Prompt tests for default optimize hypothesis/evidence wording
- Prompt tests for stricter wording when `--require-analysis` is enabled
- Guidance tests for harness reuse, hypothesis recording, and strict-analysis wording
- Runtime/request plumbing tests proving the flag reaches prompt and guidance generation
- Full verification with `ruff`, `pyright`, and `unittest`
