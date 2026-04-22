# Optimize Layered Analysis Default Design

## Summary

- Make layered analysis the default `optimize` workflow instead of an optional strict mode.
- Remove `--require-analysis` from `optimize` and `optimize-batch`.
- Teach optimize agents to escalate analysis in a fixed order:
  - `pattern triage`
  - `profiling diagnosis`
  - `IR attribution`
  - `compiler-source escalation`
- Require each round to record which analysis level it started from and why that level was sufficient or why deeper escalation was needed.
- Keep the CLI thin: this remains primarily a skill, prompt, and guidance contract change rather than a new optimize subcommand or hard runtime state machine.

## Problem

The repository now has several optimize-adjacent inputs:

- optimization pattern references
- profiling workflows and profile analysis
- IR capture and IR analysis
- optional compiler source analysis

Individually, these capabilities are useful. Together, they are easy for code agents to misuse.

The current optimize guidance treats these tools as available evidence sources, but not as a strongly ordered escalation path. In practice this creates several failure modes:

- the agent jumps straight into a familiar pattern without checking whether the code actually matches it
- the agent reaches for IR or compiler source too early
- the agent skips profiler-backed diagnosis when benchmark results are ambiguous
- the agent uses several analysis tools in parallel without a clear reason for the escalation
- the round artifacts explain what tool was used, but not why that tool was the right depth

This leads to noisy optimization loops, inconsistent reasoning quality, and missed use of the most useful evidence.

## Goals

- Make the default optimize workflow systematic and layered from shallow to deep analysis.
- Preserve pattern-based optimization when a strong shallow clue already exists.
- Make profiling the default diagnostic entrypoint when no clear pattern is visible.
- Keep IR as an explanation and attribution layer rather than the default starting point.
- Keep compiler source analysis as the final escalation for concrete compiler-side questions only.
- Make each round's analysis depth visible in round-local artifacts.
- Remove the old flag split between "default optimize" and "strict analysis optimize."

## Non-Goals

- Do not require profiling for every single round regardless of obvious code-structure clues.
- Do not require IR for every profiled round.
- Do not require compiler source analysis for ordinary optimize rounds.
- Do not add a new optimize CLI subcommand.
- Do not introduce a hard runtime state machine that blocks all flexible reasoning.
- Do not make `perf-analysis.md` mandatory for every round.

## User-Facing Behavior

### Default Optimize Workflow

`optimize` and `optimize-batch` should treat layered analysis as the default behavior.

Each round should begin by choosing one entry level:

1. `pattern triage`
2. `profiling diagnosis`
3. `IR attribution`
4. `compiler-source escalation`

The entry level is not arbitrary. It must be justified by the currently available evidence.

### Level 0: Pattern Triage

`pattern triage` is the shallowest level.

Its purpose is:

- inspect the current code structure
- inspect benchmark behavior already available for the current candidate
- consult `references/patterns/index.md`
- decide whether there is an obvious optimization pattern worth trying first

This stage is intentionally lightweight. It is not permission to blindly try several easy patterns.

The outcome should be one of:

- a clearly justified pattern-backed hypothesis exists, so the round may proceed from pattern triage
- no strong pattern clue exists, so the round must escalate to profiling diagnosis

### Level 1: Profiling Diagnosis

`profiling diagnosis` is the default deeper entrypoint for optimize analysis.

Use it when:

- pattern triage does not reveal a strong direction
- benchmark behavior is surprising or ambiguous
- a previous pattern-driven round did not explain the result clearly
- the next optimization idea depends on hotspot, pipeline, utilization, or transfer evidence

At this level, the agent should use profiling evidence to decide the next optimization direction or to determine whether deeper explanation is needed.

### Level 2: IR Attribution

`IR attribution` is the next escalation after profiling.

Use it when:

- profiler signals show suspicious symptoms
- the current implementation problem is still not well explained
- the agent needs lowering- or stage-level evidence to explain profiler findings

IR remains an explanatory and attribution layer. It should not become the default optimize entrypoint.

### Level 3: Compiler-Source Escalation

`compiler-source escalation` is the final escalation layer.

Use it only when:

- compiler source analysis is enabled
- profiling and IR evidence have already narrowed the issue
- a concrete compiler-side question remains unresolved

Typical triggers include:

- suspicious lowering transitions
- pass-specific regressions
- compiler-side explanations for an IR symptom that still lacks source-level clarity

### Reusing Existing Evidence

Later rounds do not need to restart from pattern triage if earlier rounds already produced valid evidence.

For example:

- a later round may start from `profiling diagnosis` if existing `profile/` artifacts are still the relevant evidence base
- a later round may start from `IR attribution` if profiling diagnosis has already been established and the next decision depends on IR explanation

However, the round must explicitly cite the reused evidence path and explain why starting from that deeper level is justified.

## Round Artifact Expectations

### `attempts.md`

At the start of each round, `attempts.md` should record:

- the chosen analysis level
- the current optimization hypothesis
- what evidence justifies starting from that level
- if the level is deeper than pattern triage, why the shallower level was insufficient or already exhausted

If the round escalates to a deeper level later, `attempts.md` should also record:

- the new level
- what question remained unresolved
- why the previous level could not explain it well enough

### `summary.md`

`summary.md` should describe:

- which level the round started from
- whether the round escalated to deeper analysis
- which evidence actually determined the final round decision
- which unresolved questions remain for future rounds

### `perf-analysis.md`

`perf-analysis.md` remains round-local and optional.

It should continue to serve as the standalone artifact for deeper performance diagnosis, especially once the round reaches profiling diagnosis plus optional IR attribution.

### `compiler-analysis.md`

`compiler-analysis.md` remains optional and should only appear when the round genuinely reached compiler-source escalation.

## Skill And Workflow Contract Changes

### Optimize Skill

The optimize skill should make the layered workflow explicit.

It should say all of the following clearly:

- every round begins from an identified analysis level
- pattern triage is the shallow screening layer, not blind pattern search
- profiling diagnosis is the default diagnostic escalation
- IR attribution comes after profiler-backed symptoms exist
- compiler source analysis is the final escalation

### Optimize Workflow Reference

`references/workflow.md` should reflect the same ordered escalation model.

It should describe:

- how to choose the initial level
- when to escalate
- what each escalation must record
- that later rounds may reuse deeper evidence when properly justified

### Round Performance Analysis Skill

`triton-npu-analyze-round-performance` already follows a profiler-first model.

This design should not replace that skill. Instead, it should align optimize with it:

- optimize should treat profiling diagnosis as the default escalation level
- optimize should invoke the round-analysis skill when one round needs a deeper diagnosis artifact
- IR and compiler source analysis should remain consistent with that skill's evidence order

## Prompt And Guidance Contract

### Worker And Unsupervised Prompts

Optimize prompts should no longer condition the analysis-first workflow on `require_analysis=True`.

Instead, they should always instruct the agent to:

- choose an analysis level at the start of each round
- justify why that level is appropriate
- escalate only when the current level is not sufficient
- record the escalation reason in round-local artifacts
- avoid blind tiling, autotune, or launch-parameter search when the current evidence does not justify it

### Supervisor Prompt

The supervisor prompt should audit against the same contract.

It should reject or send back rounds that:

- skip directly to IR or compiler source without explaining why profiling was insufficient or already established
- use a pattern direction without a credible pattern-triage basis
- record deep analysis artifacts without any escalation rationale

The supervisor remains an audit role. It still should not perform open-ended optimize work itself.

### Temporary Workspace Guidance

Optimize guidance written into `AGENTS.md` or `CLAUDE.md` should describe the same layered default workflow and should no longer mention an optional strict-analysis mode.

## CLI And Model Shape

### Remove `--require-analysis`

Delete `--require-analysis` from:

- `optimize`
- `optimize-batch`

This is a direct removal, not a deprecation period and not a no-op compatibility shim.

Old invocations that still pass `--require-analysis` should fail through normal argument parsing because the flag no longer exists.

### Remove `require_analysis` Plumbing

Delete the `require_analysis` boolean from:

- optimize CLI option parsing
- optimize request and option models
- prompt builders
- optimize guidance generation
- optimize runtime plumbing
- tests and docs that mention the flag

The stricter analysis wording should move into the default optimize prompt and guidance path.

## Documentation Updates

Update user-facing docs so they match the new default:

- README examples should no longer mention `--require-analysis`
- optimize docs should describe layered analysis as normal behavior
- any older design docs that are explicitly about `--require-analysis` may remain historical, but new behavior docs should not present the flag as current behavior

## Implementation Shape

The implementation should remain layered primarily through documentation and prompt contracts:

- update optimize skill docs
- update optimize workflow references
- update prompt builders
- update temporary optimize guidance
- remove CLI/model/runtime plumbing for `require_analysis`
- adjust tests to reflect the new default

This design deliberately does not require a new runtime controller or a hard state machine.

## Testing

- remove parser tests that expect `--require-analysis`
- add parser tests proving the flag is no longer accepted
- update prompt tests so the layered analysis wording appears by default
- remove prompt and guidance tests that depend on `require_analysis=True`
- update runtime and model tests to remove the deleted field
- keep or add tests that assert compiler source remains a later escalation after profiler and IR evidence

## Expected Outcome

- optimize agents follow a clearer shallow-to-deep reasoning path
- profiling becomes the default diagnostic entrypoint when pattern triage is not enough
- IR and compiler source analysis are used more selectively and with clearer purpose
- the old strict-analysis flag disappears because its behavior is now the default optimize contract
