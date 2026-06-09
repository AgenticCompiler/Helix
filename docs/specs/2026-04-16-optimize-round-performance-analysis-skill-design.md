# Optimize Round Performance Analysis Skill Design

## Summary

Add a new repository-owned skill for round-level performance diagnosis inside `optimize` workflows. The skill should analyze one `opt-round-N/` at a time, require profiler evidence, collect IR when needed, and produce a standalone `opt-round-N/perf-analysis.md` that connects performance symptoms back to concrete problems in the current operator implementation.

The skill should strongly encourage spawning a subagent for deep analysis because profile and IR evidence can be large, but it should still allow the current agent to continue when the available context is sufficient.

## Goals

- Add a new skill dedicated to one optimize round of deep performance analysis.
- Make profiler evidence the default required input for the analysis workflow.
- Allow the skill to collect or reuse IR evidence when profiler evidence alone does not explain the observed behavior well enough.
- Produce a stable standalone artifact, `opt-round-N/perf-analysis.md`, instead of mixing the full analysis into `attempts.md` or `summary.md`.
- Help the agent identify signals such as scalar/vector/cube imbalance, frequent data movement, and weak pipeline overlap.
- Force the final diagnosis to land on problems in the current operator implementation, not just on surface-level profiler or IR observations.
- Reuse and extend existing scripts where lightweight automation helps extract signals more consistently.
- Keep the CLI thin by integrating the new behavior through skills, prompt guidance, and optional round metadata rather than new commands.

## Non-Goals

- Do not introduce a new optimize CLI subcommand just for round performance analysis.
- Do not move the full diagnosis logic into scripts or try to make the analysis fully deterministic.
- Do not make `perf-analysis.md` a required artifact for every completed round in the first iteration.
- Do not make the optimize supervisor responsible for running or interpreting this skill.
- Do not require IR collection for every round by default.
- Do not auto-write the analysis back into `attempts.md` or `summary.md`.

## Skill Identity

- Skill name: `triton-npu-analyze-round-performance`
- Primary trigger: an `optimize` round needs deeper explanation than benchmark numbers alone can provide
- Scope: exactly one round, usually the current `opt-round-N/`
- Primary consumer: the worker or unsupervised optimize agent, not the supervisor

The skill should be described as a round-level performance analyzer for Triton Ascend NPU operator optimization. The description should mention all of these likely triggers:

- unexpected scalar/vector/cube ratios
- suspiciously high data-movement cost
- suspected vectorization degradation
- suspected weak software pipeline overlap or concurrency
- benchmark regressions or underwhelming wins that need deeper diagnosis
- optimize rounds that already have or can collect `profile/` and optional `ir/` evidence

## User-Visible Behavior

### Default Workflow

1. Resolve the current round directory and the round-local operator file.
2. Confirm that the round has profiler evidence under `opt-round-N/profile/` or another explicitly supplied profile path.
3. If profiler evidence is missing, collect it first through the existing profile workflow.
4. Inspect the profiler output and extract structured signals relevant to round diagnosis.
5. Decide whether existing evidence is enough. If not, collect or reuse IR evidence under `opt-round-N/ir/`.
6. Inspect IR signals that may explain the profiler symptoms.
7. Compare against parent or baseline evidence when such evidence already exists and is useful, but do not block the analysis when comparable evidence is unavailable.
8. Write the final analysis to `opt-round-N/perf-analysis.md`.

### Subagent Guidance

The skill should strongly recommend spawning a subagent before the deep analysis phase because profile and IR evidence may consume substantial context and time. This should remain a recommendation rather than a hard requirement so the caller may continue in-process when context is still manageable.

### Evidence Rules

- Profiler evidence is required for the default workflow.
- IR evidence is optional by default but should be collected when profiler signals alone do not explain the likely operator-level problem well enough.
- The skill may reuse existing round-local evidence instead of recollecting it.
- The skill may compare the current round against parent or baseline evidence when those artifacts already exist or can be resolved cheaply.
- Missing comparison evidence should be reported as a gap, not treated as a failure.

## `perf-analysis.md` Contract

The main output of the skill is `opt-round-N/perf-analysis.md`.

The file should use a stable structure so later rounds and human readers can quickly scan it:

1. `# Round Performance Analysis`
2. `## Executive Summary`
3. `## Profile Signals`
4. `## IR Signals`
5. `## Diagnosis`
6. `## Operator Implementation Issues`
7. `## Optimization Suggestions`
8. `## Evidence Gaps`

### Content Expectations

#### Round Performance Analysis

- identify the round directory
- identify the analyzed operator file
- identify any parent or baseline evidence used for comparison
- list the concrete profile and IR artifact paths used
- state whether each evidence source was reused or newly collected

#### Executive Summary

- provide two to five high-value conclusions
- explicitly distinguish observed facts from inference when the distinction matters
- keep each conclusion tied to a likely operator implementation issue

#### Profile Signals

- summarize scalar, vector, and cube time or ratio signals
- highlight the hottest operators and core types
- note whether the observed ratios look mismatched for the intended operator style
- surface data-movement-heavy hotspots when present

#### IR Signals

- identify the most relevant stages or stage transitions
- call out suspicious vector, copy, sync, load/store, or allocation patterns
- mention possible signs of weak pipeline overlap or concurrency when visible
- cite concrete stage names or artifact paths for every nontrivial claim

#### Diagnosis

- connect profiler signals to IR signals where possible
- explain why those signals imply a likely problem in the current implementation
- keep hypotheses explicit instead of overstating certainty

#### Operator Implementation Issues

- state the concrete current-implementation problems that best explain the evidence
- prefer operator-code causes such as poor access patterns, overly heavy index arithmetic, conservative masking, broken vector-friendly structure, or insufficient load/compute overlap
- do not stop at generic statements like "vector ratio is low"

#### Optimization Suggestions

- provide implementation-focused suggestions for the current operator
- tie each suggestion back to a diagnosed issue
- keep this as advice, not a full next-round execution plan

#### Evidence Gaps

- record missing IR, missing comparison evidence, or any other limitation that weakens confidence
- explicitly note when a conclusion remains heuristic

## Script Extension Strategy

The new skill should prefer extending existing scripts rather than creating a separate end-to-end analyzer.

### Profile Script Extensions

Extend `skills/triton-npu-profile-operator/scripts/profile_summary.py` with round-analysis-friendly outputs.

Add lightweight support for:

- aggregating runtime by `Core Type`
- reporting scalar, vector, and cube totals, counts, and ratios
- surfacing likely data-movement-heavy hotspots through simple operator-name or operator-type heuristics
- returning either Markdown or JSON output so the skill can consume structured signals without losing the current human-readable report

The profile script should not try to assert final root causes such as "vectorization degraded" on its own. It should extract signals, not replace the diagnosis step.

### IR Script Extensions

Extend `skills/triton-npu-analyze-ir/scripts/inspect_ir.py` with a performance-oriented summary mode.

Add a new subcommand or equivalent helper that can surface heuristic signals such as:

- unexpectedly low vector-op presence
- heavy copy, DMA, load/store, or allocation footprints
- frequent wait, barrier, or flag-setting patterns
- stage transitions where vector, copy, or sync-like patterns change sharply
- possible signs of insufficient pipeline overlap

This IR helper should support Markdown or JSON output and should remain heuristic. It should not turn performance diagnosis into a fully scripted pass/fail judgment.

## Optimize Integration

### Prompt Guidance

Optimize worker and unsupervised prompts should mention the new skill as the preferred workflow when a round needs deeper performance diagnosis. The prompt text should tell the agent to use the skill when:

- benchmark behavior is not self-explanatory
- profiler signals and operator expectations do not match
- data movement appears abnormally expensive
- pipeline or concurrency concerns need IR-backed confirmation
- strict analysis mode needs stronger evidence before further optimization

The prompt should also mention that the formal output of this deep analysis is `opt-round-N/perf-analysis.md`.

### Artifact Contract

Keep `perf-analysis.md` optional in the first iteration.

Add optional round metadata rather than new required round-state fields. Recommended optional fields:

- `perf_analysis_path`
- `analysis_comparison_sources`

This lets the optimize workflow point to the analysis artifact when present without breaking existing round validation.

### Optimize Check Behavior

`triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round` should not require `perf-analysis.md` for every round in the first iteration.

If `round-state.json` declares `perf_analysis_path`, the checker may verify that the declared file exists. The checker should not attempt semantic validation of the analysis contents in this change.

## Implementation Outline

1. Add the new `skills/triton-npu-analyze-round-performance/` directory and write `SKILL.md`.
2. Update the skill text to:
   - require or collect profiler evidence first
   - optionally collect IR evidence when needed
   - recommend subagent-based deep analysis
   - write `opt-round-N/perf-analysis.md`
3. Extend `profile_summary.py` with core-type and data-movement signal extraction plus JSON output.
4. Extend `inspect_ir.py` with a performance-oriented signal summary mode plus JSON output.
5. Update optimize prompt construction to mention the new skill at the right moments.
6. Add optional round-state plumbing for `perf_analysis_path` and `analysis_comparison_sources`.
7. Teach optimize checks to validate the declared analysis path only when it is present.

## Testing

- add or update tests for profile core-type aggregation
- add or update tests for data-movement hotspot summaries
- add or update tests for JSON output from the profile summary helper
- add or update tests for the new IR performance-signal entrypoint
- add prompt tests that pin the new optimize guidance
- add round-contract tests for optional `perf_analysis_path` parsing
- add triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round tests that validate the declared analysis path only when present
- run skill validation for the new skill
- run the standard repository verification commands after implementation

## Open Questions Deferred

- whether `perf-analysis.md` should later become a required artifact for some optimize modes
- whether future versions should add stronger round-state semantics around analysis confidence
- whether some profile and IR heuristics deserve promotion into reusable optimization-pattern references
