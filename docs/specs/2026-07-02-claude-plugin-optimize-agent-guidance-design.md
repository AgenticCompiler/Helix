# Claude Plugin Optimize Agent Guidance Design

## Goal

Make the generated `helix-optimize` Claude plugin agent self-contained enough for optimize sessions by embedding the stable workflow guidance that was previously carried by workspace memory files or launch prompts.

This change applies only to the optimize agent. The convert agent remains convert-specific and must not inherit optimize round, baseline, or performance-analysis rules.

## User-Visible Semantics

The generated `triton-optimizer` plugin should still expose `helix-optimize` and `helix-convert`.

When users choose `helix-optimize`, the agent definition should directly remind Claude to:

- read files cautiously and avoid speculative context gathering
- use staged workspace skills as the workflow source of truth
- follow user instructions and invocation-specific prompts
- treat `baseline/` as the canonical optimize baseline
- use `compare-perf` as the authoritative source for performance summaries
- follow the optimize state lifecycle through `ascend-npu-optimize-state` subcommands
- choose and record an analysis level before code edits
- escalate through pattern triage, profiling diagnosis, IR attribution, and compiler-source escalation
- avoid blind tiling or launch-parameter search
- consider the current high-priority Triton pattern reminders

The agent definition should not claim that the CLI will inject continuation context, because standalone plugin use may not run through the CLI. Continuation context should come from the user prompt, SessionStart context, workflow state, and existing round artifacts.

## Implementation

Keep the builder script as the rendering owner for generated plugin agent markdown. Add a focused optimize-guidance renderer in `scripts/build-claude-optimize-plugin.py` so tests can assert the generated agent text without depending on a built plugin artifact.

Use the current `ascend-npu-optimize-state` skill and subcommand names. Do not reintroduce old standalone skill names such as `triton-npu-optimize-submit-baseline`, `triton-npu-optimize-start-round`, or `triton-npu-optimize-submit-round`.

Use static high-priority reminders for the plugin agent. They should match current Triton kernel optimize defaults and remain concise:

- `a5-force-simt-only-discrete-access`
- `autotune`
- `grid-flatten-and-ub-buffering`

## Tests

Update plugin builder tests to verify that `agents/helix-optimize.md` includes representative workflow guidance:

- cautious file reading
- staged workspace skills as source of truth
- `baseline/` canonical baseline
- `compare-perf` authority
- analysis ladder lines
- the three high-priority pattern reminder names
- no stale `triton-npu-optimize-submit-*` or `triton-npu-optimize-start-round` names

Existing convert-agent assertions should continue proving optimize-only guidance is not copied into the convert agent.
