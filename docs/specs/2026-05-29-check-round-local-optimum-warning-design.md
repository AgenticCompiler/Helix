title: Check-Round Local Optimum Warning
created: 2026-05-29
summary: Detect recent optimize rounds whose baseline-relative gains have nearly flattened out and surface that state as a pass-with-warning signal from check-round.
---

# Check-Round Local Optimum Warning Design

## Summary

- Extend `check-round` so that, after the existing round contract passes, it inspects recent comparable rounds for a possible local-optimum trend.
- Keep the round result as `decision="pass"` and surface local-optimum detection only as warning text in `issues` and the human-facing `summary`.
- Reuse existing `perf.txt` parsing and round metric-source semantics instead of adding new required fields to `round-state.json`.
- Make both the recent-round window size and the marginal-gain threshold configurable through environment variables, with stable defaults.

## Problem

The current optimize round check answers one question well: "is this round valid enough to continue or stop?"

It does not answer a second question that matters in longer optimize sessions: "are the last few rounds still producing meaningful baseline-relative gains, or has the current optimization direction flattened out enough that we may be stuck in a local optimum?"

That gap has two practical costs:

- agents may keep spending rounds on tiny geomean changes without being reminded that the current search direction is likely exhausted
- the CLI-owned checked round flow already runs `check-round` after every round, but that one mandatory checkpoint does not currently surface trend-level performance guidance

We want a lightweight signal that helps the agent notice a possible local optimum without turning "small improvement" into a hard failure. The signal should also tell the agent what to do next: review earlier rounds and consider branching again from a round before the flat sequence instead of continuing to stack changes on the current direction.

## Goals

- Make `check-round` the single shared place that can surface a "possible local optimum" signal.
- Keep local-optimum detection advisory only: a technically valid round must still pass.
- Avoid duplicating compare-perf metrics into `round-state.json`.
- Reuse existing baseline/round perf parsing so local-optimum detection stays aligned with status and verify semantics.
- Allow users to tune the round window and marginal-gain threshold through environment variables.

## Non-Goals

- Do not change `check-round` decision semantics from `pass` to `revise-required` or `hard-fail` because of local-optimum detection.
- Do not add new required `round-state.json` fields for geomean speedup, avg improvement, or local-optimum state.
- Do not make the runtime gate stop automatically when a local-optimum warning appears.
- Do not reinterpret local-optimum detection as "the current round is bad" or "the session must stop now."
- Do not add new CLI flags for this first version; environment variables are sufficient.

## Design

### 1. Add a local-optimum analysis helper under the optimize-check skill

Create a small helper module under `skills/triton-npu-optimize-check/scripts/` that:

- discovers numeric `opt-round-*` directories in workspace order
- selects a recent window ending at the current round
- loads the canonical round perf artifact for each selected round
- computes one baseline-relative geomean speedup value per round using the shared baseline perf artifact and the round's recorded `effective_metric_source`
- returns either:
  - no warning
  - one warning string describing a possible local optimum
  - configuration warnings when the local-optimum environment variables are invalid and defaults are used

This keeps the behavior next to the existing round contract logic instead of splitting the meaning of `check-round` across skill code and CLI runtime code.

### 2. Use existing perf artifacts and baseline-relative round scores, not direct round-to-round file comparisons

Stagnation detection should not extend `ROUND_STATE_REQUIRED_FIELDS`.

Instead, it should reuse the same evidence that already exists today:

- baseline perf artifact resolved from `baseline/state.json`
- round perf artifact resolved from `round-state.json` and round artifact inspection
- round `effective_metric_source` (`kernel`, `total-op`, or fallback-compatible `mixed`)
- existing bench perf parsers already used by optimize status and verify flows

The important comparison rule is:

- do compute a baseline-relative score for each recent round
- do compare those score values across rounds
- do not compare `opt-round-N/perf.txt` directly against `opt-round-(N-1)/perf.txt` as if the round contract were parent-relative

In other words, the recent-round trend is:

```text
round_3_vs_baseline
round_4_vs_baseline
round_5_vs_baseline
```

and the local-optimum check looks at how little those baseline-relative scores move from one round to the next.

This keeps the logic aligned with the optimize workflow, where baseline remains the canonical comparison target even if a round also performs local parent-focused reasoning in its notes.

The round-local `perf.txt` artifact itself still comes from `run-bench`. The local-optimum helper must derive a baseline-relative score by pairing that round artifact with the shared baseline perf artifact under the same metric-source rules that `compare-perf` uses for optimize conclusions.

### 3. Local-optimum warning algorithm

The first version should intentionally stay simple and conservative.

Definitions:

- **window size**: how many most-recent comparable rounds to inspect, including the current round
- **meaningful gain threshold**: the minimum increase in geomean speedup that counts as a meaningful step forward

Algorithm:

1. Resolve the configured window size `N`.
2. Collect up to `N` recent rounds ending at the current round, ordered oldest to newest.
3. For each round, compute its baseline-relative geomean speedup using the round's effective metric source.
4. If fewer than `N` comparable rounds are available, skip local-optimum detection silently.
5. If the rounds in the candidate window do not share the same normalized metric basis (`kernel`, `total-op`, or `auto` for mixed fallback), skip local-optimum detection silently.
6. Compute adjacent gains between neighboring rounds in the window:

```text
gain_i = geomean_speedup(round_i) - geomean_speedup(round_{i-1})
```

7. If every adjacent gain in the window is less than or equal to the configured threshold, emit one local-optimum warning.

Example with `N=3` and threshold `0.02`:

- round-3: `1.19x`
- round-4: `1.20x`
- round-5: `1.21x`

Adjacent gains are `+0.01x` and `+0.01x`, so `check-round` warns that recent baseline-relative gains are nearly flat.

Example that should not warn:

- round-3: `1.19x`
- round-4: `1.21x`
- round-5: `1.29x`

Because at least one adjacent gain is materially larger than the threshold, the recent baseline-relative trend is still moving.

### 4. Environment variables

Add two advisory environment variables:

- `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW`
  - integer
  - default: `3`
  - meaning: how many most-recent comparable rounds to inspect, including the current round
  - minimum effective value: `2`

- `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN`
  - float
  - default: `0.02`
  - meaning: the maximum adjacent geomean-speedup gain that still counts as "almost no improvement"
  - unit: absolute speedup delta, not percent

Examples:

- `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW=4`
- `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN=0.01`

Configuration parsing rules:

- unset variables use defaults
- invalid integers/floats do not fail the round
- invalid values add a pass-time warning such as:
  - `invalid TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW='abc'; using default 3`
  - `invalid TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN='-1'; using default 0.02`

This preserves explicit diagnostics without turning local configuration mistakes into round failures.

### 5. Integrate after the existing round contract passes

`check_round()` in `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py` should keep its current validation order:

1. artifacts
2. round-state loading
3. correctness/benchmark status
4. baseline validity
5. semantic checks
6. Triton kernel continuity
7. optional `.pt` cleanup

Only after those checks pass should it run local-optimum analysis and append any local-optimum/config warnings to the pass result.

This preserves the current meaning of `check-round`: first decide whether the round is valid, then attach advisory context.

### 6. Warning format

The local-optimum warning should be specific and compact.

Recommended format:

```text
recent rounds show only marginal baseline-relative geomean speedup gains on the same metric basis (round-3 -> round-5: +0.01x, +0.01x); optimization may be stagnating in the current direction. Review earlier rounds and consider resuming from a round before this flat sequence to explore a different optimization path.
```

Important properties:

- explicitly says this is a recent-trend warning, not a failure
- includes the covered round range
- includes the observed adjacent gains
- mentions that the metric basis is consistent, which explains why the comparison was considered safe
- suggests a concrete next move: look back to an earlier round before the stagnant sequence and try a different branch of changes

The wording should reflect the practical cause we are trying to catch:

- some earlier round likely introduced a direction that looked promising
- later rounds kept building on top of that direction
- the baseline-relative gains then flattened out

So the warning should nudge the agent away from "keep polishing the same branch" and toward "revisit the history, identify a round before the flat sequence, and explore a different optimization idea from there."

### 7. Runtime behavior by round mode

No runtime stop/continue semantics change in this feature.

Effects by layer:

- `check-round` returns `decision="pass"` with warning text
- `src/triton_agent/optimize/execution.py` continues to interpret the round as passed
- continuous mode keeps the warning in the direct `check-round` result seen by the worker, and the continuous prompt explicitly tells the agent to revisit earlier rounds when that warning appears
- checked mode must preserve pass-time warning text when the session continues only because `min_rounds` is not yet satisfied, so the next worker prompt still receives the local-optimum signal
- supervised mode must carry the same CLI technical summary into both the supervisor prompt and the later worker continuation prompt
- the agent remains free to continue or stop based on the broader evidence

This keeps the new signal aligned with the user's intended semantics: `pass + warning`, not automatic stop and not repair-required.

## File-Level Changes

- `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`
  - invoke local-optimum analysis after the existing pass path succeeds
  - append local-optimum/config warnings to pass issues

- `skills/triton-npu-optimize-check/scripts/`
  - add one focused helper module for:
    - environment variable parsing
    - recent round discovery
    - baseline-relative comparable round speedup calculation
    - local-optimum warning generation

- `src/triton_agent/optimize/execution.py`
  - preserve pass-time warnings when `GateDecision.PASS_CONTINUE` is caused only by the minimum-round requirement
  - pass those warnings through checked and supervised continuation summaries

- `src/triton_agent/optimize/prompts.py`
  - tell continuous-mode workers how to react when `check-round` warns about a possible local optimum

- `tests/test_optimize_checks.py`
  - add focused local-optimum warning coverage

- `tests/test_optimize_runtime.py`
  - cover checked-mode warning propagation during `PASS_CONTINUE`

- `tests/test_cli.py`
  - cover continuous-mode prompt guidance for local-optimum warnings

No contract JSON change is required because no new required state fields are added.

## Test Plan

Add focused tests that cover:

- no warning when there are fewer than the configured number of comparable rounds
- warning when the last `N` rounds all have adjacent geomean gains less than or equal to threshold
- no warning when at least one adjacent gain exceeds threshold
- no warning when the recent rounds use inconsistent metric bases
- pass-time warning plus default fallback when either local-optimum environment variable is invalid
- no warning when the implementation would need a direct round-to-round perf comparison rather than a baseline-relative score sequence
- checked-mode continuation summaries preserve advisory warnings while still adding the minimum-round reminder
- continuous prompt tells the worker to revisit earlier rounds when a local-optimum warning appears

## Risks And Tradeoffs

- Geomean-based local-optimum detection may miss workloads where one critical latency regresses while geomean barely moves. That is acceptable for this feature because the goal is round-trend guidance, not best-round selection replacement.
- Mixed metric bases across rounds can produce misleading trend comparisons. Skipping those windows is safer than warning on questionable math.
- Environment variables are less discoverable than CLI flags, but they keep the first version small and easy to tune in scripted environments.

## Alternatives Considered

### Add speedup fields to `round-state.json`

Rejected because it duplicates information already derivable from canonical perf artifacts and increases drift risk between state and benchmark evidence.

### Put local-optimum detection only in `execution.py`

Rejected because checked-mode runtime and direct/manual `check-round` execution would then disagree about whether a local-optimum signal exists.

### Make local-optimum detection a stop signal

Rejected because the user explicitly wants advisory behavior only. A local-optimum warning should help the agent reason, not remove discretion.
