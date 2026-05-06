# Optimize Pattern Recording Design

## Summary

Clarify where optimize rounds record pattern selection without adding a new artifact or extending `round-state.json`.

Record the evolving pattern choice in `opt-round-N/attempts.md`, and record the final chosen pattern direction in `opt-round-N/summary.md`.

## Problem

The current optimize docs require rounds to record hypotheses, evidence, and outcomes, but they do not say explicitly where a selected optimize pattern should live when the agent:

- chooses one pattern during `pattern triage`
- compares multiple candidate patterns
- pivots away from an earlier pattern after deeper evidence

That ambiguity risks inconsistent round logs:

- some rounds may bury the pattern choice in free-form prose
- some may put it only in `summary.md`
- some may omit the discarded alternatives and pivot rationale entirely

## Goals

- Keep pattern-choice recording inside the existing round artifacts.
- Make `attempts.md` the authoritative place for the pattern decision trail.
- Make `summary.md` capture the final pattern direction succinctly.
- Avoid new top-level files and avoid `round-state.json` changes.

## Non-Goals

- Do not add machine-readable pattern fields.
- Do not require every round to map to a named library pattern.
- Do not force a separate pattern-routing artifact.

## Decision

### `attempts.md`

`opt-round-N/attempts.md` should record the process-level pattern trail, including:

- candidate patterns considered at the start of the round
- the selected pattern, when one is chosen
- why the selected pattern looked plausible
- rejected alternatives when they materially affected the direction
- later pivots, including when deeper profile or IR evidence weakens or overturns the earlier pattern choice

This keeps the chronological decision trail in the artifact that already owns evolving round reasoning.

### `summary.md`

`opt-round-N/summary.md` should record the conclusion-level pattern outcome, including:

- the final selected pattern direction when a named pattern guided the round
- whether the round pivoted away from an earlier pattern choice
- whether the pattern meaningfully contributed to the final result

This keeps the summary readable while still preserving the final reusable takeaway.

## Rationale

`attempts.md` is already the round's chronological log, so it is the natural place for the agent's changing pattern choice.

`summary.md` is already the round conclusion, so it should only retain the final pattern outcome rather than the full search history.

This split matches the broader optimize artifact model:

- process and pivots belong in round-local attempt logs
- conclusions and reusable takeaways belong in round summaries

## Test Impact

Update optimize contract tests so they require docs to state that:

- `attempts.md` records selected patterns and pivot rationale
- `summary.md` records the final selected pattern direction
