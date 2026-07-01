# opt-note.md Format

## Purpose

`opt-note.md` is the top-level running log for the optimization session's completed round records and final outcome summary.

Use it to:

- show which rounds were completed
- show which parent each round used
- highlight the main optimization point
- record the measured outcome
- link to the round summary for details
- point readers to the round-level attempt log when they need the full trial history

## Entry Rules

- Append one section per completed round.
- Keep entries concise.
- Link to the corresponding `opt-round-N/summary.md`.
- Link to the corresponding `opt-round-N/attempts.md`.
- Mention whether the round is now the current best candidate.
- Record both the parent and the measured improvement or regression status.
- Mention the optimization point in a way that reflects why the round was pursued.
- Do not put session-start diagnosis, tentative bottleneck narrative, or other pre-round analysis above the round history; keep that reasoning in round-local artifacts such as `opt-round-N/attempts.md`.
- Keep exactly one `## Overall Summary` section at the end of the file.
- Refresh the existing overall summary when the optimize session continues instead of appending a second final section.

## Required Template

```md
## Round N
Parent: round-M
Theme: <short optimization theme>
Result: <brief correctness and performance outcome>
Best status: <current best / validated branch / not promoted>
Summary: [opt-round-N/summary.md](opt-round-N/summary.md)
Attempts: [opt-round-N/attempts.md](opt-round-N/attempts.md)
```

After the round history, end the file with:

```md
## Overall Summary
Final best round: round-N
Avg improvement: <value or unknown>
Geomean speedup: <value or unknown>
Validated branches: <comma-separated round names or none>
Outcome: <plain-English optimization result>
Key optimization points:
  1. <optimization point>: <improvement> (round N)
  2. ...
```

## Example

```md
## Round 3
Parent: round-1
Theme: reorder independent loads before dependent reads
Result: correctness passed; latency improved from 1.82 ms to 1.57 ms versus parent
Best status: current best
Summary: [opt-round-3/summary.md](opt-round-3/summary.md)
Attempts: [opt-round-3/attempts.md](opt-round-3/attempts.md)

## Overall Summary
Final best round: round-3
Avg improvement: +13.7%
Geomean speedup: 1.16x
Validated branches: round-1
Outcome: round-3 is the fastest validated candidate and preserves correctness.
Key optimization points:
  1. reorder independent loads before dependent reads: -13.7% latency (round-3)
  2. apply vectorized mask load: -5.2% latency (round-2)
Next step: profile round-3 if more latency reduction is needed.
```

## Writing Guidance

- Prefer user-visible outcomes over implementation trivia.
- Keep the note readable as a project history log.
- Put detailed reasoning, code snippets, and deeper analysis in the per-round summary instead of the top-level note.
- Put initial hypotheses, evolving reasoning, and diagnosis pivots in `opt-round-N/attempts.md`, `summary.md`, or `perf-analysis.md`, not in the top-level note.
- Keep the top-level note concise, but make sure a reader can still tell why the chosen round direction was reasonable.
- Use `Key optimization points` to list the main optimization actions and their impact, with the round where each was applied. This checklist is consumed by automated report generation.
- Use `Geomean speedup` as the headline metric for the final best round.
- Use `Validated branches` to list non-best rounds that are still worth revisiting later.
- Keep `Outcome` and `Next step` short enough that a reader can understand the session result in one screen.
