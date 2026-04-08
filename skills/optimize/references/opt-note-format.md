# opt-note.md Format

## Purpose

`opt-note.md` is the top-level running log for the optimization session.

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
Baseline mean: <value or unknown>
Best mean: <value or unknown>
Avg improvement: <value or unknown>
Validated branches: <comma-separated round names or none>
Outcome: <plain-English optimization result>
Next step: <plain-English follow-up or none>
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
Baseline mean: 1.82 ms
Best mean: 1.57 ms
Avg improvement: +13.7%
Validated branches: round-1
Outcome: round-3 is the fastest validated candidate and preserves correctness.
Next step: profile round-3 if more latency reduction is needed.
```

## Writing Guidance

- Prefer user-visible outcomes over implementation trivia.
- Keep the note readable as a project history log.
- Put detailed reasoning, code snippets, and deeper analysis in the per-round summary instead of the top-level note.
- Use `Validated branches` to list non-best rounds that are still worth revisiting later.
- Keep `Outcome` and `Next step` short enough that a reader can understand the session result in one screen.
