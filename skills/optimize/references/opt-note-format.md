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

## Example

```md
## Round 3
Parent: round-1
Theme: reorder independent loads before dependent reads
Result: correctness passed; latency improved from 1.82 ms to 1.57 ms versus parent
Best status: current best
Summary: [opt-round-3/summary.md](opt-round-3/summary.md)
Attempts: [opt-round-3/attempts.md](opt-round-3/attempts.md)
```

## Writing Guidance

- Prefer user-visible outcomes over implementation trivia.
- Keep the note readable as a project history log.
- Put detailed reasoning, code snippets, and deeper analysis in the per-round summary instead of the top-level note.
