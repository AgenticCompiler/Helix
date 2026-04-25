## Summary

- Require optimize rounds to keep a running attempt log, not only a final round summary.

## User-Visible Behavior

- Each `opt-round-N/` directory should contain an `attempts.md` file that records the round's incremental trials.
- The final `summary.md` remains the concise conclusion, while `attempts.md` captures the sequence of tried ideas, failures, fixes, and measurements.

## Implementation Notes

- Update the optimize workflow to record every meaningful attempt within a round.
- Keep `opt-note.md` as the top-level round summary log, but move detailed within-round iteration history into `opt-round-N/attempts.md`.
