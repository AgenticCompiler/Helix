## Summary

- Update the optimize artifact contract so per-round evidence directories are explicit, not implicit.

## User-Visible Behavior

- Keep `opt-round-N/` as the root directory for one optimization round.
- Continue to require a round-local performance artifact such as `perf.txt` so benchmark claims remain easy to compare.
- Formalize `profile/` as the preferred round-local directory for profiler evidence captured during that round.
- Formalize `ir/` as the preferred round-local directory for archived Triton or Bisheng IR captured during that round.
- Keep `triton-agent-logs/` as the shared workspace log root instead of splitting optimize archives into a separate top-level directory.

## Implementation Notes

- Update the optimize skill artifact reference to show `profile/` and `ir/` in the recommended round layout.
- Update the optimize skill text and workflow guidance so agents preserve profiler and IR evidence inside the current round directory when they collect it.
- Keep `optimize-status` unchanged for now; it should continue to rely on `perf.txt` or `*_perf.txt` until a later change extends status reporting for richer evidence directories.
