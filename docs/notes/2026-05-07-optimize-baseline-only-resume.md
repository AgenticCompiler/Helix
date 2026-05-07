# Optimize Baseline-Only Resume

## Summary

- A valid `baseline/` directory by itself does not make an optimize workspace resumable.
- `optimize --resume auto` now treats a baseline-only workspace as a fresh optimize start instead of a partial optimize session.
- If `baseline/` is present but malformed, `resume auto` still fails with the concrete baseline issue.

