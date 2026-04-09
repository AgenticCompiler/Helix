# `optimize-status` Baseline Perf Selection Design

## Summary

- Refine top-level baseline perf discovery so `optimize-status` does not warn when a workspace contains both the original baseline perf file and optimized candidate perf files such as `opt_<name>_perf.txt`.
- Keep the warning for genuinely ambiguous baseline layouts.

## Behavior

- Continue looking only at top-level `*_perf.txt` files for baseline candidates.
- Prefer the unique candidate whose stem does not start with `opt_`.
- If there is exactly one top-level perf file, use it regardless of name.
- Warn with `found multiple baseline perf files` only when multiple non-`opt_` candidates remain or the selection is otherwise genuinely ambiguous.

## Rationale

- Optimize workspaces commonly keep both `kernel_perf.txt` and `opt_kernel_perf.txt`.
- The former is the intended baseline and the latter is an archived candidate artifact, not a second baseline.
- Choosing the unique non-`opt_` file matches existing naming conventions while preserving a warning for layouts that still need user attention.
