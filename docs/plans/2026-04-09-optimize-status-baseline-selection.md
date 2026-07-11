# `optimize-status` Baseline Perf Selection Implementation Plan

**Goal:** Stop `optimize-status` from warning on normal baseline-plus-candidate perf layouts while preserving warnings for truly ambiguous baseline perf discovery.

**Architecture:** Keep the change local to `src/helix/optimize/status.py` by refining baseline candidate selection. Lock the behavior with focused unit tests in the optimize-status and CLI suites.

**Tech Stack:** Python, unittest

## Steps

1. Add a failing optimize-status unit test for a workspace containing both `kernel_perf.txt` and `opt_kernel_perf.txt`.
2. Add a failing CLI regression test that confirms the warning does not appear for the same layout.
3. Update baseline selection to prefer the unique non-`opt_` top-level perf file.
4. Re-run focused tests until they pass.
5. Run lint, type checks, and the full unittest suite.
