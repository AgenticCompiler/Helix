# `optimize-status` Single-Workspace Input Implementation Plan

**Goal:** Make `optimize-status` inspect the input directory itself when it is already an optimize workspace, while preserving batch-root scans for parent directories.

**Architecture:** Keep the change small by adding a reusable optimize-artifact detection helper in `src/triton_agent/optimize/status.py` and branching in the command handler based on that helper.

**Tech Stack:** Python, unittest

## Steps

1. Add a failing CLI regression test for `optimize-status --input <workspace-dir>` where the directory itself contains optimize artifacts.
2. Add a failing status-helper test for optimize-artifact detection on a single workspace directory.
3. Implement the helper and update the command handler to inspect the input directory directly when appropriate.
4. Update optimize-status docs so they describe both single-workspace and batch-root input behavior.
5. Run focused tests, then lint, type checks, and the full unittest suite.
