# `optimize-status` Markdown Table Implementation Plan

**Goal:** Add a Markdown table output mode to `optimize-status` that reports `Geomean speedup` and `Total speedup` for workspaces with optimize-session data.

**Architecture:** Keep the feature in the CLI and render layers. Extend the optimize-status parser with `--format`, thread the selection through the command handler, and add a dedicated Markdown renderer beside the existing text renderer.

**Tech Stack:** Python, unittest, Markdown docs

## Steps

1. Add a failing parser test for `optimize-status --format markdown`.
2. Add a failing render test for Markdown output that excludes `no-session` rows and uses `-` for missing speedup values.
3. Add a failing CLI regression test that checks the rendered Markdown table.
4. Implement the new format option and Markdown renderer.
5. Update README and optimize-status docs with the new output mode.
6. Run focused tests, then lint, type checks, and the full unittest suite.
