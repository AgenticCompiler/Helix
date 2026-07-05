# Claude Plugin Fixed Optimize Modes Implementation Plan

**Goal:** Render fixed optimize mode guidance into the generated Claude optimize
plugin agent.

**Architecture:** Keep constants in the plugin builder and add a compact
generated section to the agent markdown. Verify through builder tests rather
than generated fixture files.

**Verification:** Focused plugin builder tests, then repository ruff, pyright,
and pytest.

## Steps

1. Add a failing assertion in `tests/test_claude_optimize_plugin.py` that the
   generated agent mentions `test-mode: differential` and
   `bench-mode: torch-npu-profiler`.
2. Update `scripts/build-claude-optimize-plugin.py` to render a "Fixed Optimize
   Modes" section using plugin-local constants.
3. Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`.
4. Run the standard repository verification commands.
