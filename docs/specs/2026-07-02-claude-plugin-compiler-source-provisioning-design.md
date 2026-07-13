# Claude Plugin Compiler Source Provisioning Design

## Goal

Give the standalone Claude optimize plugin the same compiler-source evidence path
as CLI-launched optimize runs, without requiring the user to run the
`helix` CLI lifecycle first.

## User-Visible Semantics

- The generated Claude optimize plugin prepares AscendNPU-IR on the first
  optimize-agent session start.
- The default checkout remains
  `~/.helix/compiler-sources/AscendNPU-IR/`, with
  `HELIX_COMPILER_SOURCE_CACHE_DIR` as the cache-root override.
- If the checkout is missing, the plugin runs a shallow `git clone --depth 1`.
- If the checkout exists, the plugin reuses it and only reads the current commit.
- The plugin does not run `git fetch`, `git pull`, or automatic refresh.
- The plugin reports the local compiler source path and commit through Claude
  `SessionStart` additional context.
- The plugin hook guard allows reads from the prepared compiler source checkout
  while still treating it as read-only evidence.
- If provisioning fails, the hook fails open for the Claude session and reports a
  short diagnostic in additional context instead of blocking unrelated optimize
  work.

## Design

Move the compiler-source provisioning logic into `hook_runtime` so the built
plugin can package it without importing `helix`. Keep
`helix.optimize.compiler_source` as a thin facade so existing CLI callers
and tests keep their current import path.

`hooks/claude_plugin/session_start.py` continues to delegate lifecycle work to
`state_bootstrap.py`. During optimize-agent session start, bootstrap should:

1. prepare or validate optimize workflow state as it does today;
2. prepare compiler source in auto mode;
3. append a compiler-source context block when provisioning succeeds or fails.

`hooks/claude_plugin/pretooluse_guard.py` should build its policy from the
workspace plus the existing compiler source checkout if one is available. The
guard should not clone during PreToolUse; cloning belongs to SessionStart so the
agent receives the path and commit before it needs the evidence.

Skill prose should refer to the CLI/plugin-provided checkout rather than a
CLI-only checkout.

## Testing

- Unit-test `hook_runtime.optimize.compiler_source` clone/reuse behavior with a
  fake git runner.
- Unit-test plugin bootstrap with a fake git runner so no network is required.
- Unit-test plugin policy generation to allow reads under an existing compiler
  source checkout.
- Update plugin subprocess tests to disable compiler-source provisioning where
  they only exercise workflow-state behavior.
- Keep existing CLI compiler-source tests passing through the compatibility
  facade.
