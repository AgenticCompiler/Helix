# Batch Optimize Prefixed Show Output

## Summary

- Add `--show-output` support to `optimize-batch`.
- Keep the batch flow non-interactive.
- Stream each workspace's live output into the current terminal with a stable text prefix like `[workspace-name] `.

## User-visible behavior

- `optimize-batch` accepts `--show-output`.
- When `--show-output` is enabled, each workspace still runs concurrently up to `--max-concurrency`.
- Live output lines from each workspace are written to the current terminal with a workspace prefix:
  - `[matmul] round 1: running tests`
  - `[layernorm] benchmark passed`
- Batch summaries remain compact and are still printed after all workspaces finish.
- The feature is text-only streaming. It does not create terminal panes, tmux splits, or a dashboard UI.

## Design notes

- Reuse the existing single-workspace optimize request flow instead of inventing a second backend path.
- Keep `show_output=True` on the per-workspace optimize request so the underlying runner still uses its streaming mode.
- Wrap the target stdout stream with a small line-prefixing writer so concurrent workspace output is readable.
- Use one shared lock around terminal writes so lines from different workspaces do not interleave at the character level.
- Route verbose batch optimize output through the same prefixed stream when `--show-output` is active so agent-launch and workspace-preparation messages stay attributable.
