# Batch Workspace Discovery Sharing Design

## Goal

Keep `gen-eval-batch` and `optimize-batch` aligned on workspace selection semantics while removing the most obvious duplicated batch helper code.

## Problems

- `optimize-batch` now accepts `--input <workspace-dir>` when the input directory itself is a single operator workspace, but `gen-eval-batch` still only scans immediate child directories.
- The two batch modules duplicate the same low-level logic for prefixed output streams and operator-file discovery, which increases drift risk.

## Desired Behavior

- `gen-eval-batch --input .` should work the same way as `optimize-batch --input .` when the current directory is a single operator workspace.
- Non-workspace child directories should not prevent the single-workspace fallback.
- Parent-directory batch behavior should remain unchanged for both commands.

## Approach

- Add a small shared batch helper module for:
  - line-prefixed output streaming
  - operator-candidate filtering
  - workspace discovery that prefers real child workspaces but falls back to the root directory when appropriate
- Keep command-specific request construction, execution, failure summaries, and rendering in their existing modules.

## Verification

- Add failing CLI tests for `gen-eval-batch --input <workspace-dir>` and the non-workspace child-directory case.
- Keep the existing `optimize-batch` and `gen-eval-batch` batch tests green.
