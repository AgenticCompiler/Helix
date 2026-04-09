# Optimize Batch Root Workspace Design

## Goal

Allow `optimize-batch --input <dir>` to run when `<dir>` itself is a single operator workspace, such as `--input .` inside a workspace containing `operator.py`.

## Current Problem

`optimize-batch` always assumes `--input` is a parent directory whose immediate child directories are operator workspaces. When the user points it at a workspace directory directly, the command scans only child directories, finds none, and exits with `No operator workspaces found under ...`.

## Desired Behavior

- If the input directory itself contains exactly one candidate operator file, `optimize-batch` should be able to run that directory as a single workspace.
- Existing batch behavior should remain unchanged when the input directory clearly contains child workspaces.
- Non-workspace child directories such as artifact folders should not block the single-workspace path.

## Approach

- Add a small workspace-discovery helper in the optimize batch layer.
- First inspect immediate child directories and classify whether any of them look like real operator workspaces.
- If no child directory looks like a real workspace, try resolving the input directory itself as one workspace.
- Otherwise preserve the current child-directory batch flow.

## Verification

- Add a CLI test proving `optimize-batch --input <workspace-dir>` runs that workspace directly.
- Keep the existing batch auto-detection tests green to confirm the parent-directory workflow still works.
