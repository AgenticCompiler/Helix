## Summary

- Add a bundled helper script under `skills/run-validation/scripts/` so skills can invoke the current project CLI without requiring the `triton-agent` console script to be installed.

## User-Visible Behavior

- Skills should use a bundled script to run project subcommands from the current repository checkout.
- The script must behave the same as the CLI entrypoint for argument parsing, stdout/stderr, and exit codes.
- The script must still work when the `skills/` directory is reached through a workspace symlink.
- Skills that depend on it should include short command templates so the agent can invoke it consistently during long workflows.

## Implementation Notes

- Place the script at `skills/run-validation/scripts/run-command.py`.
- Resolve the real script path with `Path(__file__).resolve()` before deriving the repository root.
- Import the CLI module from `src/` directly rather than relying on an installed `project.scripts` entrypoint.
