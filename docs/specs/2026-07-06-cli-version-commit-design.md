# CLI Version Commit Design

## Summary

Extend the top-level `helix` parser with a standard `-v` / `--version` flag that prints the CLI build commit and exits successfully.

## Goals

- Support `helix -v`.
- Support `helix --version`.
- Print only the build commit display value.
- Reuse the existing build-info resolver so help output and version output stay aligned.

## Non-Goals

- Do not add a `version` subcommand.
- Do not print the package version string from `pyproject.toml`.
- Do not change any subcommand parser behavior.
- Do not change the existing `Build info:` section in top-level help.
- Do not fail when commit metadata is unavailable.

## User-Visible Behavior

- `helix -v` prints one line containing only the build commit display value, then exits with status `0`.
- `helix --version` prints the same output and exits with status `0`.
- The printed value matches `get_build_info_display()`.
- When commit metadata is unavailable, the output is `unknown`.

## Implementation Notes

- Add a top-level parser argument in `src/helix/cli.py` using `argparse`'s standard version action.
- Register both `-v` and `--version` on the root parser.
- Use the existing `helix.build_info.get_build_info_display()` helper as the version string source.
- Keep the change surgical: no new resolver logic, no new metadata files, and no changes to subparsers.

## Verification

- Add CLI tests covering `main(["-v"])` and `main(["--version"])`.
- Assert the output is exactly the mocked build commit display value plus a trailing newline.
- Confirm both invocations exit with code `0`.
