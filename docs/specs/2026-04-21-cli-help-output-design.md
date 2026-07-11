# CLI Help Output Design

## Summary

- Improve `helix --help` so it reads like a navigation page instead of a flat command dump.
- Keep the implementation in the Python standard library by extending the existing `argparse` setup.
- Add short help text for every subcommand and append grouped command guidance plus examples to the top-level help output.

## Goals

- Make the top-level help easier to scan for first-time users.
- Show what each subcommand is for without requiring users to try every `--help` page.
- Keep the public command names, flags, and parsing behavior unchanged.

## Non-Goals

- Do not introduce third-party CLI formatting packages.
- Do not rename commands or change existing flag semantics.
- Do not add a separate docs-only command or interactive help mode.

## Proposed Behavior

- Top-level help should use a shorter usage line: `helix [-h] COMMAND ...`.
- Top-level help should include:
  - a one-line description of the CLI
  - the existing subcommand list with one-line summaries
  - an extra grouped reference section for generation, execution, comparison, and optimization commands
  - a short examples section
- Subcommand help should include:
  - a one-line summary in the parent command list
  - a longer description on `helix <command> --help`

## Implementation Notes

- Add help metadata to the command spec used by `build_parser()`.
- Pass `help=` and `description=` when creating each subparser.
- Use top-level parser `description`, `usage`, and `epilog` fields to append grouped command guidance and examples.
- Keep alias handling unchanged so help continues to show only canonical kebab-case commands.

## Verification

- Add parser-level tests that assert grouped help sections and examples appear in `build_parser().format_help()`.
- Add a subcommand help test that asserts the command description appears in `format_help()`.
- Run targeted CLI tests after the change.
