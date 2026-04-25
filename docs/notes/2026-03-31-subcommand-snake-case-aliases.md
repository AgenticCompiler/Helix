# Subcommand Snake Case Aliases

## Summary

Accept snake_case spellings for the CLI subcommands while keeping kebab-case as the only displayed form in help and documentation.

## User-Visible Behavior

- Users can invoke `gen_test`, `run_test`, `gen_bench`, and `run_bench` as aliases for the existing kebab-case subcommands.
- Help text and documentation continue to present only the canonical kebab-case names.
- Command behavior, prompts, defaults, and backend selection stay unchanged.

## Implementation Notes

- Normalize the first CLI argument from known snake_case aliases to the canonical kebab-case subcommand before `argparse` parses it.
- Keep alias handling in the CLI layer so the rest of the command pipeline continues to operate on canonical command names.
- Cover both alias parsing and help-text behavior with parser tests.
