# Help Color Implementation Plan

## Goal

Implement lightweight ANSI color styling for `triton-agent --help` and every subcommand help output, as specified in `docs/specs/2026-07-01-help-color-design.md`.

## Files to Touch

- `src/triton_agent/help_style.py` — new module for color detection, ANSI constants, and help text token styling.
- `src/triton_agent/cli.py` — replace direct `argparse.ArgumentParser` construction with a project-local `TritonArgumentParser` subclass, wire env-var token inventory.
- `tests/test_help_style.py` — new unit tests for the styling layer.
- `tests/test_cli.py` — add integration-style tests that exercise top-level and subcommand help printing with/without a TTY.

## Design Decisions

### Rendering model

- `argparse` continues to build plain-text help.
- `format_help()` returns plain text unchanged.
- `TritonArgumentParser.print_help()` post-processes the formatted string with the styling layer only when writing to a stream.

### Color enablement

Reimplement in `help_style.supports_color(stream, environ)` rather than importing `verbose._supports_color`, because the help feature has its own environment-variable rules (`NO_COLOR`, `CLICOLOR`, `CLICOLOR_FORCE`). The existing `verbose._supports_color` only checks `isatty()`.

Rules, evaluated in this order:

1. If `NO_COLOR` is present in the environment, disable.
2. If `CLICOLOR_FORCE == "1"`, enable.
3. If `CLICOLOR == "0"`, disable.
4. Otherwise enable when `stream.isatty()` is true.

### Token styling

- Accent color: cyan (`\033[36m`) for both option names and environment variable names in the first version.
- Wrap matched tokens with `\033[36m` and `\033[0m`.
- Build a single regex from the union of option tokens and env-var tokens.
- Sort tokens by length descending before building the regex so longer options are matched before shorter ones.
- Require word boundaries: env-var tokens are matched with `(?<![A-Z0-9_])NAME(?![A-Z0-9_])`; option tokens are matched with `(?<![\w-])TOKEN(?![\w-])` so `--help` does not match inside `--helpful` and `-h` does not match inside `foo-h`.

### Token inventories

- Option tokens: collect from `parser._actions` via `action.option_strings`. This covers both short (`-h`) and long (`--help`) forms automatically.
- Environment variable tokens: extract from `_TOP_LEVEL_ENVIRONMENT_VARIABLE_GROUPS` in `cli.py` and pass into the top-level parser instance. Subparsers receive an empty set because they do not render the environment-variable section.

### Parser integration

- Define `TritonArgumentParser(argparse.ArgumentParser)` in `cli.py`.
- Add an `env_var_names` constructor parameter defaulting to `()`.
- Override `print_help(file=None)` to:
  1. call `self.format_help()`,
  2. compute option tokens from `_actions`,
  3. call `help_style.style_help_text(text, file, option_tokens, self._env_var_names)`,
  4. write the result to `file`.
- In `build_parser()`:
  - instantiate `TritonArgumentParser(..., env_var_names=_collect_env_var_names())` for the top-level parser,
  - pass `formatter_class=argparse.RawDescriptionHelpFormatter` as before,
  - ensure subparsers use the same class via `subparsers.add_parser(...)`; `argparse` uses the parent class by default when `add_parser` is called on a subparsers action created from a `TritonArgumentParser`, but we will verify this.

### Testing strategy

- Keep all existing `format_help()` assertions plain-text.
- Add `tests/test_help_style.py` covering:
  - `supports_color` for TTY, non-TTY, `NO_COLOR`, `CLICOLOR=0`, `CLICOLOR_FORCE=1`.
  - `style_help_text` returns plain text when color disabled.
  - option tokens are colored when enabled.
  - env-var tokens are colored when enabled.
  - word-boundary protection (e.g. `--helpful` not colored, `LLM_API_KEY_X` not colored).
- Add `tests/test_cli.py` cases covering:
  - top-level `parser.print_help()` emits ANSI when stream is a TTY.
  - subcommand `parser.parse_args(["gen-test", "--help"])` emits ANSI when stdout is a TTY.
  - `NO_COLOR` disables ANSI even on a TTY.

## Verification

Run the standard repository verification commands after each meaningful change:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_help_style.py tests/test_cli.py`

## Implementation Order

1. Implement `src/triton_agent/help_style.py` with tests first (TDD).
2. Introduce `TritonArgumentParser` in `src/triton_agent/cli.py` and wire env-var tokens.
3. Add CLI integration tests for colored help output.
4. Run full verification suite and fix any issues.
