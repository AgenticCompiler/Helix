# Help Color Design

## Summary

Add lightweight ANSI color styling to `triton-agent --help` output so option names and supported environment variable names are easier to scan in a terminal. The styling must apply consistently to the top-level help text and every subcommand help text without changing the underlying help content.

## User-Visible Behavior

- Help output remains plain text by default when redirected or captured.
- Help output uses ANSI color when written to an interactive terminal.
- Option names such as `-h`, `--help`, and `--enable-subagent` are rendered in an accent color.
- Supported environment variable names such as `TRITON_AGENT_REMOTE` are rendered in an accent color.
- Descriptions, prose, default explanations, and ordinary body text remain uncolored in the initial version.
- The same styling rules apply to:
  - top-level `triton-agent --help`
  - subcommand help such as `triton-agent optimize --help`
  - manually assembled help sections such as `Environment variables:`
  - any option tokens that appear in usage lines or wrapped help prose

## Color Enablement Rules

- Color defaults to `auto`.
- In `auto` mode, emit ANSI color only when the target stream is a TTY.
- If `NO_COLOR` is present in the environment, disable color.
- If `CLICOLOR=0`, disable color.
- If `CLICOLOR_FORCE=1`, force color even when the stream is not a TTY.

The initial implementation does not add a public `--color` CLI flag. The behavior is controlled only by terminal detection and standard environment variables.

## Non-Goals

- No changes to help wording, argument semantics, grouping, or command ordering.
- No migration away from `argparse`.
- No styling for metavars, choice lists, headings, or descriptive paragraphs in the initial version.
- No dependency on `rich` or another full-screen formatting library.

## Design

### Rendering Model

Keep help generation and help styling separate.

- `argparse` remains responsible for building the canonical help text.
- `format_help()` continues to return plain text.
- A thin help styling layer post-processes the final help string only when the parser is printing to a stream.

This separation keeps tests and downstream consumers stable while limiting ANSI handling to interactive display paths.

### Parser Integration

Introduce a thin custom parser subclass for the repository CLI.

- Replace direct `argparse.ArgumentParser` construction in `build_parser()` with a project-local subclass such as `TritonArgumentParser`.
- Override the parser print path so it:
  - obtains the normal plain-text help string
  - applies optional color styling
  - writes the result to the destination stream
- Preserve existing `parse_args()` and `format_help()` behavior.

The same parser subclass should be used for subparsers so all command help text shares one rendering path.

### Styling Module

Add a dedicated helper module for help styling, for example `src/triton_agent/help_style.py`.

This module should own:

- color support detection
- environment-variable-based enablement rules
- help text token styling
- ANSI constants used only for help rendering

Keeping this logic out of `cli.py` preserves the CLI's orchestration role and keeps the color behavior feature-local.

### Token Styling Strategy

Apply color after layout, not before layout.

- Generate help text first.
- Then style known tokens in the final string.

This avoids interfering with `argparse` width calculation and wrapped alignment.

The initial token set should be conservative:

- option tokens recognized from parser actions, including short and long forms
- known supported environment variable names from the top-level environment-variable section source data

The styling layer should prefer known token inventories over broad pattern matching so ordinary uppercase prose is not colored accidentally.

## Testing Strategy

Keep existing help-content tests intact and add focused coverage for the new rendering layer.

- Existing tests that call `format_help()` should continue to assert plain text content.
- Add unit tests for the styling module covering:
  - TTY enables color in default mode
  - non-TTY disables color in default mode
  - `NO_COLOR` disables color
  - `CLICOLOR=0` disables color
  - `CLICOLOR_FORCE=1` forces color
  - option tokens are colored
  - supported environment variable names are colored
- Add an integration-style CLI help test that exercises `main([...,"--help"])` or parser help printing and verifies:
  - ANSI appears only when enabled
  - top-level and subcommand help both use the shared styling path

## Risks And Mitigations

### ANSI escape sequences can break layout

Mitigation: never inject ANSI before `argparse` performs formatting. Post-process only the final text.

### Help tests may become brittle

Mitigation: keep `format_help()` plain text and isolate ANSI assertions to new tests that explicitly inspect styled output.

### Over-coloring can reduce readability

Mitigation: limit initial styling to option names and supported environment variable names only.

## Implementation Notes

- Reuse the repository's existing TTY-aware color conventions where practical, but keep help styling logic independent from verbose/status rendering helpers.
- Prefer a single accent color for both option names and environment variable names in the first version. A later change can split the palette if needed without changing the rendering architecture.
- Avoid introducing new public contracts unless user feedback shows a need for explicit `--color` control.
