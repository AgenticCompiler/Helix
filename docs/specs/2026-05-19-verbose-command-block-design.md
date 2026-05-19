# Verbose Command Block Design

## Goal

Improve verbose command readability by keeping the command header concise while rendering long prompt bodies as a separate block.

## Problem

The current command preview output repeats the `[command]` prefix on every prompt line, which makes long optimize prompts visually noisy and harder to scan.

## User-Visible Behavior

- The first command preview line should remain `[command] <argv ...>`.
- The prompt header line should remain `[command] prompt:`.
- Prompt body lines should no longer repeat `[command]`.
- In TTY output, prompt body lines should render in a muted text color with no background fill.
- In non-TTY output, prompt body lines should stay plain text and indented.

## Design

- Keep `format_command_messages()` responsible for splitting the command preview into:
  - argv preview
  - prompt header
  - indented prompt body lines
- Add a dedicated shared emitter for command previews that:
  - uses the standard verbose prefix for the first two lines
  - prints prompt body lines without a prefix
  - applies a muted foreground color to prompt body lines only when writing to a TTY
- Route both the common backend launcher and the OpenHands-specific launcher through the same shared emitter.

## Testing

- Add a non-TTY unit test that verifies prompt body lines are not prefixed.
- Add a TTY unit test that verifies prompt body lines use the muted text styling.
- Update backend verbose tests to assert prompt body lines do not include `[command]`.

## Scope

- Do not change non-command verbose categories.
- Do not change the command argv preview format.
