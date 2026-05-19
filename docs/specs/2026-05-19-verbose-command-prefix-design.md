# Verbose Command Prefix Design

## Goal

Make verbose command previews read naturally after introducing the dedicated `[command]` prefix.

## Problem

The current verbose output prints command previews like:

```text
[command] command: opencode ...
```

The category prefix already tells the reader this line is a command preview, so the additional `command:` label is redundant.

## User-Visible Behavior

- Verbose command previews should render as `[command] <argv ...>` instead of `[command] command: <argv ...>`.
- The prompt block should remain unchanged.
- The dedicated `[command]` color should remain unchanged.

## Design

- Update the shared `format_command_messages()` helper so the first message is only the shell-joined argv preview.
- Keep the `<prompt>` placeholder in the argv preview so long prompts stay visually compact.
- Keep the existing `prompt:` follow-up lines unchanged.
- Update the OpenHands backend to reuse the shared command-formatting helper so all backends print command previews consistently.

## Testing

- Update verbose runner tests to expect `[command] <argv ...>` and no longer expect `command:`.
- Keep the TTY color test for the `command` category.

## Scope

- Do not change any non-command verbose categories.
- Do not change prompt rendering beyond removing the redundant `command:` label.
