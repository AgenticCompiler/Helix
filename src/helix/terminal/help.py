from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Iterable, TextIO


RESET = "\033[0m"
ACCENT = "\033[36m"


def supports_color(stream: TextIO, environ: Mapping[str, str] | None = None) -> bool:
    """Return whether ANSI color should be emitted for the given stream."""
    if environ is None:
        environ = os.environ

    if "NO_COLOR" in environ:
        return False

    if environ.get("CLICOLOR_FORCE") == "1":
        return True

    if environ.get("CLICOLOR") == "0":
        return False

    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    return bool(isatty())


def style_help_text(
    text: str,
    stream: TextIO,
    option_tokens: Iterable[str],
    env_var_tokens: Iterable[str],
    command_tokens: Iterable[str] = (),
    environ: Mapping[str, str] | None = None,
) -> str:
    if not supports_color(stream, environ):
        return text

    styled = text
    option_env_pattern = _build_token_pattern(option_tokens, env_var_tokens)
    if option_env_pattern is not None:
        styled = option_env_pattern.sub(lambda m: f"{ACCENT}{m.group(0)}{RESET}", styled)

    command_pattern = _build_command_pattern(command_tokens)
    if command_pattern is None:
        return styled

    return _style_command_positions(styled, command_pattern)


def _build_token_pattern(
    option_tokens: Iterable[str],
    env_var_tokens: Iterable[str],
) -> re.Pattern[str] | None:
    parts: list[str] = []

    for token in sorted(set(option_tokens), key=len, reverse=True):
        parts.append(rf"(?<!\w)({re.escape(token)})(?!\w)")

    for token in sorted(set(env_var_tokens), key=len, reverse=True):
        parts.append(rf"(?<![A-Z0-9_])({re.escape(token)})(?![A-Z0-9_])")

    if not parts:
        return None

    return re.compile("|".join(parts))


def _build_command_pattern(command_tokens: Iterable[str]) -> re.Pattern[str] | None:
    parts = [re.escape(token) for token in sorted(set(command_tokens), key=len, reverse=True)]
    if not parts:
        return None
    return re.compile("|".join(parts))


def _style_command_positions(text: str, command_pattern: re.Pattern[str]) -> str:
    styled_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        styled_line = _style_command_list_entry(line, command_pattern)
        if styled_line is not None:
            styled_lines.append(styled_line)
            continue
        styled_line = _style_command_example_entry(line, command_pattern)
        if styled_line is not None:
            styled_lines.append(styled_line)
            continue
        styled_lines.append(line)
    return "".join(styled_lines)


def _style_command_list_entry(line: str, command_pattern: re.Pattern[str]) -> str | None:
    match = re.match(r"^(\s{2,4})", line)
    if match is None:
        return None

    start = match.end()
    command_match = command_pattern.match(line, start)
    if command_match is None:
        return None

    gap_start = command_match.end()
    if not line[gap_start:].startswith("  "):
        return None

    return (
        f"{line[:command_match.start()]}"
        f"{ACCENT}{command_match.group(0)}{RESET}"
        f"{line[command_match.end():]}"
    )


def _style_command_example_entry(line: str, command_pattern: re.Pattern[str]) -> str | None:
    prefix = "  helix "
    if not line.startswith(prefix):
        return None

    command_match = command_pattern.match(line, len(prefix))
    if command_match is None:
        return None

    if command_match.end() < len(line) and not line[command_match.end()].isspace():
        return None

    return (
        f"{line[:command_match.start()]}"
        f"{ACCENT}{command_match.group(0)}{RESET}"
        f"{line[command_match.end():]}"
    )
