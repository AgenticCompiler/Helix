from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable, TextIO


RESET = "\033[0m"
COMMAND_BODY_FOREGROUND = "\033[38;5;245m"
COLORS = {
    "agent": "\033[36m",
    "agents": "\033[36m",
    "command": "\033[34m",
    "hooks": "\033[32m",
    "remote": "\033[35m",
    "skills": "\033[32m",
    "files": "\033[33m",
}


def emit_verbose(stream: TextIO, category: str, message: str) -> None:
    print(f"{_format_prefix(stream, category)} {message}", file=stream)


def emit_verbose_lines(stream: TextIO, category: str, messages: Iterable[str]) -> None:
    for message in messages:
        emit_verbose(stream, category, message)


def emit_command_block(stream: TextIO, command: list[str]) -> None:
    messages = format_command_messages(command)
    if not messages:
        return

    emit_verbose(stream, "command", messages[0])
    if len(messages) == 1:
        return

    emit_verbose(stream, "command", messages[1])
    prompt_lines = messages[2:]
    if not prompt_lines:
        return

    if _supports_color(stream):
        for line in prompt_lines:
            print(f"{COMMAND_BODY_FOREGROUND}{line}{RESET}", file=stream)
        return

    for line in prompt_lines:
        print(line, file=stream)


def format_symlink(path: Path) -> str:
    if path.is_symlink():
        return f"{path} -> {path.resolve()}"
    return str(path)


def format_command_messages(command: list[str]) -> list[str]:
    if not command:
        return ["<empty>"]

    prompt = command[-1]
    argv = command[:-1]
    # Keep the prompt separate from the argv preview so verbose output stays readable
    # even when the prompt spans many lines.
    messages = [shlex.join(argv + ["<prompt>"]), "prompt:"]
    messages.extend(f"  {line}" for line in prompt.splitlines())
    return messages


def _format_prefix(stream: TextIO, category: str) -> str:
    label = f"[{category}]"
    # Only emit ANSI color when we are writing to a real terminal; redirected logs
    # should stay plain text and easy to grep.
    if _supports_color(stream):
        return f"{COLORS.get(category, '')}{label}{RESET}"
    return label


def _supports_color(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and isatty()
