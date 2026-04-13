from __future__ import annotations

import shlex
from pathlib import Path
from typing import Iterable, TextIO


RESET = "\033[0m"
COLORS = {
    "agent": "\033[36m",
    "agents": "\033[36m",
    "remote": "\033[35m",
    "skills": "\033[32m",
    "files": "\033[33m",
}


def emit_verbose(stream: TextIO, category: str, message: str) -> None:
    print(f"{_format_prefix(stream, category)} {message}", file=stream)


def emit_verbose_lines(stream: TextIO, category: str, messages: Iterable[str]) -> None:
    for message in messages:
        emit_verbose(stream, category, message)


def format_symlink(path: Path) -> str:
    if path.is_symlink():
        return f"{path} -> {path.resolve()}"
    return str(path)


def format_command_messages(command: list[str]) -> list[str]:
    if not command:
        return ["command: <empty>"]

    prompt = command[-1]
    argv = command[:-1]
    # Keep the prompt separate from the argv preview so verbose output stays readable
    # even when the prompt spans many lines.
    messages = [f"command: {shlex.join(argv + ['<prompt>'])}", "prompt:"]
    messages.extend(f"  {line}" for line in prompt.splitlines())
    return messages


def _format_prefix(stream: TextIO, category: str) -> str:
    label = f"[{category}]"
    isatty = getattr(stream, "isatty", None)
    # Only emit ANSI color when we are writing to a real terminal; redirected logs
    # should stay plain text and easy to grep.
    if callable(isatty) and isatty():
        return f"{COLORS.get(category, '')}{label}{RESET}"
    return label
