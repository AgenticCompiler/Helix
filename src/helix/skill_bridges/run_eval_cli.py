"""Bridge for resolving the staged run-eval CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

from helix.skills.loader import operator_eval_script_path


def cli_script_path() -> Path:
    return operator_eval_script_path("cli")
