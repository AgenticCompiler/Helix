from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from triton_agent.backends.factory import create_runner
from triton_agent.diff_skills_update.skills_workspace import knowledge_skill_name
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.skills.staging import SkillLinkManager
from triton_agent.terminal.verbose import emit_verbose_lines


def run_diff_skills_agent(
    *,
    agent_name: str,
    workdir: Path,
    prompt: str,
    stream_output: bool,
    verbose: bool,
    language: Literal["triton", "tilelang"] = "triton",
    skills_root: Path | None = None,
    output_label: str = "",
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> AgentResult:
    prefixed_stdout = _prefixed_stream(stdout or sys.stdout, output_label) if stream_output and output_label else stdout
    request = AgentRequest(
        command_kind=CommandKind.DIFF_SKILLS_UPDATE,
        input_path=workdir,
        operator_path=None,
        output_path=None,
        test_mode=None,
        bench_mode=None,
        language=language,
        interact=False,
        verbose=verbose,
        stream_output=stream_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="",
        prompt=prompt,
        workdir=workdir,
        show_output_label=_safe_output_label(output_label),
    )
    if skills_root is None:
        runner = create_runner(agent_name)
        return cast(Any, runner).run(request, stdout=prefixed_stdout, stderr=stderr)

    manager = SkillLinkManager(skills_root)
    links = manager.prepare_skills(
        agent_name,
        workdir,
        skill_names=(knowledge_skill_name(language),),
    )
    verbose_stream = stderr or sys.stderr
    if verbose:
        emit_verbose_lines(verbose_stream, "skills", manager.describe_prepare(links))
    try:
        runner = create_runner(agent_name)
        return cast(Any, runner).run(request, stdout=prefixed_stdout, stderr=stderr)
    finally:
        if verbose:
            emit_verbose_lines(verbose_stream, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        if verbose and warnings:
            emit_verbose_lines(verbose_stream, "skills", warnings)


class _PrefixedStream:
    def __init__(self, stream: TextIO, label: str) -> None:
        self._stream = stream
        self._prefix = f"{label} "
        self._at_line_start = True

    def write(self, text: str) -> int:
        for char in text:
            if self._at_line_start and char not in {"\n", "\r"}:
                self._stream.write(self._prefix)
                self._at_line_start = False
            self._stream.write(char)
            if char == "\n":
                self._at_line_start = True
        return len(text)

    def flush(self) -> None:
        self._stream.flush()


def _prefixed_stream(stream: TextIO, label: str) -> TextIO:
    return cast(TextIO, _PrefixedStream(stream, label))


def _safe_output_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in label).strip("-")
