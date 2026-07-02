from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from triton_agent.backends.factory import create_runner
from triton_agent.distill.skills_workspace import knowledge_skill_name
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.paths import skills_root as repository_skills_root
from triton_agent.skills.staging import SkillLinkManager, SkillLinkSet, staged_skill_dir
from triton_agent.terminal.verbose import emit_verbose_lines


DISTILL_SKILL_NAME = "ascend-npu-distill-patterns"


def run_distill_agent(
    *,
    agent_name: str,
    workdir: Path,
    prompt: str,
    stream_output: bool,
    verbose: bool,
    language: Literal["triton", "tilelang"] = "triton",
    skills_root: Path | None = None,
    repository_skill_names: tuple[str, ...] = (DISTILL_SKILL_NAME,),
    stage_editable_knowledge: bool = True,
    output_label: str = "",
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> AgentResult:
    prefixed_stdout = _prefixed_stream(stdout or sys.stdout, output_label) if stream_output and output_label else stdout
    request = AgentRequest(
        command_kind=CommandKind.DISTILL,
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

    manager = SkillLinkManager(repository_skills_root())
    links = manager.prepare_skills(
        agent_name,
        workdir,
        skill_names=repository_skill_names,
    )
    knowledge_link = (
        _stage_editable_knowledge(
            backend=agent_name,
            workdir=workdir,
            editable_skills_dir=skills_root,
            language=language,
        )
        if stage_editable_knowledge
        else SkillLinkSet([])
    )
    verbose_stream = stderr or sys.stderr
    if verbose:
        messages = manager.describe_prepare(links)
        if knowledge_link.created_paths:
            messages.extend(f"created skill copy {path}" for path in knowledge_link.created_paths)
        emit_verbose_lines(verbose_stream, "skills", messages)
    try:
        runner = create_runner(agent_name)
        return cast(Any, runner).run(request, stdout=prefixed_stdout, stderr=stderr)
    finally:
        if verbose:
            messages = manager.describe_cleanup(links)
            messages.extend(f"removed skill copy {path}" for path in reversed(knowledge_link.created_paths))
            emit_verbose_lines(verbose_stream, "skills", messages)
        _cleanup_knowledge_link(knowledge_link)
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


def _stage_editable_knowledge(
    *,
    backend: str,
    workdir: Path,
    editable_skills_dir: Path,
    language: str,
) -> SkillLinkSet:
    knowledge_name = knowledge_skill_name(language)
    source = editable_skills_dir / knowledge_name
    if not source.is_dir():
        raise RuntimeError(f"Editable knowledge skill does not exist: {source}")
    target_root = workdir / staged_skill_dir(backend)
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / knowledge_name
    if target.exists():
        return SkillLinkSet([])
    shutil.copytree(source, target, symlinks=False)
    return SkillLinkSet([target])


def _cleanup_knowledge_link(link_set: SkillLinkSet) -> None:
    for path in reversed(link_set.created_paths):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
