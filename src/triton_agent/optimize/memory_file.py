from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional

from triton_agent.optimize.pattern_reminders import (
    build_high_priority_pattern_reminder_lines,
)
from triton_agent.optimize.subagents import optimize_subagent_recommendation_lines
from triton_agent.optimize.prompts import (
    cann_ext_api_lines,
    compiler_source_analysis_lines,
    layered_analysis_lines,
)


@dataclass
class MemoryFileState:
    """Temporary top-level memory file (`AGENTS.md` or `CLAUDE.md`) for one run."""

    guidance_path: Path
    backup_path: Optional[Path]
    created_guidance: bool


def _render_bullet_block(lines: list[str]) -> str:
    return "".join(f"- {line}\n" for line in lines)


def _render_line_block(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _render_high_priority_pattern_block(*, optimize_knowledge_skill_name: str | None) -> str:
    if optimize_knowledge_skill_name is None:
        return ""
    reminder_lines = build_high_priority_pattern_reminder_lines(optimize_knowledge_skill_name)
    if not reminder_lines:
        return ""
    return _render_line_block(["High-priority generic pattern reminders for this run:"]) + (
        _render_bullet_block(reminder_lines)
    )


def _format_speedup_target(min_speedup: float) -> str:
    return f"{min_speedup:.2f}x"


def _min_speedup_guidance_lines(*, min_speedup: float | None) -> list[str]:
    if min_speedup is None:
        return []
    return [
        f"Optimize session target: reach at least {_format_speedup_target(min_speedup)} geomean speedup over the baseline.",
        "The optimize runner injects this target into `submit-round` automatically; do not guess or override a different speedup target.",
        "If `submit-round` reports that this target is satisfied, stop the optimize session immediately.",
    ]


def _optimize_target_guidance_lines(*, optimize_target: str) -> list[str]:
    if optimize_target == "operator":
        return [
            "Target optimization scope: operator.",
            "Optimize end-to-end operator latency.",
            "You may improve wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code in this session.",
            "Do not replace the Triton Ascend NPU computation path with a pure PyTorch rewrite.",
            "When reviewing performance, keep both kernel and total-op `compare-perf` views visible and treat total-op as the canonical session conclusion.",
        ]
    return [
        "Target optimization scope: kernel.",
        "Optimize the Triton Ascend NPU kernel path itself.",
        "Do not replace the Triton Ascend NPU computation path with a pure PyTorch rewrite.",
        "When reviewing performance, prefer the kernel-oriented `compare-perf` view and record any fallback away from pure kernel results in `effective_metric_source`.",
    ]


_OPTIMIZE_GUIDANCE_RULES_BLOCK = dedent(
    """\
    IMPORTANT:
        - Use `ascend-npu-optimize-state` skill `submit-baseline` to submit the initial baseline.
        - Use `ascend-npu-optimize-state` skill `start-round` to start a new optimization round.
        - Use `ascend-npu-optimize-state` skill `set-current-round-state` if the active round's strategy or required evidence depth changes mid-round.
        - Use `ascend-npu-optimize-state` skill `submit-round` to submit each complete optimization round.

    - Read files cautiously. Do not read unrelated files speculatively or just in case.
    - Prefer the smallest source that can unblock the next decision.
    - Follow the user's instructions strictly.
    """
)


_SHARED_GUIDANCE_TEMPLATE = (
    dedent(
        """\
        # {guidance_filename}

        ## Triton Agent Optimize Orchestration

        This workspace is under optimize orchestration.

        """
    )
    + _OPTIMIZE_GUIDANCE_RULES_BLOCK
    + dedent(
        """\
        Use the staged workspace skills as the workflow source of truth.
        Invocation-specific behavior comes from the launch prompt.
        Use `supervisor-report.md` as the supervisor audit report file when supervised mode is active.
        Treat `baseline/` as the canonical optimize baseline.
        Use `compare-perf` as the authoritative source for round performance summaries.
        {analysis_block}

        {high_priority_pattern_block}

        {compiler_source_block}

        {cann_ext_api_block}"""
    )
)


_ROUND_GATED_GUIDANCE_TEMPLATE = (
    dedent(
        """\
        # {guidance_filename}

        ## Triton Agent Optimize Round Loop

        This workspace is under an optimize round loop.

        """
    )
    + _OPTIMIZE_GUIDANCE_RULES_BLOCK
    + dedent(
        """\
        Use the staged workspace skills as the workflow source of truth.
        Invocation-specific behavior comes from the launch prompt.
        The CLI will inject previous round validation results directly into the next worker prompt when another round is needed.
        Treat `baseline/` as the canonical optimize baseline.
        Use `compare-perf` as the authoritative source for round performance summaries.

        {analysis_block}

        {high_priority_pattern_block}

        {compiler_source_block}

        {cann_ext_api_block}"""
    )
)


class MemoryFileManager:
    """Owns rendering and lifecycle for the temporary workspace memory file."""

    def guidance_filename(self, agent_name: str) -> str:
        if agent_name == "claude":
            return "CLAUDE.md"
        return "AGENTS.md"

    def prepare_shared(
        self,
        workdir: Path,
        *,
        agent_name: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        min_speedup: float | None = None,
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> MemoryFileState:
        """Write the shared orchestration memory file used by supervised optimize."""
        return self._prepare(
            workdir,
            agent_name=agent_name,
            content=self._render_shared_guidance(
                guidance_filename=self.guidance_filename(agent_name),
                language=language,
                optimize_target=optimize_target,
                min_speedup=min_speedup,
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
                enable_cann_ext_api=enable_cann_ext_api,
                enable_subagent=enable_subagent,
                optimize_knowledge_skill_name=optimize_knowledge_skill_name,
            ),
        )

    def prepare_round_gated(
        self,
        workdir: Path,
        *,
        agent_name: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        min_speedup: float | None = None,
        include_supervisor_handoff: bool = True,
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> MemoryFileState:
        """Write the round-gated optimize memory file for checked/supervised modes."""
        guidance_filename = self.guidance_filename(agent_name)
        return self._prepare(
            workdir,
            agent_name=agent_name,
            content=self._render_round_gated_guidance(
                guidance_filename=guidance_filename,
                language=language,
                optimize_target=optimize_target,
                min_speedup=min_speedup,
                include_supervisor_handoff=include_supervisor_handoff,
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
                enable_cann_ext_api=enable_cann_ext_api,
                enable_subagent=enable_subagent,
                optimize_knowledge_skill_name=optimize_knowledge_skill_name,
            ),
        )

    def cleanup(self, state: MemoryFileState) -> list[str]:
        """Delete a temporary memory file or restore the user's original one."""
        warnings: list[str] = []
        if state.backup_path is None:
            try:
                if state.created_guidance and state.guidance_path.exists():
                    state.guidance_path.unlink()
            except OSError as exc:
                warnings.append(
                    f"Failed to remove temporary guidance file {state.guidance_path}: {exc}"
                )
        else:
            try:
                state.guidance_path.write_bytes(state.backup_path.read_bytes())
                state.backup_path.unlink()
            except OSError as exc:
                warnings.append(
                    "Failed to restore original guidance file "
                    f"from {state.backup_path}: {exc}"
                )
        return warnings

    def describe_prepare(self, state: MemoryFileState, *, description: str) -> list[str]:
        messages: list[str] = []
        if state.backup_path is not None:
            messages.append(f"backed up workspace guidance file to {state.backup_path}")
        messages.append(f"wrote {description} {state.guidance_path}")
        return messages

    def describe_cleanup(self, state: MemoryFileState) -> list[str]:
        if state.backup_path is None:
            return [f"removing temporary guidance file {state.guidance_path}"]
        return [
            f"restoring workspace guidance file from {state.backup_path}",
            f"removing backup file {state.backup_path}",
        ]

    def _prepare(self, workdir: Path, *, agent_name: str, content: str) -> MemoryFileState:
        """Backup any existing memory file, then replace it with optimize guidance."""
        guidance_path = workdir / self.guidance_filename(agent_name)
        guidance_preexisting = guidance_path.exists()
        backup_path: Optional[Path] = None

        if guidance_preexisting:
            backup_path = self._next_backup_path(guidance_path)
            backup_path.write_bytes(guidance_path.read_bytes())

        guidance_path.write_text(content, encoding="utf-8")
        return MemoryFileState(
            guidance_path=guidance_path,
            backup_path=backup_path,
            created_guidance=not guidance_preexisting,
        )

    def _next_backup_path(self, guidance_path: Path) -> Path:
        candidate = guidance_path.with_suffix(guidance_path.suffix + ".triton-agent.bak")
        counter = 1
        while candidate.exists():
            candidate = guidance_path.with_suffix(
                guidance_path.suffix + f".triton-agent.{counter}.bak"
            )
            counter += 1
        return candidate

    def _render_shared_guidance(
        self,
        *,
        guidance_filename: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        min_speedup: float | None = None,
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> str:
        return _SHARED_GUIDANCE_TEMPLATE.format(
            guidance_filename=guidance_filename,
            analysis_block=_render_bullet_block(
                layered_analysis_lines(round_scope="each round", language=language)
                + (
                    [
                        "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
                    ]
                    if optimize_target == "operator"
                    else []
                )
                + (
                    optimize_subagent_recommendation_lines(language=language)
                    if enable_subagent
                    else []
                )
            ),
            high_priority_pattern_block=_render_high_priority_pattern_block(
                optimize_knowledge_skill_name=optimize_knowledge_skill_name
            ),
            compiler_source_block=_render_line_block(
                _optimize_target_guidance_lines(optimize_target=optimize_target)
            )
            + _render_line_block(
                _min_speedup_guidance_lines(min_speedup=min_speedup)
            )
            + _render_line_block(
                compiler_source_analysis_lines(
                    compiler_source_path=compiler_source_path,
                    compiler_source_commit=compiler_source_commit,
                    language=language,
                )
            ),
            cann_ext_api_block=_render_line_block(
                cann_ext_api_lines(enabled=enable_cann_ext_api, language=language)
            ),
        )

    def _render_round_gated_guidance(
        self,
        *,
        guidance_filename: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        min_speedup: float | None = None,
        include_supervisor_handoff: bool = True,
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> str:
        base = _ROUND_GATED_GUIDANCE_TEMPLATE.format(
            guidance_filename=guidance_filename,
            analysis_block=_render_bullet_block(
                layered_analysis_lines(round_scope="each round", language=language)
                + (
                    [
                        "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
                    ]
                    if optimize_target == "operator"
                    else []
                )
                + (
                    optimize_subagent_recommendation_lines(language=language)
                    if enable_subagent
                    else []
                )
            ),
            high_priority_pattern_block=_render_high_priority_pattern_block(
                optimize_knowledge_skill_name=optimize_knowledge_skill_name
            ),
            compiler_source_block=_render_line_block(
                _optimize_target_guidance_lines(optimize_target=optimize_target)
            )
            + _render_line_block(
                _min_speedup_guidance_lines(min_speedup=min_speedup)
            )
            + _render_line_block(
                compiler_source_analysis_lines(
                    compiler_source_path=compiler_source_path,
                    compiler_source_commit=compiler_source_commit,
                    language=language,
                )
            ),
            cann_ext_api_block=_render_line_block(
                cann_ext_api_lines(enabled=enable_cann_ext_api, language=language)
            ),
        )
        if include_supervisor_handoff:
            base += (
                "\nUse `supervisor-report.md` as the supervisor audit report file.\n"
            )
        return base
