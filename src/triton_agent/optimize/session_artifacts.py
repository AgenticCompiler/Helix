from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from triton_agent.optimize.archive import ArchiveManager, ArchiveState
from triton_agent.optimize.memory_file import MemoryFileManager, MemoryFileState
from triton_agent.optimize.subagents import perf_diagnosis_subagent_definition
from triton_agent.subagents import SubagentManager, SubagentStageSet


@dataclass
class OptimizeSessionArtifactsState:
    """Full artifact bundle for multi-invocation optimize runs."""

    memory_file: MemoryFileState
    archive: ArchiveState
    subagent_stage_set: SubagentStageSet | None = None

    hidden_triton_agent_dir: Path | None = None
    supervisor_report_path: Path | None = None
    supervisor_history_dir: Path | None = None

    @property
    def guidance_path(self) -> Path:
        return self.memory_file.guidance_path

    @property
    def backup_path(self) -> Path | None:
        return self.memory_file.backup_path

    @property
    def created_guidance(self) -> bool:
        return self.memory_file.created_guidance

    @property
    def run_archive_dir(self) -> Path:
        return self.archive.run_archive_dir

    def agent_session_path(self, label: str) -> Path:
        return self.archive.agent_session_path(label)

    def trace_path(self, label: str) -> Path:
        return self.archive.trace_path(label)

    def trace_summary_path(self, label: str) -> Path:
        return self.archive.trace_summary_path(label)

    @property
    def shared_guidance_snapshot_path(self) -> Path | None:
        """Archive destination for the shared memory-file snapshot, when enabled."""
        return self.archive.shared_guidance_snapshot_path

    @property
    def created_paths(self) -> tuple[Path, ...]:
        created: list[Path] = []
        if self.subagent_stage_set is not None:
            created.extend(self.subagent_stage_set.created_paths)
        if self.supervisor_report_path is not None:
            created.append(self.supervisor_report_path)
        return tuple(created)


class OptimizeSessionArtifactsManager:
    """Thin facade that coordinates memory-file, supervised runtime files, and archive artifacts."""

    def __init__(
        self,
        memory_files: MemoryFileManager | None = None,
        archives: ArchiveManager | None = None,
        subagents: SubagentManager | None = None,
    ) -> None:
        self._memory_files = memory_files or MemoryFileManager()
        self._archives = archives or ArchiveManager()
        self._subagents = subagents or SubagentManager()

    def prepare_checked_session(
        self,
        workdir: Path,
        agent_name: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> OptimizeSessionArtifactsState:
        """Prepare artifacts for checked optimize without a live handoff file."""
        archive_state = self._archives.prepare(workdir, include_shared_guidance_snapshot=True)
        subagent_stage_set = self._prepare_subagents(
            agent_name=agent_name,
            workdir=workdir,
            language=language,
            optimize_target=optimize_target,
            enable_cann_ext_api=enable_cann_ext_api,
            enable_subagent=enable_subagent,
        )
        try:
            memory_file_state = self._memory_files.prepare_round_gated(
                workdir,
                agent_name=agent_name,
                language=language,
                optimize_target=optimize_target,
                include_supervisor_handoff=False,
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
                enable_cann_ext_api=enable_cann_ext_api,
                enable_subagent=enable_subagent,
                optimize_knowledge_skill_name=optimize_knowledge_skill_name,
            )
        except Exception:
            if subagent_stage_set is not None:
                self._subagents.cleanup(subagent_stage_set)
            raise
        return OptimizeSessionArtifactsState(
            memory_file=memory_file_state,
            archive=archive_state,
            subagent_stage_set=subagent_stage_set,
        )

    def prepare_supervised_session(
        self,
        workdir: Path,
        agent_name: str,
        language: str = "triton",
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        enable_subagent: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> OptimizeSessionArtifactsState:
        """Prepare the full artifact set used by worker/supervisor orchestration."""
        hidden_triton_agent_dir = self._prepare_hidden_triton_agent_dir(workdir)
        supervisor_report_path = hidden_triton_agent_dir / "supervisor-report.md"
        supervisor_history_dir = hidden_triton_agent_dir / "supervisor-history"
        supervisor_history_dir.mkdir(parents=True, exist_ok=True)
        supervisor_report_path.write_text(
            "# Optimize Supervisor Report\n\nPending first supervisor pass.\n",
            encoding="utf-8",
        )
        archive_state = self._archives.prepare(
            workdir,
            include_shared_guidance_snapshot=True,
        )
        subagent_stage_set: SubagentStageSet | None = None
        try:
            subagent_stage_set = self._prepare_subagents(
                agent_name=agent_name,
                workdir=workdir,
                language=language,
                optimize_target=optimize_target,
                enable_cann_ext_api=enable_cann_ext_api,
                enable_subagent=enable_subagent,
            )
            memory_file_state = self._memory_files.prepare_shared(
                workdir,
                agent_name=agent_name,
                optimize_target=optimize_target,
                compiler_source_path=compiler_source_path,
                compiler_source_commit=compiler_source_commit,
                enable_cann_ext_api=enable_cann_ext_api,
                enable_subagent=enable_subagent,
                optimize_knowledge_skill_name=optimize_knowledge_skill_name,
            )
        except Exception:
            if subagent_stage_set is not None:
                self._subagents.cleanup(subagent_stage_set)
            self._cleanup_hidden_triton_agent_dir(hidden_triton_agent_dir)
            raise
        return OptimizeSessionArtifactsState(
            memory_file=memory_file_state,
            archive=archive_state,
            subagent_stage_set=subagent_stage_set,
            hidden_triton_agent_dir=hidden_triton_agent_dir,
            supervisor_report_path=supervisor_report_path,
            supervisor_history_dir=supervisor_history_dir,
        )

    def archive(self, state: OptimizeSessionArtifactsState) -> list[str]:
        """Persist the final multi-invocation outputs into the run archive."""
        return self._archives.archive(
            state.archive,
            guidance_path=state.guidance_path,
            supervisor_report_path=state.supervisor_report_path,
            history_dir=state.supervisor_history_dir,
        )

    def cleanup_session(self, state: OptimizeSessionArtifactsState) -> list[str]:
        """Remove or restore the temporary top-level memory file for an optimize run."""
        warnings: list[str] = []
        warnings.extend(self._memory_files.cleanup(state.memory_file))
        if state.subagent_stage_set is not None:
            warnings.extend(self._subagents.cleanup(state.subagent_stage_set))
        return warnings

    def record_agent_session(
        self,
        state: OptimizeSessionArtifactsState,
        *,
        label: str,
        session_id: str | None,
        agent: str,
    ) -> str | None:
        """Write one compact session-id record for one optimize launch."""
        return self._archives.record_agent_session(
            state.archive,
            label=label,
            session_id=session_id,
            agent=agent,
        )

    def cleanup_supervised_session(self, state: OptimizeSessionArtifactsState) -> list[str]:
        """Archive supervised artifacts first, then tear down live runtime files."""
        warnings: list[str] = []
        try:
            warnings.extend(self.archive(state))
        except Exception as exc:
            warnings.append(f"Failed to archive optimize supervised logs: {exc}")

        warnings.extend(self._cleanup_hidden_triton_agent_dir(state.hidden_triton_agent_dir))
        warnings.extend(self.cleanup_session(state))
        return warnings

    def cleanup_checked_session(self, state: OptimizeSessionArtifactsState) -> list[str]:
        """Archive checked-mode artifacts, then tear down the live runtime files."""
        warnings: list[str] = []
        try:
            warnings.extend(self.archive(state))
        except Exception as exc:
            warnings.append(f"Failed to archive optimize checked logs: {exc}")

        warnings.extend(self.cleanup_session(state))
        return warnings

    def describe_prepare_supervised_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = self._memory_files.describe_prepare(
            state.memory_file,
            description="supervised optimize guidance file",
        )
        messages.extend(self._describe_prepare_subagents(state.subagent_stage_set))
        if state.supervisor_report_path is not None:
            messages.append(f"wrote optimize supervisor report {state.supervisor_report_path}")
        return messages

    def describe_prepare_checked_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = self._memory_files.describe_prepare(
            state.memory_file,
            description="checked optimize guidance file",
        )
        messages.extend(self._describe_prepare_subagents(state.subagent_stage_set))
        return messages

    def describe_cleanup_session(self, state: OptimizeSessionArtifactsState) -> list[str]:
        messages = self._memory_files.describe_cleanup(state.memory_file)
        messages.extend(self._describe_cleanup_subagents(state.subagent_stage_set))
        return messages

    def describe_cleanup_supervised_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = [f"archiving supervised optimize logs to {state.run_archive_dir}"]
        if state.supervisor_report_path is not None:
            messages.append(f"removing temporary optimize file {state.supervisor_report_path}")
        if state.hidden_triton_agent_dir is not None:
            messages.append(
                "removing temporary optimize runtime directory tree "
                f"{state.hidden_triton_agent_dir}"
            )
        messages.extend(self.describe_cleanup_session(state))
        return messages

    def describe_cleanup_checked_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = [f"archiving checked optimize logs to {state.run_archive_dir}"]
        messages.extend(self.describe_cleanup_session(state))
        return messages

    def _prepare_subagents(
        self,
        *,
        agent_name: str,
        workdir: Path,
        language: str,
        optimize_target: str,
        enable_cann_ext_api: bool,
        enable_subagent: bool,
    ) -> SubagentStageSet | None:
        if not enable_subagent:
            return None
        definition = perf_diagnosis_subagent_definition(
            language=language,
            optimize_target=optimize_target,
            enable_cann_ext_api=enable_cann_ext_api,
        )
        if agent_name not in definition.supported_backends:
            supported_backends = definition.supported_backends
            supported = ", ".join(f"`{name}`" for name in supported_backends[:-1])
            if supported:
                supported = f"{supported}, and `{supported_backends[-1]}`"
            else:
                supported = f"`{supported_backends[-1]}`"
            raise RuntimeError(
                f"Optimize subagent staging only supports {supported}; got `{agent_name}`."
            )
        return self._subagents.prepare(agent_name, workdir, (definition,))

    def _describe_prepare_subagents(
        self,
        stage_set: SubagentStageSet | None,
    ) -> list[str]:
        if stage_set is None:
            return []
        files = [path for path in stage_set.created_paths if path.suffix in {".md", ".toml"}]
        if not files:
            return []
        return [f"staged optimize subagent file(s): {', '.join(str(path) for path in files)}"]

    def _describe_cleanup_subagents(
        self,
        stage_set: SubagentStageSet | None,
    ) -> list[str]:
        if stage_set is None:
            return []
        files = [path for path in stage_set.created_paths if path.suffix in {".md", ".toml"}]
        if not files:
            return []
        return [f"removing staged optimize subagent file(s): {', '.join(str(path) for path in files)}"]

    def _prepare_hidden_triton_agent_dir(self, workdir: Path) -> Path:
        hidden_triton_agent_dir = workdir / ".triton-agent"
        if hidden_triton_agent_dir.exists() and any(hidden_triton_agent_dir.iterdir()):
            raise RuntimeError(
                "Existing .triton-agent/ directory contains data; remove it before starting optimize."
            )
        hidden_triton_agent_dir.mkdir(parents=True, exist_ok=True)
        return hidden_triton_agent_dir

    def _cleanup_hidden_triton_agent_dir(self, hidden_triton_agent_dir: Path | None) -> list[str]:
        if hidden_triton_agent_dir is None:
            return []
        warnings: list[str] = []
        if hidden_triton_agent_dir.name != ".triton-agent":
            return [
                "Refusing to remove unexpected optimize runtime directory "
                f"{hidden_triton_agent_dir}"
            ]
        try:
            for root, dirs, files in os.walk(
                hidden_triton_agent_dir,
                topdown=False,
                followlinks=False,
            ):
                root_path = Path(root)
                for filename in files:
                    path = root_path / filename
                    try:
                        path.unlink()
                    except OSError as exc:
                        warnings.append(f"Failed to remove temporary optimize file {path}: {exc}")
                for dirname in dirs:
                    path = root_path / dirname
                    try:
                        path.rmdir()
                    except OSError as exc:
                        warnings.append(f"Failed to remove temporary optimize directory {path}: {exc}")
            try:
                hidden_triton_agent_dir.rmdir()
            except OSError as exc:
                warnings.append(
                    "Failed to remove temporary optimize directory "
                    f"{hidden_triton_agent_dir}: {exc}"
                )
        except OSError as exc:
            warnings.append(
                "Failed to remove temporary optimize directory "
                f"{hidden_triton_agent_dir}: {exc}"
            )
        return warnings
