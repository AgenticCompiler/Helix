from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triton_agent.optimize.archive import ArchiveManager, ArchiveState
from triton_agent.optimize.memory_file import MemoryFileManager, MemoryFileState
from triton_agent.optimize.runtime_handoff import RuntimeHandoffManager, RuntimeHandoffState


@dataclass
class SharedOptimizeSessionArtifactsState:
    """Session-scoped artifacts shared by supervised and unsupervised optimize runs."""

    memory_file: MemoryFileState
    archive: ArchiveState

    @property
    def guidance_path(self) -> Path:
        """Compatibility view of the temporary top-level memory file path."""
        return self.memory_file.guidance_path

    @property
    def backup_path(self) -> Path | None:
        return self.memory_file.backup_path

    @property
    def created_guidance(self) -> bool:
        return self.memory_file.created_guidance

    @property
    def archive_root(self) -> Path:
        return self.archive.archive_root

    @property
    def run_archive_dir(self) -> Path:
        return self.archive.run_archive_dir

    @property
    def agent_sessions_path(self) -> Path:
        return self.archive.agent_sessions_path


@dataclass
class OptimizeSessionArtifactsState(SharedOptimizeSessionArtifactsState):
    """Full artifact bundle for supervised optimize runs."""

    runtime_handoff: RuntimeHandoffState

    @property
    def runtime_root(self) -> Path:
        """Root of the live `.triton-agent/` handoff tree."""
        return self.runtime_handoff.runtime_root

    @property
    def round_brief_path(self) -> Path:
        return self.runtime_handoff.round_brief_path

    @property
    def supervisor_report_path(self) -> Path:
        return self.runtime_handoff.supervisor_report_path

    @property
    def history_dir(self) -> Path:
        return self.runtime_handoff.history_dir

    @property
    def shared_guidance_snapshot_path(self) -> Path | None:
        """Archive destination for the shared memory-file snapshot, when enabled."""
        return self.archive.shared_guidance_snapshot_path

    @property
    def created_paths(self) -> tuple[Path, ...]:
        return self.runtime_handoff.created_paths


class OptimizeSessionArtifactsManager:
    """Thin facade that coordinates memory-file, runtime-handoff, and archive artifacts."""

    def __init__(
        self,
        memory_files: MemoryFileManager | None = None,
        runtime_handoffs: RuntimeHandoffManager | None = None,
        archives: ArchiveManager | None = None,
    ) -> None:
        self._memory_files = memory_files or MemoryFileManager()
        self._runtime_handoffs = runtime_handoffs or RuntimeHandoffManager()
        self._archives = archives or ArchiveManager()

    def prepare_unsupervised_session(
        self,
        workdir: Path,
        *,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        agent_name: str,
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
    ) -> SharedOptimizeSessionArtifactsState:
        """Prepare only the artifacts needed by a single-agent optimize session."""
        archive_state = self._archives.prepare(workdir)
        memory_file_state = self._memory_files.prepare_unsupervised(
            workdir,
            operator_path=operator_path,
            test_mode=test_mode,
            bench_mode=bench_mode,
            agent_name=agent_name,
            optimize_target=optimize_target,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
        )
        return SharedOptimizeSessionArtifactsState(
            memory_file=memory_file_state,
            archive=archive_state,
        )

    def prepare_supervised_session(
        self,
        workdir: Path,
        agent_name: str,
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
    ) -> OptimizeSessionArtifactsState:
        """Prepare the full artifact set used by worker/supervisor orchestration."""
        runtime_handoff_state = self._runtime_handoffs.prepare(workdir)
        archive_state = self._archives.prepare(
            workdir,
            include_shared_guidance_snapshot=True,
        )
        memory_file_state = self._memory_files.prepare_shared(
            workdir,
            agent_name=agent_name,
            optimize_target=optimize_target,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
        )
        return OptimizeSessionArtifactsState(
            memory_file=memory_file_state,
            archive=archive_state,
            runtime_handoff=runtime_handoff_state,
        )

    def archive(self, state: OptimizeSessionArtifactsState) -> list[str]:
        """Persist the final supervised handoff files and history into the run archive."""
        return self._archives.archive(
            state.archive,
            guidance_path=state.guidance_path,
            round_brief_path=state.round_brief_path,
            supervisor_report_path=state.supervisor_report_path,
            history_dir=state.history_dir,
        )

    def cleanup_unsupervised_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
    ) -> list[str]:
        """Remove or restore the temporary top-level memory file for an optimize run."""
        return self._memory_files.cleanup(state.memory_file)

    def record_agent_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
        *,
        role: str,
        session_id: str | None,
        agent: str,
    ) -> str | None:
        """Append a compact session-id record to the optimize archive."""
        return self._archives.record_agent_session(
            state.archive,
            role=role,
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

        warnings.extend(self._runtime_handoffs.cleanup(state.runtime_handoff))
        warnings.extend(self.cleanup_unsupervised_session(state))
        return warnings

    def describe_prepare_unsupervised_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
    ) -> list[str]:
        return self._memory_files.describe_prepare(
            state.memory_file,
            description="unsupervised optimize guidance file",
        )

    def describe_prepare_supervised_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = self._memory_files.describe_prepare(
            state.memory_file,
            description="supervised optimize guidance file",
        )
        messages.extend(self._runtime_handoffs.describe_prepare(state.runtime_handoff))
        return messages

    def describe_cleanup_unsupervised_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
    ) -> list[str]:
        return self._memory_files.describe_cleanup(state.memory_file)

    def describe_cleanup_supervised_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = [f"archiving supervised optimize logs to {state.run_archive_dir}"]
        messages.extend(self._runtime_handoffs.describe_cleanup(state.runtime_handoff))
        messages.extend(self.describe_cleanup_unsupervised_session(state))
        return messages
