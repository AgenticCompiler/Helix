from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from triton_agent.log_analysis import write_agent_audit
from triton_agent.optimize.archive import ArchiveManager, ArchiveState
from triton_agent.optimize.memory_file import MemoryFileManager, MemoryFileState


@dataclass
class SharedOptimizeSessionArtifactsState:
    """Session-scoped artifacts shared by supervised and continuous optimize runs."""

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
    def run_archive_dir(self) -> Path:
        return self.archive.run_archive_dir

    @property
    def agent_sessions_path(self) -> Path:
        return self.archive.agent_sessions_path

    @property
    def otel_trace_path(self) -> Path:
        return self.archive.otel_trace_path

    @property
    def otel_summary_path(self) -> Path:
        return self.archive.otel_summary_path

    @property
    def agent_audit_path(self) -> Path:
        return self.archive.agent_audit_path


@dataclass
class OptimizeSessionArtifactsState(SharedOptimizeSessionArtifactsState):
    """Full artifact bundle for multi-invocation optimize runs."""

    hidden_triton_agent_dir: Path | None = None
    supervisor_report_path: Path | None = None
    supervisor_history_dir: Path | None = None

    @property
    def shared_guidance_snapshot_path(self) -> Path | None:
        """Archive destination for the shared memory-file snapshot, when enabled."""
        return self.archive.shared_guidance_snapshot_path

    @property
    def created_paths(self) -> tuple[Path, ...]:
        created: list[Path] = []
        if self.supervisor_report_path is not None:
            created.append(self.supervisor_report_path)
        return tuple(created)


class OptimizeSessionArtifactsManager:
    """Thin facade that coordinates memory-file, supervised runtime files, and archive artifacts."""

    def __init__(
        self,
        memory_files: MemoryFileManager | None = None,
        archives: ArchiveManager | None = None,
    ) -> None:
        self._memory_files = memory_files or MemoryFileManager()
        self._archives = archives or ArchiveManager()

    def prepare_continuous_session(
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
        optimize_knowledge_skill_name: str | None = None,
    ) -> SharedOptimizeSessionArtifactsState:
        """Prepare only the artifacts needed by a single-agent optimize session."""
        archive_state = self._archives.prepare(workdir)
        memory_file_state = self._memory_files.prepare_continuous(
            workdir,
            operator_path=operator_path,
            test_mode=test_mode,
            bench_mode=bench_mode,
            agent_name=agent_name,
            optimize_target=optimize_target,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
            optimize_knowledge_skill_name=optimize_knowledge_skill_name,
        )
        return SharedOptimizeSessionArtifactsState(
            memory_file=memory_file_state,
            archive=archive_state,
        )

    def prepare_checked_session(
        self,
        workdir: Path,
        agent_name: str,
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_cann_ext_api: bool = False,
        optimize_knowledge_skill_name: str | None = None,
    ) -> OptimizeSessionArtifactsState:
        """Prepare artifacts for checked optimize without a live handoff file."""
        archive_state = self._archives.prepare(workdir, include_shared_guidance_snapshot=True)
        memory_file_state = self._memory_files.prepare_round_gated(
            workdir,
            agent_name=agent_name,
            optimize_target=optimize_target,
            include_supervisor_handoff=False,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
            optimize_knowledge_skill_name=optimize_knowledge_skill_name,
        )
        return OptimizeSessionArtifactsState(
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
        memory_file_state = self._memory_files.prepare_shared(
            workdir,
            agent_name=agent_name,
            optimize_target=optimize_target,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_cann_ext_api=enable_cann_ext_api,
            optimize_knowledge_skill_name=optimize_knowledge_skill_name,
        )
        return OptimizeSessionArtifactsState(
            memory_file=memory_file_state,
            archive=archive_state,
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

    def cleanup_continuous_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
    ) -> list[str]:
        """Remove or restore the temporary top-level memory file for an optimize run."""
        warnings = self._write_agent_audit(state)
        warnings.extend(self._memory_files.cleanup(state.memory_file))
        return warnings

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

        warnings.extend(self._cleanup_hidden_triton_agent_dir(state.hidden_triton_agent_dir))
        warnings.extend(self.cleanup_continuous_session(state))
        return warnings

    def _write_agent_audit(self, state: SharedOptimizeSessionArtifactsState) -> list[str]:
        workdir = state.archive.workdir
        return write_agent_audit(workdir=workdir, archive=state.archive)

    def cleanup_checked_session(self, state: OptimizeSessionArtifactsState) -> list[str]:
        """Archive checked-mode artifacts, then tear down the live runtime files."""
        warnings: list[str] = []
        try:
            warnings.extend(self.archive(state))
        except Exception as exc:
            warnings.append(f"Failed to archive optimize checked logs: {exc}")

        warnings.extend(self.cleanup_continuous_session(state))
        return warnings

    def describe_prepare_continuous_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
    ) -> list[str]:
        return self._memory_files.describe_prepare(
            state.memory_file,
            description="continuous optimize guidance file",
        )

    def describe_prepare_supervised_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = self._memory_files.describe_prepare(
            state.memory_file,
            description="supervised optimize guidance file",
        )
        if state.supervisor_report_path is not None:
            messages.append(f"wrote optimize supervisor report {state.supervisor_report_path}")
        return messages

    def describe_prepare_checked_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        return self._memory_files.describe_prepare(
            state.memory_file,
            description="checked optimize guidance file",
        )

    def describe_cleanup_continuous_session(
        self,
        state: SharedOptimizeSessionArtifactsState,
    ) -> list[str]:
        return self._memory_files.describe_cleanup(state.memory_file)

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
        messages.extend(self.describe_cleanup_continuous_session(state))
        return messages

    def describe_cleanup_checked_session(
        self,
        state: OptimizeSessionArtifactsState,
    ) -> list[str]:
        messages = [f"archiving checked optimize logs to {state.run_archive_dir}"]
        messages.extend(self.describe_cleanup_continuous_session(state))
        return messages

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
