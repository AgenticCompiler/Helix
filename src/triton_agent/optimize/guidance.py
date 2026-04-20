from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import List, Optional


@dataclass
class SharedOptimizeGuidanceState:
    guidance_path: Path
    backup_path: Optional[Path]
    created_guidance: bool
    archive_root: Path
    run_archive_dir: Path
    agent_sessions_path: Path


@dataclass
class OptimizeGuidanceState(SharedOptimizeGuidanceState):
    round_brief_path: Path
    supervisor_report_path: Path
    history_dir: Path
    shared_guidance_snapshot_path: Path
    created_paths: tuple[Path, ...]


class OptimizeGuidanceManager:
    def prepare_unsupervised_session(
        self,
        workdir: Path,
        *,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        agent_name: str,
        require_analysis: bool = False,
    ) -> SharedOptimizeGuidanceState:
        guidance_path = workdir / self._guidance_filename(agent_name)
        guidance_preexisting = guidance_path.exists()
        backup_path: Optional[Path] = None
        archive_root = workdir / "optimize-logs" / "triton-agent"
        run_archive_dir = archive_root / self._new_run_id()

        if guidance_preexisting:
            backup_path = self._next_backup_path(guidance_path)
            backup_path.write_bytes(guidance_path.read_bytes())

        guidance_path.write_text(
            self._render_unsupervised_guidance(
                guidance_filename=guidance_path.name,
                operator_path=operator_path,
                test_mode=test_mode,
                bench_mode=bench_mode,
                require_analysis=require_analysis,
            ),
            encoding="utf-8",
        )
        return SharedOptimizeGuidanceState(
            guidance_path=guidance_path,
            backup_path=backup_path,
            created_guidance=not guidance_preexisting,
            archive_root=archive_root,
            run_archive_dir=run_archive_dir,
            agent_sessions_path=run_archive_dir / "agent-sessions.jsonl",
        )

    def archive(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        archive_dir = state.run_archive_dir
        if archive_dir.exists():
            unexpected_paths = [
                path for path in archive_dir.iterdir() if path != state.agent_sessions_path
            ]
            if unexpected_paths:
                warnings.append(f"Refusing to overwrite existing optimize log archive at {archive_dir}")
                return warnings

        try:
            (archive_dir / "final").mkdir(parents=True, exist_ok=True)
            (archive_dir / "history").mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return [f"Failed to create optimize log archive directories under {archive_dir}: {exc}"]

        try:
            state.shared_guidance_snapshot_path.write_text(
                state.guidance_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.append(f"Failed to write shared guidance archive snapshot: {exc}")

        final_sources = (
            (state.round_brief_path, archive_dir / "final" / "round-brief.md"),
            (state.supervisor_report_path, archive_dir / "final" / "supervisor-report.md"),
        )
        for src, dest in final_sources:
            if not src.exists():
                warnings.append(f"Missing expected optimize handoff file at {src}")
                continue
            try:
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError as exc:
                warnings.append(f"Failed to archive optimize handoff file {src}: {exc}")

        if state.history_dir.exists():
            for src in sorted(state.history_dir.iterdir()):
                if not src.is_file():
                    continue
                dest = archive_dir / "history" / src.name
                try:
                    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                except OSError as exc:
                    warnings.append(f"Failed to archive optimize history file {src}: {exc}")
        return warnings

    def prepare_supervised_session(
        self,
        workdir: Path,
        agent_name: str,
        require_analysis: bool = False,
    ) -> OptimizeGuidanceState:
        guidance_path = workdir / self._guidance_filename(agent_name)
        guidance_preexisting = guidance_path.exists()
        backup_path: Optional[Path] = None

        if guidance_preexisting:
            backup_path = self._next_backup_path(guidance_path)
            backup_path.write_bytes(guidance_path.read_bytes())

        runtime_root = workdir / ".triton-agent"
        if runtime_root.exists() and any(runtime_root.iterdir()):
            raise RuntimeError(
                "Existing .triton-agent/ directory contains data; remove it before starting optimize."
            )
        runtime_root.mkdir(parents=True, exist_ok=True)
        round_brief_path = runtime_root / "round-brief.md"
        supervisor_report_path = runtime_root / "supervisor-report.md"
        history_dir = runtime_root / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        archive_root = workdir / "optimize-logs" / "triton-agent"
        run_id = self._new_run_id()
        run_archive_dir = archive_root / run_id
        shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"

        guidance_path.write_text(
            self._render_shared_guidance(
                guidance_filename=guidance_path.name,
                require_analysis=require_analysis,
            ),
            encoding="utf-8",
        )
        round_brief_path.write_text(
            "# Optimize Round Brief\n\nPending supervisor handoff.\n",
            encoding="utf-8",
        )
        supervisor_report_path.write_text(
            "# Optimize Supervisor Report\n\nPending first supervisor pass.\n",
            encoding="utf-8",
        )
        return OptimizeGuidanceState(
            guidance_path=guidance_path,
            backup_path=backup_path,
            created_guidance=not guidance_preexisting,
            archive_root=archive_root,
            run_archive_dir=run_archive_dir,
            agent_sessions_path=run_archive_dir / "agent-sessions.jsonl",
            round_brief_path=round_brief_path,
            supervisor_report_path=supervisor_report_path,
            history_dir=history_dir,
            shared_guidance_snapshot_path=shared_guidance_snapshot_path,
            created_paths=(
                guidance_path,
                round_brief_path,
                supervisor_report_path,
            ),
        )

    def cleanup_unsupervised_session(self, state: SharedOptimizeGuidanceState) -> list[str]:
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

    def record_agent_session(
        self,
        state: SharedOptimizeGuidanceState,
        *,
        role: str,
        session_id: str | None,
        agent: str,
    ) -> str | None:
        payload = {
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "role": role,
            "session_id": session_id or "unknown",
            "agent": agent,
        }
        try:
            state.agent_sessions_path.parent.mkdir(parents=True, exist_ok=True)
            with state.agent_sessions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError as exc:
            return f"Failed to record optimize agent session at {state.agent_sessions_path}: {exc}"
        return None

    def cleanup_supervised_session(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        try:
            warnings.extend(self.archive(state))
        except Exception as exc:
            warnings.append(f"Failed to archive optimize supervised logs: {exc}")

        for path in reversed(state.created_paths):
            if path == state.guidance_path and state.backup_path is not None:
                continue
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to remove temporary optimize file {path}: {exc}")

        runtime_root = state.history_dir.parent
        if runtime_root.name == ".triton-agent" and runtime_root.parent == state.guidance_path.parent:
            try:
                for root, dirs, files in os.walk(runtime_root, topdown=False, followlinks=False):
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
                    runtime_root.rmdir()
                except OSError as exc:
                    warnings.append(f"Failed to remove temporary optimize directory {runtime_root}: {exc}")
            except OSError as exc:
                warnings.append(f"Failed to remove temporary optimize directory {runtime_root}: {exc}")
        else:
            warnings.append(f"Refusing to remove unexpected optimize runtime directory {runtime_root}")

        warnings.extend(self.cleanup_unsupervised_session(state))
        return warnings

    def describe_prepare_unsupervised_session(
        self,
        state: SharedOptimizeGuidanceState,
    ) -> list[str]:
        messages: List[str] = []
        if state.backup_path is not None:
            messages.append(f"backed up workspace guidance file to {state.backup_path}")
        messages.append(f"wrote unsupervised optimize guidance file {state.guidance_path}")
        return messages

    def describe_prepare_supervised_session(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = []
        if state.backup_path is not None:
            messages.append(f"backed up workspace guidance file to {state.backup_path}")
        messages.append(f"wrote supervised optimize guidance file {state.guidance_path}")
        messages.append(f"wrote optimize round brief {state.round_brief_path}")
        messages.append(f"wrote optimize supervisor report {state.supervisor_report_path}")
        return messages

    def describe_cleanup_unsupervised_session(self, state: SharedOptimizeGuidanceState) -> list[str]:
        if state.backup_path is None:
            return [f"removing temporary guidance file {state.guidance_path}"]
        return [
            f"restoring workspace guidance file from {state.backup_path}",
            f"removing backup file {state.backup_path}",
        ]

    def describe_cleanup_supervised_session(self, state: OptimizeGuidanceState) -> list[str]:
        messages = [f"archiving supervised optimize logs to {state.run_archive_dir}"]
        for path in reversed(state.created_paths):
            if path == state.guidance_path and state.backup_path is not None:
                continue
            messages.append(f"removing temporary optimize file {path}")
        messages.append(
            f"removing temporary optimize runtime directory tree {state.history_dir.parent}"
        )
        messages.extend(self.describe_cleanup_unsupervised_session(state))
        return messages

    def _guidance_filename(self, agent_name: str) -> str:
        if agent_name == "claude":
            return "CLAUDE.md"
        return "AGENTS.md"

    def _new_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")

    def _next_backup_path(self, guidance_path: Path) -> Path:
        candidate = guidance_path.with_suffix(guidance_path.suffix + ".triton-agent.bak")
        counter = 1
        while candidate.exists():
            candidate = guidance_path.with_suffix(
                guidance_path.suffix + f".triton-agent.{counter}.bak"
            )
            counter += 1
        return candidate

    def _render_unsupervised_guidance(
        self,
        *,
        guidance_filename: str,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        require_analysis: bool,
    ) -> str:
        analysis_block = ""
        if require_analysis:
            analysis_block = (
                "- Require profiling or IR-backed evidence before the first code-changing round when possible.\n"
                "- Do not begin with blind tiling or launch-parameter search.\n"
            )

        return (
            f"# {guidance_filename}\n\n"
            "## Triton Agent Optimize Session\n\n"
            "This workspace is under an unsupervised optimize run.\n\n"
            "Own the end-to-end optimize session.\n"
            "Use the staged `triton-npu-optimize` skill as the workflow source of truth.\n"
            f"Use `{test_mode}` correctness validation for this optimize session.\n"
            f"Use `{bench_mode}` benchmark validation for this optimize session.\n"
            f"Optimize the operator at `{operator_path.name}`.\n"
            f"{analysis_block}"
        )

    def _render_shared_guidance(
        self,
        *,
        guidance_filename: str,
        require_analysis: bool,
    ) -> str:
        analysis_block = ""
        if require_analysis:
            analysis_block = (
                "- Require profiling or IR-backed evidence before the first code-changing round when possible.\n"
                "- Do not begin with blind tiling or launch-parameter search.\n"
            )

        return (
            f"# {guidance_filename}\n\n"
            "## Triton Agent Optimize Orchestration\n\n"
            "This workspace is under optimize orchestration.\n\n"
            "Use the staged workspace skills as the workflow source of truth.\n"
            "Role-specific behavior comes from the launch prompt.\n"
            "Use `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md` as live handoff files.\n"
            "Treat `baseline/` as the canonical optimize baseline.\n"
            "Use `compare-perf` as the authoritative source for round performance summaries.\n"
            f"{analysis_block}"
        )
