from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Optional


@dataclass
class ArchiveState:
    """Paths for one optimize run archive under `triton-agent-logs/triton-agent/`."""

    archive_root: Path
    run_archive_dir: Path
    agent_sessions_path: Path
    shared_guidance_snapshot_path: Optional[Path] = None


class ArchiveManager:
    """Owns run archive layout, handoff snapshots, and session-id recording."""

    def __init__(self, run_id_factory: Callable[[], str] | None = None) -> None:
        self._run_id_factory = run_id_factory or self._new_run_id

    def prepare(self, workdir: Path, *, include_shared_guidance_snapshot: bool = False) -> ArchiveState:
        """Describe where this optimize run will archive logs and metadata."""
        archive_root = workdir / "triton-agent-logs" / "triton-agent"
        run_archive_dir = archive_root / self._run_id_factory()
        shared_guidance_snapshot_path = None
        if include_shared_guidance_snapshot:
            shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"
        return ArchiveState(
            archive_root=archive_root,
            run_archive_dir=run_archive_dir,
            agent_sessions_path=run_archive_dir / "agent-sessions.jsonl",
            shared_guidance_snapshot_path=shared_guidance_snapshot_path,
        )

    def archive(
        self,
        state: ArchiveState,
        *,
        guidance_path: Path,
        round_brief_path: Path,
        supervisor_report_path: Path,
        history_dir: Path,
    ) -> list[str]:
        """Copy final supervised outputs into the per-run archive directory."""
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

        if state.shared_guidance_snapshot_path is not None:
            try:
                state.shared_guidance_snapshot_path.write_text(
                    guidance_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            except OSError as exc:
                warnings.append(f"Failed to write shared guidance archive snapshot: {exc}")

        final_sources = (
            (round_brief_path, archive_dir / "final" / "round-brief.md"),
            (supervisor_report_path, archive_dir / "final" / "supervisor-report.md"),
        )
        for src, dest in final_sources:
            if not src.exists():
                warnings.append(f"Missing expected optimize handoff file at {src}")
                continue
            try:
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError as exc:
                warnings.append(f"Failed to archive optimize handoff file {src}: {exc}")

        if history_dir.exists():
            for src in sorted(history_dir.iterdir()):
                if not src.is_file():
                    continue
                dest = archive_dir / "history" / src.name
                try:
                    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                except OSError as exc:
                    warnings.append(f"Failed to archive optimize history file {src}: {exc}")
        return warnings

    def record_agent_session(
        self,
        state: ArchiveState,
        *,
        role: str,
        session_id: str | None,
        agent: str,
    ) -> str | None:
        """Append one JSONL session record without disturbing the rest of the archive."""
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

    def _new_run_id(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
