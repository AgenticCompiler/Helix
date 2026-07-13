from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Optional

from helix.trace.core import new_trace_run_id


@dataclass
class ArchiveState:
    """Paths for one optimize run archive under `helix-logs/{run_id}/`."""

    run_archive_dir: Path
    shared_guidance_snapshot_path: Optional[Path] = None

    @property
    def run_id(self) -> str:
        return self.run_archive_dir.name

    @property
    def workdir(self) -> Path:
        return self.run_archive_dir.parent.parent

    @property
    def show_output_path(self) -> Path:
        return self.run_archive_dir / "show-output.log"

    def trace_path(self, label: str) -> Path:
        return self.run_archive_dir / f"trace-{label}.jsonl"

    def trace_summary_path(self, label: str) -> Path:
        return self.run_archive_dir / f"trace-{label}.summary.json"

    def agent_session_path(self, label: str) -> Path:
        return self.run_archive_dir / f"agent-session-{label}.json"

    @property
    def supervisor_handoffs_dir(self) -> Path:
        return self.run_archive_dir / "supervisor-handoffs"


class ArchiveManager:
    """Owns run archive layout, handoff snapshots, and session-id recording."""

    def __init__(self, run_id_factory: Callable[[], str] | None = None) -> None:
        self._run_id_factory = run_id_factory or self._new_run_id

    def prepare(self, workdir: Path, *, include_shared_guidance_snapshot: bool = False) -> ArchiveState:
        """Describe where this optimize run will archive logs and metadata."""
        run_archive_dir = workdir / "helix-logs" / self._run_id_factory()
        shared_guidance_snapshot_path = None
        if include_shared_guidance_snapshot:
            shared_guidance_snapshot_path = run_archive_dir / "shared-guidance.md"
        return ArchiveState(
            run_archive_dir=run_archive_dir,
            shared_guidance_snapshot_path=shared_guidance_snapshot_path,
        )

    def archive(
        self,
        state: ArchiveState,
        *,
        guidance_path: Path,
        supervisor_report_path: Path | None,
        handoff_dir: Path | None,
    ) -> list[str]:
        """Copy final multi-invocation outputs into the per-run archive directory."""
        warnings: list[str] = []
        archive_dir = state.run_archive_dir
        # Runtime files created during checked/supervised phases (show-output,
        # otel traces, tool-traces) are expected to already exist in the run
        # directory. Only treat truly unexpected children as a stale archive.
        _EXPECTED_NAMES = frozenset({
            "show-output.log", "tool-traces.jsonl",
            "supervisor-handoffs", "round-timings", "shared-guidance.md", "supervisor-report.md",
        })
        if archive_dir.exists():
            unexpected_paths = [
                path for path in archive_dir.iterdir()
                if path.name not in _EXPECTED_NAMES
                and not path.name.startswith("show-output-")
                and not (path.name.startswith("trace-") and (path.name.endswith(".jsonl") or path.name.endswith(".summary.json")))
                and not (path.name.startswith("agent-session-") and path.name.endswith(".json"))
            ]
            if unexpected_paths:
                warnings.append(f"Refusing to overwrite existing optimize log archive at {archive_dir}")
                return warnings

        handoff_files: list[Path] = []
        if handoff_dir is not None and handoff_dir.exists():
            handoff_files = [src for src in sorted(handoff_dir.iterdir()) if src.is_file()]

        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
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

        if supervisor_report_path is not None:
            dest = archive_dir / "supervisor-report.md"
            if supervisor_report_path.exists():
                try:
                    dest.write_text(supervisor_report_path.read_text(encoding="utf-8"), encoding="utf-8")
                except OSError as exc:
                    warnings.append(f"Failed to archive optimize handoff file {supervisor_report_path}: {exc}")
            else:
                warnings.append(f"Missing expected optimize handoff file at {supervisor_report_path}")

        if handoff_files:
            try:
                state.supervisor_handoffs_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                warnings.append(f"Failed to create optimize handoff archive directory under {archive_dir}: {exc}")
                return warnings
            for src in handoff_files:
                dest = state.supervisor_handoffs_dir / src.name
                try:
                    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                except OSError as exc:
                    warnings.append(f"Failed to archive optimize handoff file {src}: {exc}")
        return warnings

    def record_agent_session(
        self,
        state: ArchiveState,
        *,
        label: str,
        session_id: str | None,
        agent: str,
    ) -> str | None:
        """Write one compact session record for a single optimize agent launch."""
        payload = {
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "session_id": session_id or "unknown",
            "agent": agent,
        }
        path = state.agent_session_path(label)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            return f"Failed to record optimize agent session at {path}: {exc}"
        return None

    def _new_run_id(self) -> str:
        return new_trace_run_id(prefix="optimize")
