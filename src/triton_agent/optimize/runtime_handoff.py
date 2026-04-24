from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class RuntimeHandoffState:
    """Live supervised handoff files that worker and supervisor share during a run."""

    runtime_root: Path
    round_brief_path: Path
    supervisor_report_path: Path
    history_dir: Path
    created_paths: tuple[Path, ...]


class RuntimeHandoffManager:
    """Owns the `.triton-agent/` runtime tree used only in supervised optimize mode."""

    def prepare(self, workdir: Path) -> RuntimeHandoffState:
        """Create a clean live handoff tree for the next supervised optimize session."""
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

        round_brief_path.write_text(
            "# Optimize Round Brief\n\nPending supervisor handoff.\n",
            encoding="utf-8",
        )
        supervisor_report_path.write_text(
            "# Optimize Supervisor Report\n\nPending first supervisor pass.\n",
            encoding="utf-8",
        )
        return RuntimeHandoffState(
            runtime_root=runtime_root,
            round_brief_path=round_brief_path,
            supervisor_report_path=supervisor_report_path,
            history_dir=history_dir,
            created_paths=(
                round_brief_path,
                supervisor_report_path,
            ),
        )

    def cleanup(self, state: RuntimeHandoffState) -> list[str]:
        """Remove the temporary runtime tree, but only when it matches the expected shape."""
        warnings: list[str] = []
        runtime_root = self._runtime_root(state)
        if runtime_root.name == ".triton-agent" and runtime_root.parent == state.history_dir.parent.parent:
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
        return warnings

    def describe_prepare(self, state: RuntimeHandoffState) -> list[str]:
        return [
            f"wrote optimize round brief {state.round_brief_path}",
            f"wrote optimize supervisor report {state.supervisor_report_path}",
        ]

    def describe_cleanup(self, state: RuntimeHandoffState) -> list[str]:
        runtime_root = self._runtime_root(state)
        return [
            f"removing temporary optimize file {state.round_brief_path}",
            f"removing temporary optimize file {state.supervisor_report_path}",
            f"removing temporary optimize runtime directory tree {runtime_root}",
        ]

    def _runtime_root(self, state: RuntimeHandoffState) -> Path:
        """Fallback for older tests or callers that only populated `history_dir`."""
        runtime_root = getattr(state, "runtime_root", None)
        if runtime_root is not None:
            return runtime_root
        return state.history_dir.parent
