from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class OptimizeGuidanceState:
    agents_path: Path
    backup_path: Optional[Path]
    created_agents: bool


class OptimizeGuidanceManager:
    def prepare(
        self, workdir: Path, operator_path: Path, test_mode: str, bench_mode: str
    ) -> OptimizeGuidanceState:
        agents_path = workdir / "AGENTS.md"
        backup_path: Optional[Path] = None

        if agents_path.exists():
            backup_path = self._next_backup_path(workdir)
            backup_path.write_text(agents_path.read_text(encoding="utf-8"), encoding="utf-8")

        agents_path.write_text(
            self._render_guidance(operator_path, test_mode=test_mode, bench_mode=bench_mode),
            encoding="utf-8",
        )
        return OptimizeGuidanceState(
            agents_path=agents_path,
            backup_path=backup_path,
            created_agents=True,
        )

    def cleanup(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        try:
            if state.created_agents and state.agents_path.exists():
                state.agents_path.unlink()
        except OSError as exc:
            warnings.append(f"Failed to remove temporary AGENTS file {state.agents_path}: {exc}")

        if state.backup_path is not None:
            try:
                state.agents_path.write_text(
                    state.backup_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                state.backup_path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to restore original AGENTS file from {state.backup_path}: {exc}")
        return warnings

    def describe_prepare(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = []
        if state.backup_path is not None:
            messages.append(f"backed up workspace AGENTS file to {state.backup_path}")
        messages.append(f"wrote optimize guidance file {state.agents_path}")
        return messages

    def describe_cleanup(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = [f"removed temporary optimize guidance file {state.agents_path}"]
        if state.backup_path is not None:
            messages.append(f"restored workspace AGENTS file from {state.backup_path}")
        return messages

    def _next_backup_path(self, workdir: Path) -> Path:
        for index in range(1000):
            candidate = workdir / (
                ".triton-agent-AGENTS.backup.md"
                if index == 0
                else f".triton-agent-AGENTS.backup.{index}.md"
            )
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not allocate AGENTS backup path in {workdir}")

    def _render_guidance(self, operator_path: Path, test_mode: str, bench_mode: str) -> str:
        return "\n".join(
            [
                "# AGENTS.md",
                "",
                "## Triton Agent Optimize Session",
                "",
                "## Mission",
                f"- Improve the operator at `{operator_path}` while preserving correctness.",
                "- Work only in derived round directories.",
                "- Never edit the original operator in place.",
                "",
                "## Baseline",
                "- Treat the original operator as round 0.",
                "- Ensure correctness tests and benchmark cases exist before optimization starts.",
                f"- Use `{test_mode}` correctness validation for this optimize run.",
                f"- Use `{bench_mode}` benchmark validation for this optimize run.",
                "- Record a baseline correctness and benchmark result before evaluating optimization wins.",
                "",
                "## Gates",
                "- Run correctness validation before every benchmark check.",
                "- If correctness fails, repair the current round operator first.",
                "- Accept a round only when correctness passes and performance improves.",
                "",
                "## Search",
                "- Pick parents from validated candidates.",
                "- Do not assume the current best version is always the right parent.",
                "- Keep useful validated branches even when they are not the current best.",
                "",
                "## Records",
                "- Keep artifacts in `opt-round-N/`.",
                "- Update `attempts.md` throughout each round, not only at the end.",
                "- Write `summary.md` for every completed round.",
                "- Write optimization points and measured outcome in each summary.",
                "- Update `opt-note.md` after every completed round.",
            ]
        ) + "\n"
