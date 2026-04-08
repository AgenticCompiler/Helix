from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class OptimizeGuidanceState:
    guidance_path: Path
    backup_path: Optional[Path]
    created_guidance: bool


class OptimizeGuidanceManager:
    def prepare(
        self,
        workdir: Path,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        agent_name: str,
    ) -> OptimizeGuidanceState:
        guidance_path = workdir / self._guidance_filename(agent_name)
        backup_path: Optional[Path] = None

        if guidance_path.exists():
            backup_path = self._next_backup_path(guidance_path)
            backup_path.write_text(guidance_path.read_text(encoding="utf-8"), encoding="utf-8")

        guidance_path.write_text(
            self._render_guidance(
                operator_path,
                test_mode=test_mode,
                bench_mode=bench_mode,
                guidance_filename=guidance_path.name,
            ),
            encoding="utf-8",
        )
        return OptimizeGuidanceState(
            guidance_path=guidance_path,
            backup_path=backup_path,
            created_guidance=True,
        )

    def cleanup(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        try:
            if state.created_guidance and state.guidance_path.exists():
                state.guidance_path.unlink()
        except OSError as exc:
            warnings.append(
                f"Failed to remove temporary guidance file {state.guidance_path}: {exc}"
            )

        if state.backup_path is not None:
            try:
                state.guidance_path.write_text(
                    state.backup_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                state.backup_path.unlink()
            except OSError as exc:
                warnings.append(
                    "Failed to restore original guidance file "
                    f"from {state.backup_path}: {exc}"
                )
        return warnings

    def describe_prepare(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = []
        if state.backup_path is not None:
            messages.append(f"backed up workspace guidance file to {state.backup_path}")
        messages.append(f"wrote optimize guidance file {state.guidance_path}")
        return messages

    def describe_cleanup(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = [f"removed temporary optimize guidance file {state.guidance_path}"]
        if state.backup_path is not None:
            messages.append(f"restored workspace guidance file from {state.backup_path}")
        return messages

    def _guidance_filename(self, agent_name: str) -> str:
        if agent_name == "claude":
            return "CLAUDE.md"
        return "AGENTS.md"

    def _next_backup_path(self, guidance_path: Path) -> Path:
        workdir = guidance_path.parent
        backup_stem = guidance_path.stem
        suffix = guidance_path.suffix
        for index in range(1000):
            candidate = workdir / (
                f".triton-agent-{backup_stem}.backup{suffix}"
                if index == 0
                else f".triton-agent-{backup_stem}.backup.{index}{suffix}"
            )
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not allocate guidance backup path in {workdir}")

    def _render_guidance(
        self,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        guidance_filename: str,
    ) -> str:
        return "\n".join(
            [
                f"# {guidance_filename}",
                "",
                "## Triton Agent Optimize Session",
                "",
                "## Mission",
                f"- Improve the Triton operator for Ascend NPU at `{operator_path}` while preserving correctness.",
                "- Work only in derived round directories.",
                "- Never edit the original operator in place.",
                "- Optimize only the existing NPU Triton operator implementation.",
                "- Preserve the Triton operator call path as the thing being optimized.",
                "- Do not delete or bypass the Triton operator call path.",
                "- Do not replace Triton operator calls with direct PyTorch operator calls or `torch.nn.Module` implementations.",
                "",
                "## Baseline",
                "- Treat the original operator as round 0.",
                "- Ensure correctness tests and benchmark cases exist before optimization starts.",
                f"- Use `{test_mode}` correctness validation for this optimize run.",
                f"- Use `{bench_mode}` benchmark validation for this optimize run.",
                "- If you need to generate or regenerate correctness tests, include multiple test cases that cover representative shapes, inputs, or edge conditions instead of a single case.",
                "- If you need to generate or regenerate benchmark cases, include multiple benchmark cases instead of a single case.",
                "- Record a baseline correctness and benchmark result before evaluating optimization wins.",
                "",
                "## Investigation",
                "- Start by consulting the staged `optimize` skill to understand the existing Triton NPU optimization rules and search patterns available in this repository.",
                "- Use the staged `ascend-npu-operator-profiler` skill when you need hotspot evidence, bottleneck measurements, or benchmark-driven profiling data to guide optimization choices.",
                "- Use the staged `ascend-operator-ir-analyzer` skill when you need to inspect Triton or Bisheng IR, confirm lowering behavior, or understand why an optimization did or did not take effect.",
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
