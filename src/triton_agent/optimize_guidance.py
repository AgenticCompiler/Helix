from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class OptimizeGuidanceState:
    guidance_path: Path
    backup_path: Optional[Path]
    created_guidance: bool
    role_dir: Path
    worker_brief_path: Path
    supervisor_brief_path: Path
    round_brief_path: Path
    supervisor_report_path: Path
    created_paths: tuple[Path, ...]


class OptimizeGuidanceManager:
    def prepare(
        self,
        workdir: Path,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        agent_name: str,
        require_analysis: bool = False,
    ) -> OptimizeGuidanceState:
        guidance_path = workdir / self._guidance_filename(agent_name)
        backup_path: Optional[Path] = None

        if guidance_path.exists():
            backup_path = self._next_backup_path(guidance_path)
            backup_path.write_text(guidance_path.read_text(encoding="utf-8"), encoding="utf-8")

        role_dir = workdir / ".triton-agent" / "roles"
        worker_brief_path = role_dir / "optimize-worker.md"
        supervisor_brief_path = role_dir / "optimize-supervisor.md"
        round_brief_path = workdir / ".triton-agent" / "round-brief.md"
        supervisor_report_path = workdir / ".triton-agent" / "supervisor-report.md"

        role_dir.mkdir(parents=True, exist_ok=True)
        guidance_path.write_text(
            self._render_shared_guidance(guidance_filename=guidance_path.name),
            encoding="utf-8",
        )
        worker_brief_path.write_text(
            self._render_worker_brief(
                operator_path,
                test_mode=test_mode,
                bench_mode=bench_mode,
                require_analysis=require_analysis,
            ),
            encoding="utf-8",
        )
        supervisor_brief_path.write_text(
            self._render_supervisor_brief(require_analysis=require_analysis),
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
            created_guidance=True,
            role_dir=role_dir,
            worker_brief_path=worker_brief_path,
            supervisor_brief_path=supervisor_brief_path,
            round_brief_path=round_brief_path,
            supervisor_report_path=supervisor_report_path,
            created_paths=(
                guidance_path,
                worker_brief_path,
                supervisor_brief_path,
                round_brief_path,
                supervisor_report_path,
            ),
        )

    def cleanup(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        for path in reversed(state.created_paths):
            if path == state.guidance_path and state.backup_path is not None:
                continue
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to remove temporary optimize file {path}: {exc}")

        for directory in (state.role_dir, state.role_dir.parent):
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            except OSError as exc:
                warnings.append(f"Failed to remove temporary optimize directory {directory}: {exc}")

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
                state.guidance_path.write_text(
                    state.backup_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
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
        messages.append(f"wrote shared optimize guidance file {state.guidance_path}")
        messages.append(f"wrote optimize worker brief {state.worker_brief_path}")
        messages.append(f"wrote optimize supervisor brief {state.supervisor_brief_path}")
        return messages

    def describe_cleanup(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = []
        for path in reversed(state.created_paths):
            if path == state.guidance_path:
                continue
            messages.append(f"removing temporary optimize file {path}")
        for directory in (state.role_dir, state.role_dir.parent):
            messages.append(f"removing temporary optimize directory {directory} when empty")
        if state.backup_path is not None:
            messages.append(f"restoring workspace guidance file from {state.backup_path}")
        else:
            messages.append(f"removing temporary optimize guidance file {state.guidance_path}")
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

    def _render_shared_guidance(self, *, guidance_filename: str) -> str:
        return "\n".join(
            [
                f"# {guidance_filename}",
                "",
                "## Triton Agent Optimize Orchestration",
                "",
                "- This workspace is under optimize orchestration.",
                "- Use the staged workspace skills as the workflow source of truth.",
                "- Read the role brief for this invocation before acting.",
                "- Worker and supervisor roles are assigned by the launch prompt.",
                "- Do not put worker-only or supervisor-only role assignment in this shared guidance file.",
                "- Supervisor repair is limited to metadata derived from existing facts.",
                "- Do not fabricate benchmark, profiler, or IR evidence.",
                "",
            ]
        )

    def _render_worker_brief(
        self,
        operator_path: Path,
        *,
        test_mode: str,
        bench_mode: str,
        require_analysis: bool = False,
    ) -> str:
        lines = [
            "# Optimize Worker Role Brief",
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
            "- Check whether correctness tests and benchmark cases already exist before generating anything new.",
            "- Do not regenerate them when reusable harnesses are already present.",
            f"- Use `{test_mode}` correctness validation for this optimize run.",
            f"- Use `{bench_mode}` benchmark validation for this optimize run.",
            "- If you need to generate or regenerate correctness tests, include multiple test cases that cover representative shapes, inputs, or edge conditions instead of a single case.",
            "- If you need to generate or regenerate benchmark cases, include multiple benchmark cases instead of a single case.",
            "- Record a baseline correctness and benchmark result before evaluating optimization wins.",
            "- Write a short diagnosis summary before the first code-changing round.",
            "",
            "## Investigation",
            "- Start by consulting the staged `optimize` skill to understand the existing Triton NPU optimization rules and search patterns available in this repository.",
            "- Use the staged `ascend-npu-operator-profiler` skill when you need hotspot evidence, bottleneck measurements, or benchmark-driven profiling data to guide optimization choices.",
            "- Use the staged `ascend-operator-ir-analyzer` skill when you need to inspect Triton or Bisheng IR, confirm lowering behavior, or understand why an optimization did or did not take effect.",
            "- State the hypothesis, why it may help, and what evidence supports it before editing code.",
            "- If you skip profiling or IR capture for a round, explain why the existing evidence is sufficient.",
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
            "- Keep round-local benchmark evidence in `perf.txt` or a copied round perf file.",
            "- Keep profiler evidence under `opt-round-N/profile/` when collected.",
            "- Keep IR archives under `opt-round-N/ir/` when collected.",
            "- Update `attempts.md` throughout each round, not only at the end.",
            "- Write `summary.md` for every completed round.",
            "- Write optimization points and measured outcome in each summary.",
            "- Update `opt-note.md` after every completed round.",
            "- Record `Geomean speedup` and `Total speedup` in the final `## Overall Summary` block.",
            "- Use `Geomean speedup` as the headline metric when deciding the final best round.",
            "- Leave `opt-note.md` ending with one `## Overall Summary` block that states the final best round, the overall benchmark outcome, the speedup metrics, any useful validated branches, and the recommended next step.",
        ]
        if require_analysis:
            lines.extend(
                [
                    "",
                    "## Strict Analysis",
                    "- Before the first code-changing round, gather profiling or IR-backed evidence.",
                    "- If one analysis path is unavailable, record why and explain what evidence replaces it.",
                    "- Do not begin with blind tiling or launch-parameter search.",
                ]
            )
        return "\n".join(lines) + "\n"

    def _render_supervisor_brief(self, *, require_analysis: bool = False) -> str:
        lines = [
            "# Optimize Supervisor Role Brief",
            "",
            "## Supervisor Mission",
            "- This invocation is an audit and handoff pass for a completed optimize round.",
            "- Do not perform open-ended optimization work.",
            "- Repair metadata only when the underlying evidence already exists.",
            "- Emit a gate result for the completed round.",
            "- Produce the next-round brief only when continuation is allowed.",
            "- Block the session when required benchmark, profiler, IR, or correctness evidence is missing.",
        ]
        if require_analysis:
            lines.extend(
                [
                    "",
                    "## Strict Analysis",
                    "- Require existing profiling or IR-backed evidence, or require the next worker round to explain why the remaining evidence is sufficient.",
                ]
            )
        return "\n".join(lines) + "\n"
