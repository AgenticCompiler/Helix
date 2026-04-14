from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional

@dataclass
class SharedOptimizeGuidanceState:
    guidance_path: Path
    backup_path: Optional[Path]
    created_guidance: bool


@dataclass
class OptimizeGuidanceState(SharedOptimizeGuidanceState):
    round_brief_path: Path
    supervisor_report_path: Path
    history_dir: Path
    archive_root: Path
    run_archive_dir: Path
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
        )

    def archive(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        archive_dir = state.run_archive_dir
        if archive_dir.exists() and any(archive_dir.iterdir()):
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
        # A supervised optimize run temporarily stages:
        # 1. shared top-level guidance (`AGENTS.md`/`CLAUDE.md`)
        # 2. runtime handoff files under `.triton-agent/` that let worker and
        #    supervisor coordinate across round boundaries
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
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
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
            round_brief_path=round_brief_path,
            supervisor_report_path=supervisor_report_path,
            history_dir=history_dir,
            archive_root=archive_root,
            run_archive_dir=run_archive_dir,
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

    def cleanup_supervised_session(self, state: OptimizeGuidanceState) -> list[str]:
        warnings: list[str] = []
        # Cleanup is intentionally two-phase: archive first so the final worker /
        # supervisor conversation survives, then remove temporary runtime files
        # and restore the original workspace guidance file.
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
        messages: List[str] = []
        if state.backup_path is not None:
            messages.append(f"restoring workspace guidance file from {state.backup_path}")
        else:
            messages.append(f"removing temporary optimize guidance file {state.guidance_path}")
        return messages

    def describe_cleanup_supervised_session(self, state: OptimizeGuidanceState) -> list[str]:
        messages: List[str] = []
        messages.append(
            f"archiving supervised optimize logs to {state.run_archive_dir} before runtime cleanup"
        )
        for path in reversed(state.created_paths):
            if path == state.guidance_path:
                continue
            messages.append(f"removing temporary optimize file {path}")
        runtime_root = state.history_dir.parent
        messages.append(f"removing temporary optimize runtime directory tree {runtime_root}")
        messages.extend(self.describe_cleanup_unsupervised_session(state))
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

    def _render_shared_guidance(self, *, guidance_filename: str, require_analysis: bool = False) -> str:
        lines = [
            f"# {guidance_filename}",
            "",
            "## Triton Agent Optimize Orchestration",
            "",
            "- This workspace is under optimize orchestration.",
            "- Use the staged workspace skills as the workflow source of truth.",
            "- Role-specific behavior comes from the launch prompt.",
            "- Use `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md` as live handoff files.",
            "- Supervisor repair is limited to metadata derived from existing facts.",
            "- Do not fabricate benchmark, profiler, or IR evidence.",
            "- Treat `baseline/` as the canonical optimize baseline for this workspace.",
            "- Use `compare-perf` as the authoritative source for claimed speedups and benchmark deltas.",
        ]
        if require_analysis:
            lines.extend(
                [
                    "- Require profiling or IR-backed evidence before the first code-changing round when possible.",
                    "- Do not begin with blind tiling or launch-parameter search.",
                ]
            )
        lines.extend(["", ""])
        return "\n".join(lines)

    def _render_unsupervised_guidance(
        self,
        *,
        guidance_filename: str,
        operator_path: Path,
        test_mode: str,
        bench_mode: str,
        require_analysis: bool = False,
    ) -> str:
        lines = [
            f"# {guidance_filename}",
            "",
            "## Triton Agent Optimize Session",
            "",
            "- This workspace is under an unsupervised optimize run.",
            "- Own the end-to-end optimize session.",
            "- Use the staged workspace skills as the workflow source of truth.",
            f"- Optimize the existing Triton Ascend NPU operator at `{operator_path}`.",
            "- Do not edit the original operator in place; work through `baseline/` and `opt-round-*` artifacts.",
            "- Preserve the Triton operator call path as the thing being optimized.",
            "- Do not delete or bypass the Triton operator call path.",
            "- Do not replace Triton operator calls with direct PyTorch operators or `torch.nn.Module` implementations.",
            "- Establish or reuse `baseline/` before creating `opt-round-1`.",
            "- Use `baseline/perf.txt` for canonical performance comparisons.",
            "- Use `compare-perf` as the authoritative source for claimed speedups and benchmark deltas.",
            "- Check whether correctness tests and benchmark cases already exist before generating anything new.",
            f"- Use `{test_mode}` correctness validation for this optimize run.",
            f"- Use `{bench_mode}` benchmark validation for this optimize run.",
            "- If you need to generate or regenerate correctness tests, include multiple representative test cases instead of a single case.",
            "- If you need to generate or regenerate benchmark cases, include multiple benchmark cases instead of a single case.",
            "- Write a short diagnosis summary before the first code-changing round.",
            "- Record each round's hypothesis and rationale in `attempts.md` before editing code.",
            "- Keep `summary.md` and `opt-note.md` up to date before stopping.",
            "- Use the staged `optimize` skill first, then use the staged profiler and IR-analysis skills when benchmark numbers alone are not enough.",
            "- If you skip profiling or IR capture for a round, explain why the existing evidence is already sufficient.",
        ]
        if require_analysis:
            lines.extend(
                [
                    "- Before the first code-changing round, gather profiling or IR-backed evidence.",
                    "- If one analysis path is unavailable, record why and what evidence replaced it.",
                    "- Do not begin with blind tiling or launch-parameter search.",
                ]
            )
        lines.extend(["", ""])
        return "\n".join(lines)
