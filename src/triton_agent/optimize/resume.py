from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from triton_agent.bench_runner import parse_bench_metadata
from triton_agent.test_runner import parse_test_metadata


@dataclass(frozen=True)
class ResumeResolution:
    workspace_state: str
    resume_existing_session: bool
    test_mode: str | None
    bench_mode: str | None


@dataclass(frozen=True)
class WorkspaceInspection:
    state: str
    detail: str | None
    test_mode: str | None
    bench_mode: str | None


def classify_optimize_workspace(input_path: Path, workdir: Path) -> WorkspaceInspection:
    opt_note_path = workdir / "opt-note.md"
    has_opt_note = opt_note_path.exists()
    has_rounds = any(path.is_dir() for path in workdir.glob("opt-round-*"))
    test_harnesses = _existing_test_harnesses(input_path)
    bench_harness = input_path.with_name(f"bench_{input_path.stem}.py")
    has_bench = bench_harness.exists()

    if not (has_opt_note or has_rounds or test_harnesses or has_bench):
        return WorkspaceInspection(
            state="no-session",
            detail=None,
            test_mode=None,
            bench_mode=None,
        )

    if not has_opt_note:
        return WorkspaceInspection("partial-session", f"missing {opt_note_path.name}", None, None)
    if not has_rounds:
        return WorkspaceInspection("partial-session", "missing opt-round-* directory", None, None)
    if not test_harnesses:
        return WorkspaceInspection(
            "partial-session",
            f"missing generated test harness for {input_path.name}",
            None,
            None,
        )
    if len(test_harnesses) > 1:
        return WorkspaceInspection("partial-session", "multiple test harnesses exist", None, None)
    if not has_bench:
        return WorkspaceInspection("partial-session", f"missing {bench_harness.name}", None, None)

    test_mode = _parse_test_mode(test_harnesses[0])
    if test_mode is None:
        return WorkspaceInspection(
            "partial-session",
            f"unreadable test-mode metadata in {test_harnesses[0].name}",
            None,
            None,
        )

    bench_mode = _parse_bench_mode(bench_harness)
    if bench_mode is None:
        return WorkspaceInspection(
            "partial-session",
            f"unreadable bench-mode metadata in {bench_harness.name}",
            None,
            None,
        )

    return WorkspaceInspection("resumable-session", None, test_mode, bench_mode)


def resolve_optimize_resume(
    input_path: Path,
    workdir: Path,
    *,
    resume_mode: str,
    requested_test_mode: str | None,
    requested_bench_mode: str | None,
) -> ResumeResolution:
    inspection = classify_optimize_workspace(input_path, workdir)
    if resume_mode == "fresh":
        if inspection.state != "no-session":
            raise ValueError(f"resume fresh refused because optimize artifacts already exist in {workdir}")
        return ResumeResolution(
            workspace_state="no-session",
            resume_existing_session=False,
            test_mode=requested_test_mode or "differential",
            bench_mode=requested_bench_mode or "standalone",
        )

    if resume_mode == "continue":
        if requested_test_mode is not None:
            raise ValueError("--resume continue cannot be combined with --test-mode")
        if requested_bench_mode is not None:
            raise ValueError("--resume continue cannot be combined with --bench-mode")
        return _require_resumable_session(input_path, workdir, inspection)

    if inspection.state == "no-session":
        return ResumeResolution(
            workspace_state="no-session",
            resume_existing_session=False,
            test_mode=requested_test_mode or "differential",
            bench_mode=requested_bench_mode or "standalone",
        )
    if inspection.state == "partial-session":
        raise ValueError(f"resume auto found partial optimize state: {inspection.detail}")
    if requested_test_mode is not None:
        raise ValueError("--resume auto cannot be combined with --test-mode when reusing an existing session")
    if requested_bench_mode is not None:
        raise ValueError("--resume auto cannot be combined with --bench-mode when reusing an existing session")
    return ResumeResolution(
        workspace_state="resumable-session",
        resume_existing_session=True,
        test_mode=inspection.test_mode,
        bench_mode=inspection.bench_mode,
    )


def _require_resumable_session(
    input_path: Path,
    workdir: Path,
    inspection: WorkspaceInspection,
) -> ResumeResolution:
    opt_note_path = workdir / "opt-note.md"
    if not opt_note_path.exists():
        raise ValueError(f"resume continue requires existing opt-note.md: {opt_note_path}")
    if not any(path.is_dir() for path in workdir.glob("opt-round-*")):
        raise ValueError(
            f"resume continue requires at least one existing opt-round-* directory in {workdir}"
        )

    test_harnesses = _existing_test_harnesses(input_path)
    if not test_harnesses:
        raise ValueError(
            f"resume continue requires an existing generated test harness for {input_path.name}"
        )
    if len(test_harnesses) > 1:
        raise ValueError(
            "resume continue found multiple test harnesses. Keep only the active optimize test harness."
        )

    bench_harness = input_path.with_name(f"bench_{input_path.stem}.py")
    if not bench_harness.exists():
        raise ValueError(
            f"resume continue requires an existing generated benchmark harness: {bench_harness}"
        )

    if inspection.test_mode is None:
        raise ValueError(
            f"resume continue requires readable 'test-mode' metadata: {test_harnesses[0]}"
        )
    if inspection.bench_mode is None:
        raise ValueError(
            f"resume continue requires readable 'bench-mode' metadata: {bench_harness}"
        )

    return ResumeResolution(
        workspace_state="resumable-session",
        resume_existing_session=True,
        test_mode=inspection.test_mode,
        bench_mode=inspection.bench_mode,
    )


def _existing_test_harnesses(input_path: Path) -> list[Path]:
    candidates = [
        input_path.with_name(f"differential_test_{input_path.stem}.py"),
        input_path.with_name(f"test_{input_path.stem}.py"),
    ]
    return [path for path in candidates if path.exists()]


def _parse_test_mode(test_file: Path) -> str | None:
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        return None
    return str(mode)


def _parse_bench_mode(bench_file: Path) -> str | None:
    metadata = parse_bench_metadata(bench_file)
    mode = metadata.get("bench-mode")
    if mode not in {"standalone", "msprof"}:
        return None
    return str(mode)
