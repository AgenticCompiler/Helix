from __future__ import annotations

from pathlib import Path

from hook_runtime.optimize import resume as _runtime_resume

ResumeResolution = _runtime_resume.ResumeResolution
WorkspaceInspection = _runtime_resume.WorkspaceInspection


def classify_optimize_workspace(input_path: Path, workdir: Path) -> WorkspaceInspection:
    return _runtime_resume.classify_optimize_workspace(input_path, workdir)


def reset_optimize_workspace(input_path: Path, workdir: Path) -> None:
    _runtime_resume.reset_optimize_workspace(input_path, workdir)


def resolve_optimize_resume(
    input_path: Path,
    workdir: Path,
    *,
    resume_mode: str,
    reset_optimize: bool,
    requested_test_mode: str | None,
    requested_bench_mode: str | None,
) -> ResumeResolution:
    return _runtime_resume.resolve_optimize_resume(
        input_path,
        workdir,
        resume_mode=resume_mode,
        reset_optimize=reset_optimize,
        requested_test_mode=requested_test_mode,
        requested_bench_mode=requested_bench_mode,
    )
