from __future__ import annotations

from typing import Literal

from shared.models import OptimizeCheckResult


def append_pass_issues_to_summary(summary: str, issues: tuple[str, ...]) -> str:
    if not issues:
        return summary
    return f"{summary} Notes: {'; '.join(issues)}"


def build_check_result(
    *,
    kind: Literal["baseline", "round"],
    status: Literal["pass", "fail"],
    issues: tuple[str, ...],
    summary: str | None = None,
    next_option: str | None = None,
) -> OptimizeCheckResult:
    if summary is None:
        summary = (
            append_pass_issues_to_summary(f"{kind} check passed", issues)
            if status == "pass"
            else f"{kind} check requires fixes: {'; '.join(issues)}"
        )
    return OptimizeCheckResult(
        kind=kind,
        status=status,
        issues=issues,
        summary=summary,
        next_option=next_option,
    )
