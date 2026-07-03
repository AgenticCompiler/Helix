from __future__ import annotations

from shared.models import OptimizeCheckResult


def build_check_payload(result: OptimizeCheckResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": result.kind,
        "status": result.status,
        "issues": list(result.issues),
        "guideline": result.summary,
    }
    if result.next_option is not None:
        payload["next_option"] = result.next_option
    return payload


def build_workflow_failure_payload(
    *,
    kind: str,
    issue: str,
    guideline: str,
) -> dict[str, object]:
    return {
        "kind": kind,
        "status": "fail",
        "issues": [issue],
        "guideline": guideline,
    }
