from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union, cast

from triton_agent.optimize.models import OptimizeCheckResult
from triton_agent.skill_loader import load_skill_script_module


def check_baseline(baseline_dir: Path) -> OptimizeCheckResult:
    module = load_skill_script_module(
        "npu-optimize-submit-baseline",
        "optimize_submit_baseline",
    )
    return _normalize_result(module.check_baseline(baseline_dir))


def check_round(
    round_dir: Path,
    *,
    current_round: int | None = None,
    final_round: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
) -> OptimizeCheckResult:
    module = load_skill_script_module(
        "npu-optimize-submit-round",
        "optimize_submit_round",
    )
    kwargs: dict[str, object] = {
        "current_round": current_round,
        "final_round": final_round,
    }
    if optimize_target is not None:
        kwargs["optimize_target"] = optimize_target
    return _normalize_result(module.check_round(round_dir, **kwargs))


def _normalize_result(raw_result: object) -> OptimizeCheckResult:
    if isinstance(raw_result, OptimizeCheckResult):
        return raw_result

    if isinstance(raw_result, dict):
        data = cast(dict[str, Any], raw_result)
    else:
        data = {
            "status": getattr(raw_result, "status", None),
            "kind": getattr(raw_result, "kind"),
            "issues": getattr(raw_result, "issues"),
            "summary": getattr(raw_result, "summary"),
            "next_option": getattr(raw_result, "next_option", None),
        }

    issues = _normalize_issues(data["issues"])
    kind = _normalize_kind(data["kind"])
    status = _normalize_status(data)

    return OptimizeCheckResult(
        kind=kind,
        status=status,
        issues=issues,
        summary=str(data["summary"]),
        next_option=_normalize_optional_str(data.get("next_option")),
    )


def _normalize_issues(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Optimize check issues must be a list or tuple of strings")
    return tuple(str(item) for item in cast(Union[list[object], tuple[object, ...]], value))


def _normalize_kind(value: object) -> Literal["baseline", "round"]:
    text = str(value)
    if text not in {"baseline", "round"}:
        raise ValueError(f"Unexpected optimize check kind: {text}")
    return cast(Literal["baseline", "round"], text)


def _normalize_status(data: dict[str, Any]) -> Literal["pass", "fail"]:
    raw_status = data.get("status")
    if raw_status is not None:
        text = str(raw_status)
        if text not in {"pass", "fail"}:
            raise ValueError(f"Unexpected optimize check status: {text}")
        return cast(Literal["pass", "fail"], text)

    raw_decision = data.get("decision")
    if raw_decision is not None:
        text = str(raw_decision)
        if text == "pass":
            return "pass"
        if text in {"revise-required", "hard-fail", "revise-metadata"}:
            return "fail"
        raise ValueError(f"Unexpected optimize check decision: {text}")

    raw_ok = data.get("ok")
    if raw_ok is not None:
        return "pass" if bool(raw_ok) else "fail"

    raise KeyError("Optimize check result must contain status, decision, or ok")


def _normalize_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
