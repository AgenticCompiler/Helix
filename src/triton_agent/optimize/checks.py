from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union, cast

from triton_agent.optimize.models import OptimizeCheckResult
from triton_agent.skill_loader import load_skill_script_module


def check_baseline(baseline_dir: Path) -> OptimizeCheckResult:
    module = load_skill_script_module("triton-npu-optimize-check", "optimize_check")
    return _normalize_result(module.check_baseline(baseline_dir))


def check_round(
    round_dir: Path,
    *,
    min_rounds: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
) -> OptimizeCheckResult:
    module = load_skill_script_module("triton-npu-optimize-check", "optimize_check")
    kwargs: dict[str, object] = {"min_rounds": min_rounds}
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
            "ok": getattr(raw_result, "ok"),
            "kind": getattr(raw_result, "kind"),
            "decision": getattr(raw_result, "decision"),
            "issues": getattr(raw_result, "issues"),
            "summary": getattr(raw_result, "summary"),
        }

    issues = _normalize_issues(data["issues"])
    kind = _normalize_kind(data["kind"])
    decision = _normalize_decision(data["decision"])

    return OptimizeCheckResult(
        ok=bool(data["ok"]),
        kind=kind,
        decision=decision,
        issues=issues,
        summary=str(data["summary"]),
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


def _normalize_decision(value: object) -> Literal["pass", "revise-required", "hard-fail"]:
    text = str(value)
    if text not in {"pass", "revise-required", "hard-fail"}:
        raise ValueError(f"Unexpected optimize check decision: {text}")
    return cast(Literal["pass", "revise-required", "hard-fail"], text)
