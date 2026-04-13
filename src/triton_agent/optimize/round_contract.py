from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from triton_agent.optimize.models import RoundArtifactsInspection, RoundState

_ROUND_STATE_REQUIRED_FIELDS = (
    "round",
    "parent_round",
    "hypothesis",
    "evidence_sources",
    "correctness_status",
    "benchmark_status",
    "perf_artifact",
    "canonical_baseline",
    "comparison_target",
    "perf_summary_source",
    "summary_path",
    "opt_note_updated",
    "next_recommendation",
)
_ROUND_METADATA_FILENAMES = {
    "attempts.md",
    "summary.md",
    "perf.txt",
    "round-state.json",
}


def load_round_state(round_dir: Path) -> RoundState:
    round_state_path = round_dir / "round-state.json"
    try:
        payload = json.loads(round_state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing round-state.json in {round_dir}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid round-state.json in {round_dir}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"round-state.json must contain an object in {round_dir}")
    data = cast(dict[str, Any], payload)

    missing_fields = [
        field_name for field_name in _ROUND_STATE_REQUIRED_FIELDS if field_name not in data
    ]
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise ValueError(f"missing required round-state fields: {missing_text}")

    evidence_sources_value = data["evidence_sources"]
    if not isinstance(evidence_sources_value, list):
        raise ValueError("round-state evidence_sources must be a list of strings")
    evidence_sources_raw = cast(list[Any], evidence_sources_value)
    evidence_sources: list[str] = []
    for item in evidence_sources_raw:
        if not isinstance(item, str):
            raise ValueError("round-state evidence_sources must be a list of strings")
        evidence_sources.append(item)

    return RoundState(
        round_name=str(data["round"]),
        parent_round=str(data["parent_round"]),
        hypothesis=str(data["hypothesis"]),
        evidence_sources=tuple(evidence_sources),
        correctness_status=str(data["correctness_status"]),
        benchmark_status=str(data["benchmark_status"]),
        perf_artifact=str(data["perf_artifact"]),
        canonical_baseline=str(data["canonical_baseline"]),
        comparison_target=str(data["comparison_target"]),
        perf_summary_source=str(data["perf_summary_source"]),
        summary_path=str(data["summary_path"]),
        opt_note_updated=bool(data["opt_note_updated"]),
        next_recommendation=str(data["next_recommendation"]),
        analysis_skipped_reason=_optional_str(data.get("analysis_skipped_reason")),
        profile_dir=_optional_str(data.get("profile_dir")),
        ir_dir=_optional_str(data.get("ir_dir")),
        validated_candidate=_optional_bool(data.get("validated_candidate")),
    )


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    attempts_path = _existing_file(round_dir / "attempts.md")
    summary_path = _existing_file(round_dir / "summary.md")
    round_state_path = _existing_file(round_dir / "round-state.json")
    perf_path = _find_perf_artifact(round_dir)
    operator_path = _find_round_operator(round_dir)

    issues: list[str] = []
    if attempts_path is None:
        issues.append("missing attempts.md")
    if summary_path is None:
        issues.append("missing summary.md")
    if round_state_path is None:
        issues.append("missing round-state.json")
    if perf_path is None:
        issues.append("missing perf artifact")
    if operator_path is None:
        issues.append("missing round-local operator output")

    return RoundArtifactsInspection(
        round_dir=round_dir,
        operator_path=operator_path,
        attempts_path=attempts_path,
        summary_path=summary_path,
        perf_path=perf_path,
        round_state_path=round_state_path,
        issues=tuple(issues),
    )


def _existing_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _find_perf_artifact(round_dir: Path) -> Path | None:
    perf_txt = round_dir / "perf.txt"
    if perf_txt.is_file():
        return perf_txt
    perf_files = sorted(round_dir.glob("*_perf.txt"))
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def _find_round_operator(round_dir: Path) -> Path | None:
    candidates = [
        path
        for path in sorted(round_dir.iterdir())
        if path.is_file() and path.name not in _ROUND_METADATA_FILENAMES and not path.name.endswith("_perf.txt")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        preferred_python = [path for path in candidates if path.suffix == ".py"]
        if len(preferred_python) == 1:
            return preferred_python[0]
        return candidates[0]
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)
