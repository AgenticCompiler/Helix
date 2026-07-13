from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from helix.distill.models import OperatorDistillResult, SkipRecord


def report_path_for_pair(pair_stem: str, simulate_dir: Path, *, pair_count_in_dir: int) -> Path:
    if pair_count_in_dir == 1:
        return simulate_dir / "report.json"
    return simulate_dir / f"report_{pair_stem}.json"


def read_json_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, object], data)


def write_pair_report(result: OperatorDistillResult) -> None:
    data: dict[str, Any] = {
        "status": result.status,
        "operator_dir": str(result.pair.operator_dir),
        "source_kind": result.pair.source_kind,
        "baseline": str(result.pair.baseline_path),
        "expected": str(result.pair.expected_path),
        "learned_lessons": (
            str(result.pair.learned_lessons_path)
            if result.pair.learned_lessons_path is not None
            else None
        ),
        "matched_patterns": result.matched_patterns,
        "updated_patterns": result.updated_patterns,
        "message": result.message,
        "iterations": [
            {
                "iteration": item.iteration,
                "status": item.status,
                "candidate_path": str(item.candidate_path),
                "simulate_return_code": item.simulate_return_code,
                "analysis_return_code": item.analysis_return_code,
                "analysis_summary": item.analysis_summary,
                "updated_patterns": item.updated_patterns,
            }
            for item in result.iterations
        ],
    }
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_skip_report(record: SkipRecord, report_path: Path) -> None:
    data = {
        "status": "skipped",
        "operator_dir": str(record.operator_dir),
        "opt_path": str(record.opt_path) if record.opt_path is not None else None,
        "reason": record.reason,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
