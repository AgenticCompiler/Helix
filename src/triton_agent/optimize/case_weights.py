from __future__ import annotations

import json
from pathlib import Path

from triton_agent.skill_loader import load_skill_script_module


def derive_workspace_case_weights(input_path: Path, workdir: Path) -> Path:
    output = workdir / "case_weights.json"
    module = load_skill_script_module("triton-npu-case-weighting", "case_weighting")
    payload = module.derive_weights(
        cases_json=_resolve_cases_json(input_path, workdir),
        bench_file=_resolve_bench_file(input_path, workdir),
        full_perf=_resolve_full_perf(input_path, workdir),
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _resolve_cases_json(input_path: Path, workdir: Path) -> Path:
    candidates = [
        workdir / f"{stem}.json"
        for stem in _operator_stems(input_path)
    ]
    candidates.extend(sorted(path for path in workdir.glob("*.json") if path.name != "case_weights.json"))
    return _first_existing_unique(candidates, label="case JSON")


def _resolve_bench_file(input_path: Path, workdir: Path) -> Path:
    candidates = [
        workdir / f"bench_{stem}.py"
        for stem in _operator_stems(input_path)
    ]
    candidates.extend(sorted(workdir.glob("bench_*.py")))
    return _first_existing_unique(candidates, label="bench file")


def _resolve_full_perf(input_path: Path, workdir: Path) -> Path | None:
    candidates = [
        directory / f"{stem}{suffix}"
        for directory in (workdir, workdir / "baseline")
        for stem in _operator_stems(input_path)
        for suffix in ("_full_case_perf.txt", "_full-case_perf.txt", "_perf.txt")
    ]
    candidates.extend(sorted(workdir.glob("*full_case_perf.txt")))
    candidates.extend(sorted(workdir.glob("*full-case_perf.txt")))
    candidates.extend(sorted(workdir.glob("*_perf.txt")))
    existing = _unique_existing(candidates)
    return existing[0] if existing else None


def _operator_stems(input_path: Path) -> tuple[str, ...]:
    stem = input_path.stem
    if stem.startswith("triton_"):
        return stem, stem.removeprefix("triton_")
    return (stem,)


def _first_existing_unique(candidates: list[Path], *, label: str) -> Path:
    existing = _unique_existing(candidates)
    if not existing:
        raise ValueError(f"Unable to derive case weights because no {label} was found")
    return existing[0]


def _unique_existing(candidates: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing
