from __future__ import annotations

from pathlib import Path
from typing import TextIO

from triton_agent.diff_skills_update.models import DiscoveryResult, OperatorPair, SkipRecord


def discover_operator_pairs(
    root: Path,
    *,
    stream: TextIO | None = None,
    exclude_dirs: set[Path] | None = None,
) -> DiscoveryResult:
    if not root.exists():
        raise ValueError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a directory: {root}")

    pairs: list[OperatorPair] = []
    skips: list[SkipRecord] = []
    excluded = {path.resolve() for path in exclude_dirs or set()}
    for operator_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if operator_dir.resolve() in excluded:
            continue
        opt_files = sorted(operator_dir.glob("opt_*.py"))
        if not opt_files:
            skips.append(_record_skip(operator_dir, "no opt_*.py file found", stream=stream))
            continue
        for opt_path in opt_files:
            baseline_name = opt_path.name.removeprefix("opt_")
            baseline_path = operator_dir / baseline_name
            if not baseline_path.exists():
                skips.append(
                    _record_skip(
                        operator_dir,
                        f"missing baseline file {baseline_name} for {opt_path.name}",
                        opt_path=opt_path,
                        stream=stream,
                    )
                )
                continue
            if not baseline_path.is_file():
                skips.append(
                    _record_skip(
                        operator_dir,
                        f"baseline path is not a file: {baseline_path.name}",
                        opt_path=opt_path,
                        stream=stream,
                    )
                )
                continue
            pairs.append(
                OperatorPair(
                    operator_dir=operator_dir,
                    baseline_path=baseline_path,
                    expected_path=opt_path,
                )
            )
    return DiscoveryResult(pairs=tuple(pairs), skips=tuple(skips))


def _record_skip(
    operator_dir: Path,
    reason: str,
    *,
    opt_path: Path | None = None,
    stream: TextIO | None = None,
) -> SkipRecord:
    record = SkipRecord(operator_dir=operator_dir, reason=reason, opt_path=opt_path)
    if stream is not None:
        print(f"skip {operator_dir}: {reason}", file=stream)
    return record
