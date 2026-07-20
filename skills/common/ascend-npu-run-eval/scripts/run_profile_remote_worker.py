"""Self-contained remote worker for one torch-npu-profiler profile case."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from env_registry import TRITON_ALWAYS_COMPILE
from result_payload import ResultPayload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("--bench-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args(argv)
    import run_bench_execution

    previous = os.environ.get(TRITON_ALWAYS_COMPILE)
    os.environ[TRITON_ALWAYS_COMPILE] = "1"
    try:
        result = run_bench_execution.profile_bench_case_quick(
            Path(args.bench_file), Path(args.operator_file), args.case_id
        )
    finally:
        if previous is None:
            os.environ.pop(TRITON_ALWAYS_COMPILE, None)
        else:
            os.environ[TRITON_ALWAYS_COMPILE] = previous
    profile_name = _latest_profile_name() if int(result["return_code"]) == 0 else None
    _write_payload(Path(args.result_file), result, profile_name)
    return 0


def _latest_profile_name() -> str | None:
    candidates = [
        path
        for path in Path(".").rglob("PROF_*")
        if path.is_dir() and _is_valid_profile_dir(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).as_posix()


def _is_valid_profile_dir(profile_dir: Path) -> bool:
    output_dir = profile_dir / "mindstudio_profiler_output"
    return output_dir.is_dir() and any(output_dir.glob("op_statistic_*.csv"))


def _write_payload(result_file: Path, result: ResultPayload, profile_name: str | None) -> None:
    result_file.write_text(
        json.dumps(
            {
                "result": {
                    "return_code": int(result["return_code"]),
                    "stdout": str(result["stdout"]),
                    "stderr": str(result["stderr"]),
                    "stalled": bool(result["stalled"]),
                    "session_id": result["session_id"],
                },
                "profile_name": profile_name,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
