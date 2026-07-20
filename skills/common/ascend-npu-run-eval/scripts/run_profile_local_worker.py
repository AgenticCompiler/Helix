"""Fixed local worker for a single benchmark profile case."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from run_profile_execution import execute_local_profile, resolve_local_profile_dir
from result_payload import ResultPayload
from run_runtime import result_succeeded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("--bench-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args(argv)
    bench_file = Path(args.bench_file).expanduser().resolve()
    operator_file = Path(args.operator_file).expanduser().resolve()
    result = execute_local_profile(bench_file, operator_file, args.case_id)
    profile_dir = resolve_local_profile_dir(bench_file.parent) if result_succeeded(result) else None
    _write_payload(Path(args.result_file), result, profile_dir)
    return 0


def _write_payload(result_file: Path, result: ResultPayload, profile_dir: Path | None) -> None:
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
                "profile_dir": None if profile_dir is None else str(profile_dir.resolve()),
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
