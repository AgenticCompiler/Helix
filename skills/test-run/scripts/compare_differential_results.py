#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from optimizer.script.differential_compare import (
    ORACLE_COMPARE_LEVELS,
    compare_values,
    extract_ordered_results,
    load_result_payload,
    resolve_compare_tolerances,
)


def compare_result_files(oracle_result_path: Path, compare_result_path: Path, level: str) -> int:
    try:
        rtol, atol = resolve_compare_tolerances(level)
    except ValueError:
        print(
            f"FAIL: invalid compare level '{level}', "
            f"expected one of {sorted(ORACLE_COMPARE_LEVELS)}"
        )
        return 2

    expected_payload = load_result_payload(oracle_result_path, map_location="cpu")
    actual_payload = load_result_payload(compare_result_path, map_location="cpu")

    expected, expected_error = extract_ordered_results(expected_payload, "oracle")
    if expected_error:
        print(f"FAIL: {expected_error}")
        return 1

    actual, actual_error = extract_ordered_results(actual_payload, "compare")
    if actual_error:
        print(f"FAIL: {actual_error}")
        return 1

    mismatch = compare_values(expected, actual, "output", rtol, atol)
    if mismatch:
        print(f"FAIL: {mismatch}")
        return 1

    print(
        "PASS: ordered outputs match "
        f"(level={level.strip().lower()}, rtol={rtol}, atol={atol})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Triton Ascend differential result payload files."
    )
    parser.add_argument("oracle_result", type=Path, help="Path to oracle_result_*.pt")
    parser.add_argument("compare_result", type=Path, help="Path to compare_result_*.pt")
    parser.add_argument(
        "--compare-level",
        default="balanced",
        choices=sorted(ORACLE_COMPARE_LEVELS),
        help="Tolerance preset for ordered output comparison.",
    )
    args = parser.parse_args()

    return compare_result_files(args.oracle_result, args.compare_result, args.compare_level)


if __name__ == "__main__":
    raise SystemExit(main())
