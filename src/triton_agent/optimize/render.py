from __future__ import annotations

import sys
from typing import TextIO

from triton_agent.optimize.models import BatchOptimizeResult


def render_batch_optimize_results(
    results: list[BatchOptimizeResult],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    ordered_results = sorted(results, key=lambda item: item.workspace.name)
    succeeded = sum(1 for item in ordered_results if item.status == "ok")
    failed = sum(1 for item in ordered_results if item.status == "failed")
    skipped = sum(1 for item in ordered_results if item.status == "skipped")
    for item in ordered_results:
        status = {
            "ok": "OK",
            "failed": "FAIL",
            "skipped": "SKIP",
        }[item.status]
        print(f"[{status}] {item.workspace.name}: {item.message}", file=stream)
    print(f"Summary: {succeeded} succeeded, {failed} failed, {skipped} skipped", file=stream)
    return 0 if failed == 0 and ordered_results else 1


__all__ = ["render_batch_optimize_results"]
