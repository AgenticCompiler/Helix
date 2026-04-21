from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from triton_agent.optimize.models import BatchOptimizeResult
from triton_agent.optimize.render import render_batch_optimize_results
from triton_agent.optimize.status import find_latest_verify_state
from triton_agent.optimize.verify import (
    OptimizeVerifyOptions,
    prepare_optimize_verify_target,
    run_optimize_verify,
)


def run_optimize_verify_batch(
    root: Path,
    *,
    force_verify: bool = False,
    stdout: TextIO | None = None,
    options: OptimizeVerifyOptions | None = None,
) -> int:
    stream = stdout or sys.stdout
    workspaces = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspaces:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    verify_options = options or OptimizeVerifyOptions()
    results: list[BatchOptimizeResult] = []
    for workspace in workspaces:
        latest_state = find_latest_verify_state(workspace)
        if latest_state is not None and not force_verify:
            results.append(
                BatchOptimizeResult(
                    workspace=workspace,
                    status="ok",
                    message=f"reused verify-state.json: {latest_state}",
                )
            )
            continue
        try:
            target = prepare_optimize_verify_target(workspace)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            results.append(
                BatchOptimizeResult(
                    workspace=workspace,
                    status="skipped",
                    message=str(exc),
                )
            )
            continue
        result = run_optimize_verify(target, verify_options)
        if result.return_code == 0:
            results.append(
                BatchOptimizeResult(
                    workspace=workspace,
                    status="ok",
                    message=f"verified: {result.state_path}",
                )
            )
            continue
        results.append(
            BatchOptimizeResult(
                workspace=workspace,
                status="failed",
                message=f"verify exited with return code {result.return_code}",
            )
        )
    return render_batch_optimize_results(results, stdout=stream)
