from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.optimize.models import BatchOptimizeResult
from triton_agent.optimize.render import render_batch_optimize_results
from triton_agent.status.core import workspace_has_optimize_artifacts

from .log_check_launcher import run_log_check


def run_log_check_batch(
    root: Path,
    *,
    output_file: str = "log_check_result.json",
    summary_file: str = "log_check_summary.json",
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = False,
    log_tools: bool = False,
    max_concurrency: int = 1,
    stdout: TextIO | None = None,
    run_one: Callable[..., int] | None = None,
) -> int:
    stream = stdout or sys.stdout
    workspaces = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not workspaces:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    output_lock = threading.Lock()
    log_check_runner = run_one or run_log_check
    results: list[BatchOptimizeResult] = []

    def _run_item(workspace: Path) -> BatchOptimizeResult:
        if not workspace_has_optimize_artifacts(workspace):
            return BatchOptimizeResult(workspace=workspace, status="skipped", message="no optimize artifacts found")
        output_path = workspace / output_file
        if output_path.exists():
            return BatchOptimizeResult(workspace=workspace, status="skipped", message="report already exists")
        rc = log_check_runner(
            target_path=workspace,
            output_json=output_file,
            agent_name=agent_name,
            verbose=verbose,
            show_output=show_output,
            log_tools=log_tools,
        )
        if rc != 0:
            return BatchOptimizeResult(workspace=workspace, status="failed", message=f"log check exited with return code {rc}")
        if not output_path.is_file():
            return BatchOptimizeResult(workspace=workspace, status="failed", message=f"missing {output_file}")
        passed, message = _summarize_from_json(output_path)
        return BatchOptimizeResult(
            workspace=workspace,
            status="ok" if passed else "failed",
            message=message,
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[BatchOptimizeResult], Path] = {}
        for workspace in workspaces:
            futures[executor.submit(_run_item, workspace)] = workspace
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                workspace = futures[future]
                result = BatchOptimizeResult(
                    workspace=workspace,
                    status="failed",
                    message=f"unexpected log-check failure: {exc}",
                )
            with output_lock:
                results.append(result)

    summary_path = write_log_check_batch_summary(root, results, output_file=summary_file)
    print(f"Batch log-check summary: {summary_path}", file=stream)
    return render_batch_optimize_results(results, stdout=stream)


def summarize_log_check_output(path: Path) -> tuple[bool, str]:
    """Summarize log_check result from a log_check_result.json path."""
    return _summarize_from_json(path)


def _summarize_from_json(json_path: Path) -> tuple[bool, str]:
    import json

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"failed to read {json_path.name}: {exc}"
    if not isinstance(payload, dict):
        return False, f"{json_path.name} is not a JSON object"
    data = cast(dict[str, Any], payload)
    overall = data.get("overall")
    if overall == "PASS":
        return True, "overall PASS"
    if overall == "FAIL":
        failed_checks = data.get("failed_checks", "")
        return False, str(failed_checks) or "overall FAIL"
    return False, f"unexpected overall value: {overall!r}"


def write_log_check_batch_summary(
    root: Path,
    results: list[BatchOptimizeResult],
    *,
    output_file: str,
) -> Path:
    import json

    ordered = sorted(results, key=lambda item: item.workspace.name)
    passed = [item for item in ordered if item.status == "ok"]
    failed = [item for item in ordered if item.status == "failed"]
    skipped = [item for item in ordered if item.status == "skipped"]

    def _item_to_dict(item: BatchOptimizeResult) -> dict[str, str]:
        return {
            "workspace": item.workspace.name,
            "status": item.status,
            "message": item.message,
        }

    payload = {
        "overall": "PASS" if not failed else "FAIL",
        "total": len(ordered),
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
        "results": {
            "passed": [_item_to_dict(item) for item in passed],
            "failed": [_item_to_dict(item) for item in failed],
            "skipped": [_item_to_dict(item) for item in skipped],
        },
    }
    path = root / output_file
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "run_log_check_batch",
    "summarize_log_check_output",
    "write_log_check_batch_summary",
]
