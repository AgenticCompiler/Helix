from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TextIO

from triton_agent.optimize.models import BatchOptimizeResult
from triton_agent.optimize.render import render_batch_optimize_results
from triton_agent.status.core import workspace_has_optimize_artifacts

from .log_check_launcher import run_log_check


def run_log_check_batch(
    root: Path,
    *,
    output_file: str = "log_check_result.md",
    summary_file: str = "log_check_summary.md",
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
            output_json=f"{Path(output_file).stem}.json",
            agent_name=agent_name,
            verbose=verbose,
            show_output=show_output,
            log_tools=log_tools,
        )
        if rc != 0:
            return BatchOptimizeResult(workspace=workspace, status="failed", message=f"log check exited with return code {rc}")
        if not output_path.is_file():
            return BatchOptimizeResult(workspace=workspace, status="failed", message=f"missing {output_file}")
        passed, message = summarize_log_check_output(output_path)
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
    """Summarize log_check result from JSON (preferred) or markdown (fallback).

    *path* is the log_check_result.md file path. The corresponding JSON file is
    derived by replacing the suffix.
    """
    json_path = path.with_name("log_check_result.json")
    if json_path.is_file():
        return _summarize_from_json(json_path)

    # Fallback: parse legacy markdown
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"failed to read {path.name}: {exc}"
    lines = [line.strip() for line in text.splitlines()]
    overall = _first_summary_value(lines, "overall")
    failed_checks = _first_summary_value(lines, "failed_checks")
    if overall == "PASS":
        return True, "overall PASS"
    if overall == "FAIL":
        return False, failed_checks or "overall FAIL"
    if "result: FAIL" in text:
        return False, "missing overall summary; found result: FAIL"
    if "result: PASS" in text:
        return True, "missing overall summary; all visible results are PASS"
    return False, "missing overall summary"


def _summarize_from_json(json_path: Path) -> tuple[bool, str]:
    import json

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"failed to read {json_path.name}: {exc}"
    if not isinstance(data, dict):
        return False, f"{json_path.name} is not a JSON object"
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
    ordered = sorted(results, key=lambda item: item.workspace.name)
    passed = [item for item in ordered if item.status == "ok"]
    failed = [item for item in ordered if item.status == "failed"]
    skipped = [item for item in ordered if item.status == "skipped"]
    lines = [
        "# Log Check Batch Summary",
        "",
        f"overall: {'PASS' if not failed else 'FAIL'}",
        f"total: {len(ordered)}",
        f"passed: {len(passed)}",
        f"failed: {len(failed)}",
        f"skipped: {len(skipped)}",
        "",
        "## Passed",
        *_format_result_items(passed),
        "",
        "## Failed",
        *_format_result_items(failed),
        "",
        "## Skipped",
        *_format_result_items(skipped),
        "",
    ]
    path = root / output_file
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _first_summary_value(lines: list[str], key: str) -> str | None:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _format_result_items(items: list[BatchOptimizeResult]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item.workspace.name}: {item.message}" for item in items]


__all__ = [
    "run_log_check_batch",
    "summarize_log_check_output",
    "write_log_check_batch_summary",
]
