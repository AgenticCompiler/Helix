from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import TextIO, cast

from triton_agent.batch_utils import (
    NO_CANDIDATE_OPERATOR_FILE,
    PrefixedTextStream,
    discover_batch_workspaces,
)
from triton_agent.mcp import managed_mcp_scope, managed_mcp_server_names_for_request
from triton_agent.models import AgentResult, CommandKind
from triton_agent.optimize.naming import (
    resolve_batch_optimize_operator_file,
)
from triton_agent.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_devices,
    configured_batch_npu_slots,
    validate_batch_affinity_capacity,
)
from triton_agent.optimize.models import BatchOptimizeResult, BatchOptimizeWorkspace, OptimizeRunOptions
from triton_agent.optimize.render import render_batch_optimize_results
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request
from triton_agent.optimize_upload.client import UploadUrlMissingError
from triton_agent.optimize_upload.workflow import upload_optimize_workspace
from triton_agent.skill_staging import resolve_staged_skills

_BATCH_STATUS_FILENAME = "optimize-batch-status.json"
_BATCH_STATUS_VERSION = 1


def run_optimize_batch(
    root: Path,
    options: OptimizeRunOptions,
    *,
    max_concurrency: int,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    optimize_request_runner = run_request or run_optimize_request
    if options.reset_optimize:
        clear_optimize_batch_status_file(root)
    batch_status = load_optimize_batch_status(root)
    discovered, failures = discover_batch_workspaces(
        root,
        resolve_operator_file=resolve_batch_optimize_operator_file,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
    runnable = [
        BatchOptimizeWorkspace(workspace=workspace, operator_file=operator_file)
        for workspace, operator_file in discovered
    ]
    results: list[BatchOptimizeResult] = []
    for workspace, message in failures:
        ws_key = optimize_batch_workspace_key(root, workspace)
        record = batch_status.get(ws_key)
        if record is not None and record.get("status") == "completed":
            results.append(
                BatchOptimizeResult(
                    workspace=workspace,
                    status="skipped",
                    message="already completed",
                )
            )
        else:
            results.append(
                BatchOptimizeResult(
                    workspace=workspace,
                    status="failed",
                    message=message,
                )
            )
    if not runnable and not results:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    output_lock = threading.Lock()
    stream = stdout or sys.stdout
    devices = configured_batch_npu_devices()
    if not options.enable_mcp:
        validate_batch_affinity_capacity(devices, max_concurrency=max_concurrency)
    affinity_pool = (
        BatchNpuAffinityPool(slots)
        if not options.enable_mcp and (slots := configured_batch_npu_slots()) is not None
        else None
    )

    def _run_item(
        item: BatchOptimizeWorkspace,
        forwarded_stdout: TextIO | None = None,
        forwarded_stderr: TextIO | None = None,
    ) -> AgentResult:
        request = build_optimize_request(item.operator_file, item.workspace, options)
        if affinity_pool is not None:
            with affinity_pool.acquire() as device:
                request = replace(
                    request,
                    extra_env={
                        **(request.extra_env or {}),
                        **affinity_env_for_device(device),
                    },
                )
                if forwarded_stdout is not None or forwarded_stderr is not None:
                    return optimize_request_runner(request, forwarded_stdout, forwarded_stderr)
                return optimize_request_runner(request)
        if forwarded_stdout is not None or forwarded_stderr is not None:
            return optimize_request_runner(request, forwarded_stdout, forwarded_stderr)
        return optimize_request_runner(request)

    staged_skill_names, _ = resolve_staged_skills(
        CommandKind.OPTIMIZE,
        optimize_knowledge=options.optimize_knowledge,
        optimize_target=options.optimize_target,
        enable_cann_ext_api=options.enable_cann_ext_api,
        enable_mcp=options.enable_mcp,
    )
    scope = (
        managed_mcp_scope()
        if managed_mcp_server_names_for_request(staged_skill_names, enable_mcp=options.enable_mcp)
        else nullcontext()
    )
    with scope, ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[AgentResult], BatchOptimizeWorkspace] = {}
        for item in runnable:
            if should_skip_optimize_batch_workspace(root, item.workspace, item.operator_file, batch_status):
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        status="skipped",
                        message="already completed",
                    )
                )
                continue
            try:
                build_optimize_request(item.operator_file, item.workspace, options)
            except ValueError as exc:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        status="failed",
                        message=str(exc),
                    )
                )
                update_optimize_batch_workspace_status(
                    root,
                    item.workspace,
                    item.operator_file,
                    status="incomplete",
                )
                continue
            if options.stream_output:
                prefix = f"[{item.workspace.name}] "
                prefixed_stream = PrefixedTextStream(stream, prefix, output_lock)
                forwarded_stream = cast(TextIO, prefixed_stream)
                futures[executor.submit(_run_item, item, forwarded_stream, forwarded_stream)] = item
            else:
                futures[executor.submit(_run_item, item)] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        status="failed",
                        message=f"unexpected optimize failure: {exc}",
                    )
                )
                update_optimize_batch_workspace_status(
                    root,
                    item.workspace,
                    item.operator_file,
                    status="incomplete",
                )
                continue
            if result.succeeded:
                if options.upload_enabled:
                    try:
                        upload_optimize_workspace(item.workspace, verbose=options.verbose)
                    except UploadUrlMissingError:
                        if options.verbose:
                            print(
                                f"[{item.workspace.name}] Auto-upload skipped: URL not set.",
                                file=sys.stderr,
                            )
                    except (ValueError, RuntimeError) as exc:
                        if options.verbose:
                            print(
                                f"[{item.workspace.name}] Auto-upload warning: {exc}",
                                file=sys.stderr,
                            )
                if options.report:
                    from triton_agent.report.workspace import generate_workspace_report

                    try:
                        if options.verbose:
                            print(
                                f"[{item.workspace.name}] Auto-report: generating report.md...",
                                file=sys.stderr,
                            )
                        from triton_agent.report.workspace import generate_workspace_report
                        report_ok, report_msg = generate_workspace_report(
                            workspace=item.workspace,
                            agent_name=options.agent_name,
                            show_output=options.stream_output,
                        )
                        if options.verbose:
                            status = "completed" if report_ok else f"warning: {report_msg}"
                            print(
                                f"[{item.workspace.name}] Auto-report: {status}",
                                file=sys.stderr,
                            )
                    except Exception as exc:
                        if options.verbose:
                            print(
                                f"[{item.workspace.name}] Auto-report warning: {exc}",
                                file=sys.stderr,
                            )
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        status="ok",
                        message=f"optimized {item.operator_file.name}",
                    )
                )
                update_optimize_batch_workspace_status(
                    root,
                    item.workspace,
                    item.operator_file,
                    status="completed",
                )
            else:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        status="failed",
                        message=summarize_batch_optimize_failure(result),
                    )
                )
                update_optimize_batch_workspace_status(
                    root,
                    item.workspace,
                    item.operator_file,
                    status="incomplete",
                )

    return render_batch_optimize_results(results, stdout=stream)

def summarize_batch_optimize_failure(result: AgentResult) -> str:
    for output in (result.stderr, result.stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"optimize exited with return code {result.return_code}"


def optimize_batch_status_file(root: Path) -> Path:
    return root / _BATCH_STATUS_FILENAME


def clear_optimize_batch_status_file(root: Path) -> None:
    path = optimize_batch_status_file(root)
    if path.exists():
        path.unlink()


def load_optimize_batch_status(root: Path) -> dict[str, dict[str, str]]:
    path = optimize_batch_status_file(root)
    if not path.is_file():
        return {}
    try:
        payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(payload_obj, dict):
        return {}
    payload = cast(dict[object, object], payload_obj)
    if payload.get("version") != _BATCH_STATUS_VERSION:
        return {}
    workspaces_obj = payload.get("workspaces")
    if not isinstance(workspaces_obj, dict):
        return {}
    workspaces = cast(dict[object, object], workspaces_obj)

    normalized: dict[str, dict[str, str]] = {}
    for key, value in workspaces.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        value_dict = cast(dict[object, object], value)
        status = value_dict.get("status")
        operator_file = value_dict.get("operator_file")
        if not isinstance(status, str) or not isinstance(operator_file, str):
            continue
        normalized[key] = {
            "status": status,
            "operator_file": operator_file,
        }
    return normalized


def should_skip_optimize_batch_workspace(
    root: Path,
    workspace: Path,
    operator_file: Path,
    status_entries: dict[str, dict[str, str]],
) -> bool:
    record = status_entries.get(optimize_batch_workspace_key(root, workspace))
    if record is None:
        return False
    if record.get("status") != "completed":
        return False
    return record.get("operator_file") == operator_file.name


def update_optimize_batch_workspace_status(
    root: Path,
    workspace: Path,
    operator_file: Path,
    *,
    status: str,
) -> None:
    entries = load_optimize_batch_status(root)
    entries[optimize_batch_workspace_key(root, workspace)] = {
        "status": status,
        "operator_file": operator_file.name,
    }
    payload = {
        "version": _BATCH_STATUS_VERSION,
        "workspaces": dict(sorted(entries.items())),
    }
    optimize_batch_status_file(root).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def optimize_batch_workspace_key(root: Path, workspace: Path) -> str:
    relative = workspace.relative_to(root)
    if relative == Path("."):
        return "."
    return relative.as_posix()
