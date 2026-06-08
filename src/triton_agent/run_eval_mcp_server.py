from __future__ import annotations
# pyright: reportUnknownMemberType=false, reportUnusedFunction=false

import asyncio
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, TYPE_CHECKING
from urllib.parse import parse_qs, quote

if TYPE_CHECKING:
    from fastmcp import FastMCP

from triton_agent.npu_affinity import parse_batch_npu_devices, parse_batch_workers_per_npu
from triton_agent.skill_loader import operator_eval_script_path


RUN_EVAL_MCP_SERVER_NAME = "triton-agent-run-eval"
_MCP_PATH = "/mcp"
_HOST = "127.0.0.1"


def build_slot_pool(assigned_npus: str, workers_per_npu: int) -> tuple[str, ...]:
    devices = parse_npu_devices(assigned_npus)
    if devices is None:
        raise ValueError("TRITON_AGENT_BATCH_NPU_DEVICES must not be empty")
    if workers_per_npu < 1:
        raise ValueError("TRITON_AGENT_BATCH_WORKERS_PER_NPU must be at least 1")
    return devices


def parse_npu_devices(raw: str | None) -> tuple[str, ...] | None:
    return parse_batch_npu_devices(raw)


class NpuDevicePool:
    def __init__(self, slots: tuple[str, ...]) -> None:
        self._condition = threading.Condition()
        self._available = list(slots)

    @contextmanager
    def acquire(self) -> Iterator[str]:
        with self._condition:
            while not self._available:
                self._condition.wait()
            device = self._available.pop(0)
        try:
            yield device
        finally:
            with self._condition:
                self._available.append(device)
                self._condition.notify()


def configured_slot_pool() -> NpuDevicePool:
    raw_devices = os.environ.get("TRITON_AGENT_BATCH_NPU_DEVICES")
    normalized_devices = raw_devices.strip() if raw_devices is not None else ""
    devices = parse_batch_npu_devices(normalized_devices or "0")
    if devices is None:
        raise ValueError("Managed run-eval MCP server resolved no NPU devices.")
    raw_workers = os.environ.get("TRITON_AGENT_BATCH_WORKERS_PER_NPU")
    workers_per_npu = parse_batch_workers_per_npu(raw_workers if raw_workers and raw_workers.strip() else None)
    return NpuDevicePool(build_slot_pool(",".join(devices), workers_per_npu))


def create_server(*, slot_pool: NpuDevicePool | None = None) -> "FastMCP":
    FastMCP, _ = _load_fastmcp_dependencies()
    pool = slot_pool or configured_slot_pool()
    server = FastMCP(RUN_EVAL_MCP_SERVER_NAME)

    @server.tool(
        name="run-test-baseline",
        description="Run the baseline operator against a test case and optionally compare an archived differential result.",
    )
    def run_test_baseline(
        test_file: Annotated[str, Field(description="Absolute path to the test entry file.")],
        operator_file: Annotated[str, Field(description="Absolute path to the operator implementation file.")],
        test_mode: Annotated[
            str | None,
            Field(description="Optional test mode override. Supported values: standalone, differential."),
        ] = None,
        remote: Annotated[str | None, Field(description="Optional remote execution target.")] = None,
        remote_workdir: Annotated[str | None, Field(description="Optional remote workspace root override.")] = None,
    ) -> dict[str, object]:
        workspace = current_workspace()
        arguments = _build_run_test_arguments(
            test_file=test_file,
            operator_file=operator_file,
            test_mode=test_mode,
            compare_level=None,
            baseline_result=None,
            baseline_operator_file=None,
            remote=remote,
            remote_workdir=remote_workdir,
            keep_remote_workdir=False,
            verbose=False,
        )
        with _lease_device(pool) as leased_device:
            return _run_subcommand(
                "run-test-baseline",
                arguments,
                leased_device=leased_device,
                workspace=workspace,
            )

    @server.tool(
        name="run-test-optimize",
        description="Run the optimized operator against a test case and compare it with baseline evidence.",
    )
    def run_test_optimize(
        test_file: Annotated[str, Field(description="Absolute path to the test entry file.")],
        operator_file: Annotated[str, Field(description="Absolute path to the optimized operator implementation file.")],
        baseline_operator_file: Annotated[
            str | None,
            Field(description="Absolute path to the baseline operator file used to produce comparison output."),
        ] = None,
        baseline_result: Annotated[
            str | None,
            Field(description="Absolute path to an archived baseline result used for differential comparison."),
        ] = None,
        test_mode: Annotated[
            str | None,
            Field(description="Optional test mode override. Supported values: standalone, differential."),
        ] = None,
        compare_level: Annotated[
            str | None,
            Field(description="Optional differential comparison strictness. Supported values: strict, balanced, relaxed."),
        ] = None,
        remote: Annotated[str | None, Field(description="Optional remote execution target.")] = None,
        remote_workdir: Annotated[str | None, Field(description="Optional remote workspace root override.")] = None,
    ) -> dict[str, object]:
        workspace = current_workspace()
        arguments = _build_run_test_arguments(
            test_file=test_file,
            operator_file=operator_file,
            test_mode=test_mode,
            compare_level=compare_level,
            baseline_result=baseline_result,
            baseline_operator_file=baseline_operator_file,
            remote=remote,
            remote_workdir=remote_workdir,
            keep_remote_workdir=False,
            verbose=False,
        )
        with _lease_device(pool) as leased_device:
            return _run_subcommand(
                "run-test-optimize",
                arguments,
                leased_device=leased_device,
                workspace=workspace,
            )

    @server.tool(
        name="run-bench",
        description="Run a benchmark workload on the operator and return the generated perf artifact path.",
    )
    def run_bench(
        bench_file: Annotated[str, Field(description="Absolute path to the benchmark entry file.")],
        operator_file: Annotated[str, Field(description="Absolute path to the operator implementation file.")],
        bench_mode: Annotated[
            str | None,
            Field(description="Optional benchmark mode override. Supported values: standalone, msprof."),
        ] = None,
        remote: Annotated[str | None, Field(description="Optional remote execution target.")] = None,
        remote_workdir: Annotated[str | None, Field(description="Optional remote workspace root override.")] = None,
    ) -> dict[str, object]:
        workspace = current_workspace()
        arguments = [
            "--bench-file",
            bench_file,
            "--operator-file",
            operator_file,
        ]
        if bench_mode is not None:
            arguments.extend(["--bench-mode", bench_mode])
        _append_common_remote_arguments(
            arguments,
            remote=remote,
            remote_workdir=remote_workdir,
            keep_remote_workdir=False,
            verbose=False,
        )
        with _lease_device(pool) as leased_device:
            return _run_subcommand(
                "run-bench",
                arguments,
                leased_device=leased_device,
                workspace=workspace,
            )

    @server.tool(
        name="profile-bench",
        description="Run a benchmark profile collection and return the generated profile directory.",
    )
    def profile_bench(
        bench_file: Annotated[str, Field(description="Absolute path to the benchmark entry file.")],
        operator_file: Annotated[str, Field(description="Absolute path to the operator implementation file.")],
        bench_mode: Annotated[
            str | None,
            Field(description="Optional benchmark mode override. Supported values: standalone, msprof."),
        ] = None,
        case_id: Annotated[str | None, Field(description="Optional benchmark case id to profile.")] = None,
        bench: Annotated[int | None, Field(description="Optional numeric benchmark case index to profile.")] = None,
        kernel_name: Annotated[str | None, Field(description="Optional kernel name filter for profiling.")] = None,
        target_op: Annotated[
            str | None,
            Field(description="Optional operator name to highlight in the generated profile summary."),
        ] = None,
        remote: Annotated[str | None, Field(description="Optional remote execution target.")] = None,
        remote_workdir: Annotated[str | None, Field(description="Optional remote workspace root override.")] = None,
    ) -> dict[str, object]:
        workspace = current_workspace()
        arguments = [
            "--bench-file",
            bench_file,
            "--operator-file",
            operator_file,
        ]
        if bench_mode is not None:
            arguments.extend(["--bench-mode", bench_mode])
        if case_id is not None:
            arguments.extend(["--case-id", case_id])
        if bench is not None:
            arguments.extend(["--bench", str(bench)])
        if kernel_name is not None:
            arguments.extend(["--kernel-name", kernel_name])
        if target_op is not None:
            arguments.extend(["--target-op", target_op])
        _append_common_remote_arguments(
            arguments,
            remote=remote,
            remote_workdir=remote_workdir,
            keep_remote_workdir=False,
            verbose=False,
        )
        with _lease_device(pool) as leased_device:
            return _run_subcommand(
                "profile-bench",
                arguments,
                leased_device=leased_device,
                workspace=workspace,
            )

    @server.tool(
        name="profile-report",
        description="Summarize an existing profile directory without running a new benchmark.",
    )
    def profile_report(
        profile_dir: Annotated[str, Field(description="Absolute path to an existing profile output directory.")],
        target_op: Annotated[str | None, Field(description="Optional operator name filter for the summary.")] = None,
        format: Annotated[str | None, Field(description="Optional report format override.")] = None,
        top: Annotated[int | None, Field(description="Optional number of top items to include in the summary.")] = None,
    ) -> dict[str, object]:
        workspace = current_workspace()
        arguments = ["--profile-dir", profile_dir]
        if target_op is not None:
            arguments.extend(["--target-op", target_op])
        if format is not None:
            arguments.extend(["--format", format])
        if top is not None:
            arguments.extend(["--top", str(top)])
        return _run_subcommand(
            "profile-report",
            arguments,
            leased_device=None,
            workspace=workspace,
        )

    @server.tool(
        name="compare-perf",
        description="Compare two existing perf artifacts and report latency regressions or improvements.",
    )
    def compare_perf(
        baseline: Annotated[str, Field(description="Absolute path to the baseline perf artifact.")],
        compare: Annotated[str, Field(description="Absolute path to the candidate perf artifact.")],
        skip_error: Annotated[
            bool,
            Field(description="Skip parse errors encountered while reading perf artifacts and continue comparing valid entries."),
        ] = False,
        metric_source: Annotated[
            str | None,
            Field(description="Metric source selection for the comparison view. Supported values: auto, kernel, total-op, all."),
        ] = None,
    ) -> dict[str, object]:
        workspace = current_workspace()
        arguments = ["--baseline", baseline, "--compare", compare]
        if skip_error:
            arguments.append("--skip-error")
        if metric_source is not None:
            arguments.extend(["--metric-source", metric_source])
        return _run_subcommand(
            "compare-perf",
            arguments,
            leased_device=None,
            workspace=workspace,
        )

    return server


def current_workspace() -> Path:
    request = get_http_request()
    raw_query = request.scope.get("query_string", b"")
    parsed = parse_qs(raw_query.decode("utf-8"))
    values = parsed.get("workspace", [])
    if not values:
        raise ValueError("workspace query parameter is required for run-eval MCP requests.")
    workspace = Path(values[-1]).expanduser()
    if not workspace.is_absolute():
        raise ValueError(f"workspace query parameter must be an absolute path: {workspace}")
    return workspace.resolve()


def build_mcp_url(*, port: int, workspace: Path) -> str:
    encoded_workspace = quote(str(workspace.resolve()), safe="/")
    return f"http://{_HOST}:{port}{_MCP_PATH}?workspace={encoded_workspace}"


def _build_run_test_arguments(
    *,
    test_file: str,
    operator_file: str,
    test_mode: str | None,
    compare_level: str | None,
    baseline_result: str | None,
    baseline_operator_file: str | None,
    remote: str | None,
    remote_workdir: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
) -> list[str]:
    arguments = [
        "--test-file",
        test_file,
        "--operator-file",
        operator_file,
    ]
    if test_mode is not None:
        arguments.extend(["--test-mode", test_mode])
    if compare_level is not None:
        arguments.extend(["--compare-level", compare_level])
    if baseline_result is not None:
        arguments.extend(["--baseline-result", baseline_result])
    if baseline_operator_file is not None:
        arguments.extend(["--baseline-operator-file", baseline_operator_file])
    _append_common_remote_arguments(
        arguments,
        remote=remote,
        remote_workdir=remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
    )
    return arguments


def _append_common_remote_arguments(
    arguments: list[str],
    *,
    remote: str | None,
    remote_workdir: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
) -> None:
    if remote is not None:
        arguments.extend(["--remote", remote])
    if remote_workdir is not None:
        arguments.extend(["--remote-workdir", remote_workdir])
    if keep_remote_workdir:
        arguments.append("--keep-remote-workdir")
    if verbose:
        arguments.append("--verbose")


@contextmanager
def _lease_device(slot_pool: NpuDevicePool) -> Iterator[str]:
    with slot_pool.acquire() as leased_device:
        yield leased_device


def _run_subcommand(
    subcommand: str,
    arguments: list[str],
    *,
    leased_device: str | None,
    workspace: Path,
) -> dict[str, object]:
    run_command_script = operator_eval_script_path("run-command")
    command = [sys.executable, str(run_command_script), subcommand, *arguments]
    env = dict(os.environ)
    if leased_device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = leased_device
    completed = subprocess.run(
        command,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        env=env,
    )
    return _result_from_completed_process(subcommand=subcommand, completed=completed)


def _result_from_completed_process(
    *,
    subcommand: str,
    completed: subprocess.CompletedProcess[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    for line in completed.stdout.splitlines():
        if line.startswith("Perf file: "):
            payload["perf_path"] = line.removeprefix("Perf file: ").strip()
        elif line.startswith("Archived result: "):
            payload["archived_result"] = line.removeprefix("Archived result: ").strip()
        elif line.startswith("Profile directory: "):
            payload["profile_dir"] = line.removeprefix("Profile directory: ").strip()
        elif line.startswith("Remote workspace: "):
            payload["remote_workspace"] = line.removeprefix("Remote workspace: ").strip()
    if completed.returncode != 0 and "error" not in payload:
        payload["error"] = f"{subcommand} failed with return code {completed.returncode}"
    return payload


def _reserved_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((_HOST, 0))
    sock.listen(2048)
    return sock


def _reserved_socket_for_port(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((_HOST, port))
    sock.listen(2048)
    return sock


@dataclass
class RunningHttpMCPServer:
    server: "FastMCP"
    port: int
    _uvicorn_server: Any
    _thread: threading.Thread

    @property
    def endpoint(self) -> str:
        return f"http://{_HOST}:{self.port}{_MCP_PATH}"

    def url_for_workspace(self, workspace: Path) -> str:
        return build_mcp_url(port=self.port, workspace=workspace)

    def close(self) -> None:
        self._uvicorn_server.should_exit = True
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("Timed out while stopping managed run-eval MCP server.")


def start_http_server(*, port: int = 0) -> RunningHttpMCPServer:
    uvicorn = _load_uvicorn()
    server = create_server()
    app = server.http_app(path=_MCP_PATH, transport="http")
    sock = _reserved_socket() if port == 0 else _reserved_socket_for_port(port)
    port = cast_port(sock)
    config = uvicorn.Config(
        app,
        host=_HOST,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    uvicorn_server = uvicorn.Server(config)
    ready = threading.Event()
    error: list[BaseException] = []

    def _serve() -> None:
        try:
            async def _run() -> None:
                await uvicorn_server.serve(sockets=[sock])
            asyncio.run(_run())
        except BaseException as exc:  # pragma: no cover - defensive boundary
            error.append(exc)
        finally:
            ready.set()

    thread = threading.Thread(target=_serve, name="run-eval-mcp-http", daemon=True)
    thread.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if error:
            raise RuntimeError("Managed run-eval MCP server failed to start.") from error[0]
        if uvicorn_server.started:
            ready.set()
            break
        time.sleep(0.01)
    if not uvicorn_server.started:
        ready.wait(timeout=0.1)
        if error:
            raise RuntimeError("Managed run-eval MCP server failed to start.") from error[0]
        raise RuntimeError("Timed out while starting managed run-eval MCP server.")
    return RunningHttpMCPServer(
        server=server,
        port=port,
        _uvicorn_server=uvicorn_server,
        _thread=thread,
    )


def serve_http_server_forever(*, port: int = 0) -> int:
    server = start_http_server(port=port)
    try:
        print(f"Run-eval MCP server listening at {server.endpoint}")
        print(f"Workspace URL template: {build_mcp_url(port=server.port, workspace=Path('/abs/workspace'))}")
        _serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
    return 0


def _serve_forever() -> None:
    while True:
        time.sleep(3600)


def cast_port(sock: socket.socket) -> int:
    host, port = sock.getsockname()[:2]
    assert host == _HOST
    return int(port)

def get_http_request() -> Any:
    _, request_loader = _load_fastmcp_dependencies()
    return request_loader()


def Field(*args: Any, **kwargs: Any) -> Any:
    from pydantic import Field as pydantic_field

    return pydantic_field(*args, **kwargs)


def _load_fastmcp_dependencies() -> tuple["type[FastMCP]", Callable[[], Any]]:
    from fastmcp import FastMCP
    from fastmcp.server.dependencies import get_http_request

    return FastMCP, get_http_request


def _load_uvicorn() -> Any:
    import uvicorn

    return uvicorn
