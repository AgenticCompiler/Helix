from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TextIO

from triton_agent.models import AgentResult
from triton_agent.process_runner import run_process
from triton_agent.verbose import emit_verbose


@dataclass(frozen=True)
class RemoteSpec:
    user_host: str
    port: int | None


def parse_remote_spec(raw: str) -> RemoteSpec:
    if "@" not in raw:
        raise ValueError(f"Remote target must be in user@host[:port] form: {raw}")
    if ":" not in raw:
        return RemoteSpec(user_host=raw, port=None)

    user_host, possible_port = raw.rsplit(":", 1)
    if not possible_port.isdigit():
        raise ValueError(f"Remote target port must be numeric: {raw}")
    return RemoteSpec(user_host=user_host, port=int(possible_port))


def create_remote_workspace(
    remote: str,
    remote_workdir: str | None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[RemoteSpec, str]:
    spec = parse_remote_spec(remote)
    if remote_workdir:
        root = shlex.quote(remote_workdir)
        pattern = shlex.quote(str(PurePosixPath(remote_workdir) / "triton-agent-XXXXXX"))
        remote_command = f"mkdir -p {root} && mktemp -d {pattern}"
    else:
        remote_command = "mktemp -d"
    command = _ssh_command(spec, remote_command)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_process(command, ".", mode="buffered", stall_timeout_seconds=120)
    if not result.succeeded:
        raise RuntimeError(result.stderr or result.stdout or "Failed to create remote workspace.")
    workspace = result.stdout.strip().splitlines()[-1].strip()
    if not workspace:
        raise RuntimeError("Remote workspace command did not return a path.")
    return spec, workspace


def cleanup_remote_workspace(
    spec: RemoteSpec,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _ssh_command(spec, f"rm -rf {shlex.quote(remote_workspace)}")
    _maybe_emit_remote_command(command, verbose, stderr)
    run_process(
        command,
        ".",
        mode="buffered",
        stall_timeout_seconds=120,
    )


def copy_file_to_remote(
    spec: RemoteSpec,
    local_path: Path,
    remote_path: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _scp_to_remote_command(spec, local_path, remote_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_process(
        command,
        ".",
        mode="buffered",
        stall_timeout_seconds=300,
    )
    if not result.succeeded:
        raise RuntimeError(result.stderr or result.stdout or f"Failed to copy {local_path} to remote.")


def copy_file_from_remote(
    spec: RemoteSpec,
    remote_path: str,
    local_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    command = _scp_from_remote_command(spec, remote_path, local_path)
    _maybe_emit_remote_command(command, verbose, stderr)
    result = run_process(
        command,
        ".",
        mode="buffered",
        stall_timeout_seconds=300,
    )
    if not result.succeeded:
        raise RuntimeError(result.stderr or result.stdout or f"Failed to copy {remote_path} from remote.")


def run_remote_command_streaming(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> AgentResult:
    command = _ssh_command(spec, f"cd {shlex.quote(remote_workspace)} && {remote_command}")
    _maybe_emit_remote_command(command, verbose, stderr)
    return run_process(
        command,
        ".",
        mode="streaming",
        stall_timeout_seconds=900,
    )


def run_remote_command_buffered(
    spec: RemoteSpec,
    remote_workspace: str,
    remote_command: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> AgentResult:
    command = _ssh_command(spec, f"cd {shlex.quote(remote_workspace)} && {remote_command}")
    _maybe_emit_remote_command(command, verbose, stderr)
    return run_process(
        command,
        ".",
        mode="buffered",
        stall_timeout_seconds=900,
    )


def _ssh_command(spec: RemoteSpec, remote_command: str) -> list[str]:
    command = ["ssh"]
    if spec.port is not None:
        command.extend(["-p", str(spec.port)])
    command.extend([spec.user_host, f"bash -lc {shlex.quote(remote_command)}"])
    return command


def _scp_to_remote_command(spec: RemoteSpec, local_path: Path, remote_path: str) -> list[str]:
    command = ["scp"]
    if spec.port is not None:
        command.extend(["-P", str(spec.port)])
    command.extend([str(local_path), f"{spec.user_host}:{remote_path}"])
    return command


def _scp_from_remote_command(spec: RemoteSpec, remote_path: str, local_path: Path) -> list[str]:
    command = ["scp"]
    if spec.port is not None:
        command.extend(["-P", str(spec.port)])
    command.extend([f"{spec.user_host}:{remote_path}", str(local_path)])
    return command


def _maybe_emit_remote_command(command: list[str], verbose: bool, stderr: TextIO | None) -> None:
    if not verbose or stderr is None:
        return
    emit_verbose(stderr, "remote", f"command: {shlex.join(command)}")
