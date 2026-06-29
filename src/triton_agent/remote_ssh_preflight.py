from __future__ import annotations

import locale
import subprocess
from typing import Protocol, TypedDict, cast

from triton_agent.skill_loader import load_operator_eval_script_module


class RemoteSpec(TypedDict):
    user_host: str
    port: int | None


class RunRuntimeModule(Protocol):
    def parse_remote_spec(self, raw: str) -> RemoteSpec: ...


_AUTH_FAILURE_MARKERS = (
    "permission denied",
    "publickey",
    "too many authentication failures",
    "no supported authentication methods available",
)
_CONNECT_TIMEOUT_SECONDS = 5
_PRECHECK_TIMEOUT_SECONDS = 10


def _parse_remote_spec(remote: str) -> RemoteSpec:
    module = cast(RunRuntimeModule, load_operator_eval_script_module("run_runtime"))
    return module.parse_remote_spec(remote)


def build_remote_ssh_preflight_command(remote: str) -> list[str]:
    spec = _parse_remote_spec(remote)
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        f"ConnectTimeout={_CONNECT_TIMEOUT_SECONDS}",
    ]
    if spec["port"] is not None:
        command.extend(["-p", str(spec["port"])])
    command.extend([spec["user_host"], "true"])
    return command


def format_ssh_copy_id_command(remote: str) -> str:
    spec = _parse_remote_spec(remote)
    if spec["port"] is None:
        return f"ssh-copy-id {spec['user_host']}"
    return f"ssh-copy-id -p {spec['port']} {spec['user_host']}"


def ensure_remote_ssh_ready(remote: str) -> None:
    command = build_remote_ssh_preflight_command(remote)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=False,
            check=False,
            timeout=_PRECHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH preflight timed out while connecting to {remote}.") from exc

    if result.returncode == 0:
        return

    detail = _decode_subprocess_output(result.stderr or result.stdout).strip() or f"SSH preflight failed for {remote}."
    if _is_auth_failure(detail):
        raise RuntimeError(
            f"Remote target {remote} is not ready for key-based SSH access.\n"
            f"Run `{format_ssh_copy_id_command(remote)}` and enter the remote login password to copy your public key."
        )
    raise RuntimeError(detail)


def _is_auth_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _AUTH_FAILURE_MARKERS)


def _decode_subprocess_output(data: bytes | str | None) -> str:
    # This intentionally duplicates the UTF-8-first decoding logic from the
    # staged run_runtime skill helper instead of importing skill-side code into
    # src/.
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        preferred = locale.getpreferredencoding(False) or "utf-8"
        return data.decode(preferred, errors="replace")
