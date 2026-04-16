#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from typing import NamedTuple


class CaptureDetails(NamedTuple):
    dumped_ir_dir: str
    compile_command: list[str]


def make_result(
    *,
    return_code: int,
    stdout: str,
    stderr: str,
    stalled: bool = False,
    session_id: str | None = None,
) -> dict[str, object]:
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "stalled": stalled,
        "session_id": session_id,
    }


def extract_capture_details(stdout: str) -> CaptureDetails:
    dumped_ir_dir: str | None = None
    raw_compile_command: str | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Dumping intermediate results to "):
            dumped_ir_dir = stripped.removeprefix("Dumping intermediate results to ").strip()
        elif stripped.startswith("[DEBUG] cmd_list: "):
            raw_compile_command = stripped.removeprefix("[DEBUG] cmd_list: ").strip()

    if dumped_ir_dir is None:
        raise RuntimeError("Missing 'Dumping intermediate results to ...' line in command stdout.")
    if raw_compile_command is None:
        raise RuntimeError("Missing '[DEBUG] cmd_list: ...' line in command stdout.")
    return CaptureDetails(
        dumped_ir_dir=dumped_ir_dir,
        compile_command=_normalize_compile_command_tokens(shlex.split(raw_compile_command)),
    )


def rewrite_compile_command(
    command: list[str],
    *,
    archived_input: Path,
    stage_dir: Path,
) -> list[str]:
    command = _normalize_compile_command_tokens(command)
    if len(command) < 2:
        raise RuntimeError(f"Unexpected compile command shape: {command}")

    rewritten: list[str] = [command[0]]
    input_replaced = False
    index = 1
    while index < len(command):
        token = command[index]
        if token.startswith("--bishengir-print-ir-after="):
            index += 1
            continue
        if token == "--bishengir-print-ir-after":
            index += 2
            continue
        if token.startswith("--mlir-print-ir-tree-dir="):
            index += 1
            continue
        if token == "--mlir-print-ir-after-all":
            index += 1
            continue
        if not input_replaced and _looks_like_ttadapter_input(token):
            rewritten.append(str(archived_input))
            input_replaced = True
            index += 1
            continue
        rewritten.append(token)
        index += 1

    if not input_replaced:
        rewritten.insert(1, str(archived_input))

    rewritten.append("--mlir-print-ir-after-all")
    rewritten.append(f"--mlir-print-ir-tree-dir={stage_dir}")
    return rewritten


def write_manifest(
    archive_dir: Path,
    *,
    bench_file: Path,
    operator_file: Path,
    rendered_command: list[str],
    remote: str | None,
    dumped_ir_dir: str,
    original_compile_command: list[str],
    replay_compile_command: list[str],
    archived_input: Path,
) -> Path:
    manifest_path = archive_dir / "capture-manifest.json"
    payload = {
        "bench_file": str(bench_file),
        "operator_file": str(operator_file),
        "rendered_command": rendered_command,
        "remote": remote,
        "dumped_ir_dir": dumped_ir_dir,
        "original_compile_command": original_compile_command,
        "replay_compile_command": replay_compile_command,
        "archived_input": str(archived_input),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def capture_local_archive(
    *,
    bench_file: Path,
    operator_file: Path,
    archive_dir: Path,
) -> Path:
    _prepare_empty_archive_dir(archive_dir)
    command = build_execution_command(bench_file=bench_file, operator_file=operator_file)
    result = _run_local_command(command, cwd=bench_file.parent)
    if int(result["return_code"]) != 0:
        raise RuntimeError(str(result["stderr"]) or str(result["stdout"]) or "Command failed.")

    details = extract_capture_details(str(result["stdout"]))
    triton_dump_dir = archive_dir / "triton_dump"
    shutil.copytree(Path(details.dumped_ir_dir), triton_dump_dir)
    archived_input = _resolve_archived_input(triton_dump_dir)
    replay_command = rewrite_compile_command(
        details.compile_command,
        archived_input=archived_input,
        stage_dir=archive_dir / "bishengir_stages",
    )
    _run_local_replay(replay_command, archive_dir / "all-ir.txt")
    return write_manifest(
        archive_dir,
        bench_file=bench_file,
        operator_file=operator_file,
        rendered_command=command,
        remote=None,
        dumped_ir_dir=details.dumped_ir_dir,
        original_compile_command=details.compile_command,
        replay_compile_command=replay_command,
        archived_input=archived_input,
    )


def capture_remote_archive(
    *,
    bench_file: Path,
    operator_file: Path,
    archive_dir: Path,
    remote: str,
    remote_workdir: str | None,
    keep_remote_workdir: bool,
    verbose: bool = False,
    stderr=None,
) -> tuple[Path, str]:
    _prepare_archive_destination(archive_dir)
    spec, remote_root = create_remote_workspace(remote, remote_workdir, verbose=verbose, stderr=stderr)
    remote_source_dir = f"{remote_root}/workspace"
    remote_archive_dir = f"{remote_root}/archive"
    remote_bench_file = Path(bench_file.name)
    remote_operator_file = Path(operator_file.name)
    command = build_execution_command(
        bench_file=remote_bench_file,
        operator_file=remote_operator_file,
        python_executable="python3",
    )
    try:
        run_remote_command_buffered(
            spec,
            remote_root,
            f"mkdir -p {shlex.quote(remote_source_dir)} {shlex.quote(remote_archive_dir)}",
            verbose=verbose,
            stderr=stderr,
        )
        _stage_required_files(
            spec,
            bench_file=bench_file,
            operator_file=operator_file,
            remote_source_dir=remote_source_dir,
            verbose=verbose,
            stderr=stderr,
        )
        run_result = _run_remote_debug_command(
            spec,
            remote_source_dir,
            command,
            verbose=verbose,
            stderr=stderr,
        )
        if int(run_result["return_code"]) != 0:
            raise RuntimeError(str(run_result["stderr"]) or str(run_result["stdout"]) or "Remote command failed.")

        details = extract_capture_details(str(run_result["stdout"]))
        remote_copy_command = (
            f"rm -rf {shlex.quote(remote_archive_dir)}/triton_dump "
            f"{shlex.quote(remote_archive_dir)}/bishengir_stages "
            f"{shlex.quote(remote_archive_dir)}/all-ir.txt && "
            f"cp -R {shlex.quote(details.dumped_ir_dir)} {shlex.quote(remote_archive_dir)}/triton_dump"
        )
        copy_result = run_remote_command_buffered(
            spec,
            remote_root,
            remote_copy_command,
            verbose=verbose,
            stderr=stderr,
        )
        if int(copy_result["return_code"]) != 0:
            raise RuntimeError(
                str(copy_result["stderr"]) or str(copy_result["stdout"]) or "Failed to archive remote Triton dump."
            )

        remote_archived_input = PurePosixPath(remote_archive_dir) / "triton_dump" / "kernel.ttadapter.mlir"
        replay_command = rewrite_compile_command(
            details.compile_command,
            archived_input=Path(str(remote_archived_input)),
            stage_dir=Path(remote_archive_dir) / "bishengir_stages",
        )
        replay_result = run_remote_command_buffered(
            spec,
            remote_root,
            _build_remote_replay_command(replay_command, remote_archive_dir),
            verbose=verbose,
            stderr=stderr,
        )
        if int(replay_result["return_code"]) != 0:
            raise RuntimeError(
                _format_failed_command_message(
                    "Failed to replay remote compile command",
                    _build_remote_replay_command(replay_command, remote_archive_dir),
                    stdout=str(replay_result["stdout"]),
                    stderr=str(replay_result["stderr"]),
                )
            )

        copy_directory_from_remote(
            spec,
            f"{remote_root}/archive",
            archive_dir,
            verbose=verbose,
            stderr=stderr,
        )
        archived_input = _resolve_archived_input(archive_dir / "triton_dump")
        manifest_path = write_manifest(
            archive_dir,
            bench_file=bench_file,
            operator_file=operator_file,
            rendered_command=command,
            remote=remote,
            dumped_ir_dir=details.dumped_ir_dir,
            original_compile_command=details.compile_command,
            replay_compile_command=replay_command,
            archived_input=archived_input,
        )
        return manifest_path, remote_root
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_root, verbose=verbose, stderr=stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture Triton Ascend IR into an IR directory.")
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--bench-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--remote")
    parser.add_argument("--remote-workdir")
    parser.add_argument("--keep-remote-workdir", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive_dir = Path(args.ir_dir).expanduser().resolve()
    bench_file = _resolve_existing_path(args.bench_file, "Bench file")
    operator_file = _resolve_existing_path(args.operator_file, "Operator file")
    try:
        if args.remote:
            manifest_path, remote_workspace = capture_remote_archive(
                bench_file=bench_file,
                operator_file=operator_file,
                archive_dir=archive_dir,
                remote=args.remote,
                remote_workdir=args.remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
            print(f"Capture manifest: {manifest_path}")
            if args.keep_remote_workdir:
                print(f"Remote workspace: {remote_workspace}")
            return 0

        manifest_path = capture_local_archive(
            bench_file=bench_file,
            operator_file=operator_file,
            archive_dir=archive_dir,
        )
        print(f"Capture manifest: {manifest_path}")
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _prepare_empty_archive_dir(archive_dir: Path) -> None:
    if archive_dir.exists():
        raise FileExistsError(f"IR directory already exists: {archive_dir}")
    archive_dir.mkdir(parents=True)


def _prepare_archive_destination(archive_dir: Path) -> None:
    if archive_dir.exists():
        raise FileExistsError(f"IR directory already exists: {archive_dir}")
    archive_dir.parent.mkdir(parents=True, exist_ok=True)


def _resolve_existing_path(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    return path


def _run_local_command(command: list[str], *, cwd: Path) -> dict[str, object]:
    env = dict(os.environ)
    env["TRITON_DEBUG"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return make_result(
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _run_local_replay(command: list[str], stderr_path: Path) -> None:
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            _format_failed_command_message(
                "Replay compile command failed",
                _shell_join_command(command),
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )


def _normalize_compile_command_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--append-bisheng-options="):
            value = token.split("=", 1)[1]
            merged: list[str] = [value] if value else []
            index += 1
            while index < len(tokens) and not tokens[index].startswith("-"):
                merged.append(tokens[index])
                index += 1
            normalized.append(f"--append-bisheng-options={' '.join(merged).strip()}")
            continue
        normalized.append(token)
        index += 1
    return normalized


def _looks_like_ttadapter_input(token: str) -> bool:
    return token.endswith("kernel.ttadapter.mlir") or token.endswith(".ttadapter.mlir")


def build_execution_command(
    *,
    bench_file: Path,
    operator_file: Path,
    python_executable: str | None = None,
) -> list[str]:
    operator_arg = operator_file.name
    if bench_file.parent != operator_file.parent:
        operator_arg = os.path.relpath(operator_file, bench_file.parent)
    interpreter = sys.executable if python_executable is None else python_executable
    return [
        interpreter,
        bench_file.name,
        "--operator-file",
        operator_arg,
    ]


def _resolve_archived_input(triton_dump_dir: Path) -> Path:
    matches = sorted(triton_dump_dir.rglob("kernel.ttadapter.mlir"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one kernel.ttadapter.mlir under {triton_dump_dir}, found {len(matches)}."
        )
    return matches[0]


def _stage_required_files(
    spec: Any,
    *,
    bench_file: Path,
    operator_file: Path,
    remote_source_dir: str,
    verbose: bool = False,
    stderr=None,
) -> None:
    if bench_file.name == operator_file.name and bench_file.resolve() != operator_file.resolve():
        raise RuntimeError(
            f"Remote staging would collide on basename '{bench_file.name}': {bench_file} vs {operator_file}"
        )
    copy_file_to_remote(
        spec,
        bench_file.resolve(),
        f"{remote_source_dir}/{bench_file.name}",
        verbose=verbose,
        stderr=stderr,
    )
    copy_file_to_remote(
        spec,
        operator_file.resolve(),
        f"{remote_source_dir}/{operator_file.name}",
        verbose=verbose,
        stderr=stderr,
    )


def _run_remote_debug_command(
    spec: Any,
    remote_source_dir: str,
    command: list[str],
    *,
    verbose: bool = False,
    stderr=None,
) -> dict[str, object]:
    remote_command = "export TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1 && " + _shell_join_command(command)
    return run_remote_command_buffered(
        spec,
        remote_source_dir,
        remote_command,
        verbose=verbose,
        stderr=stderr,
    )


def _build_remote_replay_command(command: list[str], remote_archive_dir: str) -> str:
    stderr_path = f"{remote_archive_dir}/all-ir.txt"
    return (
        f"mkdir -p {shlex.quote(remote_archive_dir)}/bishengir_stages && "
        f"{_shell_join_command(command)} 2> {shlex.quote(stderr_path)}"
    )


def _shell_join_command(command: list[str]) -> str:
    rendered: list[str] = []
    for token in command:
        if token.startswith("--append-bisheng-options="):
            value = token.split("=", 1)[1]
            rendered.append(f"--append-bisheng-options={shlex.quote(value)}")
            continue
        rendered.append(shlex.quote(token))
    return " ".join(rendered)


def _format_failed_command_message(
    title: str,
    command: str,
    *,
    stdout: str,
    stderr: str,
) -> str:
    parts = [f"{title}: {command}"]
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.rstrip()}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.rstrip()}")
    return "\n".join(parts)


def _load_runtime_helpers() -> dict[str, Any]:
    script = (
        Path(__file__).resolve().parents[2]
        / "triton-npu-run-eval"
        / "scripts"
        / "run_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("capture_ir_run_runtime", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared runtime helpers from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "create_remote_workspace": module.create_remote_workspace,
        "cleanup_remote_workspace": module.cleanup_remote_workspace,
        "copy_file_to_remote": module.copy_file_to_remote,
        "copy_directory_from_remote": module.copy_directory_from_remote,
        "run_remote_command_buffered": module.run_remote_command_buffered,
    }


_RUNTIME_HELPERS = _load_runtime_helpers()
create_remote_workspace = _RUNTIME_HELPERS["create_remote_workspace"]
cleanup_remote_workspace = _RUNTIME_HELPERS["cleanup_remote_workspace"]
copy_file_to_remote = _RUNTIME_HELPERS["copy_file_to_remote"]
copy_directory_from_remote = _RUNTIME_HELPERS["copy_directory_from_remote"]
run_remote_command_buffered = _RUNTIME_HELPERS["run_remote_command_buffered"]


if __name__ == "__main__":
    raise SystemExit(main())
