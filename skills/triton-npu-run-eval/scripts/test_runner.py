from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib.util
import shutil
import sys
import traceback
from io import StringIO
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TextIO, cast

from run_runtime import (
    env_int,
    ResultPayload,
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    make_result,
    result_succeeded,
    run_streaming_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
)


def _test_timeout() -> int:
    return env_int("TRITON_AGENT_TEST_TIMEOUT_SECONDS", 900)


@dataclass(frozen=True)
class DifferentialTestCase:
    case_id: str
    fn: Callable[[], object]


def _differential_archive_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
) -> tuple[ResultPayload, Path | None]:
    if test_mode == "differential" and _has_differential_test_contract(test_file):
        archive_path = _differential_archive_path(operator_file)
        result = _run_declarative_differential_test(test_file, operator_file, archive_path)
        archived_result = archive_path if result_succeeded(result) else None
        return result, archived_result
    return _run_legacy_local_test(test_file, operator_file, test_mode)


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in test_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if metadata:
                break
            continue
        if not stripped.startswith("#"):
            break
        body = stripped[1:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def load_differential_test_cases(
    test_file: Path,
    operator_file: Path,
) -> list[DifferentialTestCase]:
    test_path = test_file.resolve()
    operator_path = operator_file.resolve()
    test_module = _load_module(test_path, f"differential_test_{test_path.stem}")
    build_operator_api = _require_callable(test_module, "build_operator_api", test_path)
    build_cases = _require_callable(test_module, "build_differential_test_cases", test_path)
    operator_module = _load_module(operator_path, f"differential_operator_{operator_path.stem}")
    operator_api = build_operator_api(operator_module)
    raw_cases = build_cases(operator_api)
    return _normalize_differential_cases(raw_cases)


def _has_differential_test_contract(test_file: Path) -> bool:
    test_path = test_file.resolve()
    test_module = _load_module(test_path, f"differential_test_{test_path.stem}")
    return callable(getattr(test_module, "build_operator_api", None)) and callable(
        getattr(test_module, "build_differential_test_cases", None)
    )


def _normalize_differential_cases(raw_cases: object) -> list[DifferentialTestCase]:
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, Mapping) or not isinstance(raw_cases, Iterable):
        raise ValueError("Differential test hook 'build_differential_test_cases' must return an iterable of cases")
    cases: list[DifferentialTestCase] = []
    seen_case_ids: set[str] = set()
    for raw_case in cast(Iterable[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError("Differential test cases must be mappings")
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Differential test case is missing required string field 'id'")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate differential test case id: {case_id}")
        case_fn = case_map.get("fn")
        if not callable(case_fn):
            raise ValueError(f"Differential test case '{case_id}' is missing required callable field 'fn'")
        seen_case_ids.add(case_id)
        cases.append(DifferentialTestCase(case_id=case_id, fn=cast(Callable[[], object], case_fn)))
    if not cases:
        raise ValueError("Differential test hook 'build_differential_test_cases' returned no cases")
    return cases


def _run_declarative_differential_test(
    test_file: Path,
    operator_file: Path,
    archive_path: Path,
) -> ResultPayload:
    try:
        import torch
    except ImportError as exc:
        return make_result(
            return_code=1,
            stdout="",
            stderr=f"Missing differential test dependency: {exc}",
        )

    try:
        cases = load_differential_test_cases(test_file, operator_file)
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        results: list[object] = []
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            for case in cases:
                results.append(case.fn())
                _synchronize(torch)
        torch.save({"results": results}, archive_path)
        return make_result(
            return_code=0,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )
    except Exception:
        return make_result(
            return_code=1,
            stdout="",
            stderr=traceback.format_exc(),
        )


def _run_legacy_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
) -> tuple[ResultPayload, Path | None]:
    command = [sys.executable, str(test_file), "--operator-file", str(operator_file)]
    result = run_streaming_process(command, str(test_file.parent), stall_timeout_seconds=_test_timeout())
    archived_result = None
    if test_mode == "differential" and result_succeeded(result):
        archived_result = archive_differential_result(test_file, operator_file)
    return result, archived_result


def _run_legacy_remote_test(
    spec: RemoteSpec,
    remote_workspace: str,
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    result = run_remote_command_streaming(
        spec,
        remote_workspace,
        ["python3", test_file.name, "--operator-file", operator_file.name],
        stall_timeout_seconds=_test_timeout(),
        verbose=verbose,
        stderr=stderr,
    )
    archived_result = None
    if test_mode == "differential" and result_succeeded(result):
        archived_result = _copy_remote_differential_result(
            spec,
            remote_workspace,
            test_file,
            operator_file,
            verbose=verbose,
            stderr=stderr,
        )
    return result, archived_result, remote_workspace


def _build_remote_differential_command(test_name: str, operator_name: str) -> list[str]:
    remote_script = f"""
import importlib
import importlib.util
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

def _load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(f"{{module_name}}_{{Path(module_path).stem}}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {{module_path}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _require_callable(module, name, test_path):
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Differential test module missing required hook '{{name}}': {{test_path}}")
    return candidate

def _synchronize(torch_module):
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()

test_file = Path({test_name!r})
operator_file = Path({operator_name!r})
archive_file = operator_file.parent / f"{{operator_file.stem}}_result.pt"
try:
    torch = importlib.import_module("torch")
    test_module = _load_module(test_file, f"differential_test_{{test_file.stem}}")
    build_operator_api = _require_callable(test_module, "build_operator_api", test_file)
    build_cases = _require_callable(test_module, "build_differential_test_cases", test_file)
    operator_module = _load_module(operator_file, f"differential_operator_{{operator_file.stem}}")
    operator_api = build_operator_api(operator_module)
    raw_cases = build_cases(operator_api)
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, dict) or not hasattr(raw_cases, "__iter__"):
        raise ValueError("Differential test hook 'build_differential_test_cases' must return an iterable of cases")
    results = []
    seen_case_ids = set()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError("Differential test cases must be mappings")
            case_id = raw_case.get("id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError("Differential test case is missing required string field 'id'")
            if case_id in seen_case_ids:
                raise ValueError(f"Duplicate differential test case id: {{case_id}}")
            case_fn = raw_case.get("fn")
            if not callable(case_fn):
                raise ValueError(f"Differential test case '{{case_id}}' is missing required callable field 'fn'")
            seen_case_ids.add(case_id)
            results.append(case_fn())
            _synchronize(torch)
    torch.save({{"results": results}}, archive_file)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
"""
    return ["python3", "-c", remote_script]


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{module_name}_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_callable(module: object, name: str, test_path: Path) -> Callable[..., Any]:
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Differential test module missing required hook '{name}': {test_path}")
    return cast(Callable[..., Any], candidate)


def _synchronize(torch_module: Any) -> None:
    if hasattr(torch_module, "npu"):
        torch_module.npu.synchronize()


def run_remote_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    remote: str,
    remote_workdir: str | None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    remote_test = f"{remote_workspace}/{test_file.name}"
    remote_operator = f"{remote_workspace}/{operator_file.name}"
    local_test_cases = test_file.with_suffix(".json")
    remote_test_cases = f"{remote_workspace}/{local_test_cases.name}"
    try:
        copy_file_to_remote(spec, test_file, remote_test, verbose=verbose, stderr=stderr)
        if local_test_cases.exists():
            copy_file_to_remote(
                spec,
                local_test_cases,
                remote_test_cases,
                verbose=verbose,
                stderr=stderr,
            )
        copy_file_to_remote(spec, operator_file, remote_operator, verbose=verbose, stderr=stderr)
        if test_mode == "differential" and _has_differential_test_contract(test_file):
            archive_path = _differential_archive_path(operator_file)
            result = run_remote_command_streaming(
                spec,
                remote_workspace,
                _build_remote_differential_command(test_file.name, operator_file.name),
                stall_timeout_seconds=_test_timeout(),
                verbose=verbose,
                stderr=stderr,
            )
            archived_result = None
            if result_succeeded(result):
                archived_result = _copy_remote_differential_archive(
                    spec,
                    remote_workspace,
                    archive_path,
                    verbose=verbose,
                    stderr=stderr,
                )
            return result, archived_result, remote_workspace
        return _run_legacy_remote_test(
            spec,
            remote_workspace,
            test_file,
            operator_file,
            test_mode,
            verbose=verbose,
            stderr=stderr,
        )
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def find_case_insensitive_result_file(directory: Path) -> Path | None:
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.name.lower() == "test_result.pt":
            return candidate
    return None


def archive_differential_result(test_file: Path, operator_file: Path) -> Path:
    result_file = find_case_insensitive_result_file(operator_file.parent)
    if result_file is None and test_file.parent != operator_file.parent:
        result_file = find_case_insensitive_result_file(test_file.parent)
    if result_file is None:
        raise FileNotFoundError(
            f"Differential result payload not found beside operator or test file: {operator_file.parent}"
        )

    archive_name = f"{operator_file.stem}_result.pt"
    archive_path = operator_file.parent / archive_name
    shutil.copy2(result_file, archive_path)
    return archive_path
def _copy_remote_differential_result(
    spec: RemoteSpec,
    remote_workspace: str,
    test_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> Path:
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        (
            "python3 -c "
            + repr(
                "import pathlib; "
                "matches = sorted(p.name for p in pathlib.Path('.').iterdir() "
                "if p.is_file() and p.name.lower() == 'test_result.pt'); "
                "print(matches[0] if matches else '')"
            )
        ),
        verbose=verbose,
        stderr=stderr,
    )
    stdout = str(result["stdout"])
    remote_name = stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""
    if not remote_name:
        raise FileNotFoundError(
            f"Differential result payload not found in remote workspace for {test_file.name}"
        )
    archive_path = operator_file.parent / f"{operator_file.stem}_result.pt"
    copy_file_from_remote(
        spec,
        f"{remote_workspace}/{remote_name}",
        archive_path,
        verbose=verbose,
        stderr=stderr,
    )
    return archive_path


def _copy_remote_differential_archive(
    spec: RemoteSpec,
    remote_workspace: str,
    archive_path: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> Path:
    copy_file_from_remote(
        spec,
        f"{remote_workspace}/{archive_path.name}",
        archive_path,
        verbose=verbose,
        stderr=stderr,
    )
    return archive_path
