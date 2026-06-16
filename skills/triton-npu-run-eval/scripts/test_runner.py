from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import os
import sys
import textwrap
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, TextIO, cast

from debug_device import maybe_print_visible_devices
from run_runtime import (
    ResultPayload,
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    env_int,
    make_result,
    result_succeeded,
    run_remote_command_streaming,
)


SCRIPT_DIR = Path(__file__).resolve().parent
_WARNING_PREFIX = "[WARNING]"


def _test_timeout() -> int:
    return env_int("TRITON_AGENT_TEST_TIMEOUT_SECONDS", 900)


@dataclass(frozen=True)
class DifferentialTestCase:
    case_id: str
    inputs: tuple[object, ...] | list[object]
    fn: Callable[[], object]


def _differential_archive_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    maybe_print_visible_devices()
    if test_mode == "standalone":
        result = _run_import_only_standalone_test(test_file, operator_file)
        return _filter_result_payload(result, verbose=verbose), None
    if test_mode == "differential":
        archive_path = _differential_archive_path(operator_file)
        result = _run_declarative_differential_test(test_file, operator_file, archive_path)
        archived_result = archive_path if result_succeeded(result) else None
        return _filter_result_payload(result, verbose=verbose), archived_result
    raise ValueError(f"Unsupported test mode: {test_mode}")


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
    with _temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
        test_module = _load_module(test_path, f"differential_test_{test_path.stem}")
        build_operator_api = _require_callable(test_module, "build_operator_api", test_path)
        build_cases = _require_callable(test_module, "build_differential_test_cases", test_path)
        operator_module = _load_module(operator_path, f"differential_operator_{operator_path.stem}")
        operator_api = build_operator_api(operator_module)
        raw_cases = build_cases(operator_api)
    return _normalize_differential_cases(raw_cases)


def _normalize_differential_cases(raw_cases: object) -> list[DifferentialTestCase]:
    return [
        DifferentialTestCase(
            case_id=cast(str, record["id"]),
            inputs=cast(tuple[object, ...] | list[object], record["inputs"]),
            fn=cast(Callable[[], object], record["fn"]),
        )
        for record in _normalize_differential_case_records(raw_cases)
    ]


def _normalize_differential_case_records(raw_cases: object) -> list[dict[str, object]]:
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, Mapping) or not isinstance(raw_cases, Iterable):
        raise ValueError("Differential test hook 'build_differential_test_cases' must return an iterable of cases")
    records: list[dict[str, object]] = []
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
        raw_inputs = case_map.get("inputs")
        if not isinstance(raw_inputs, (list, tuple)):
            raise ValueError(
                f"Differential test case '{case_id}' is missing required list/tuple field 'inputs'"
            )
        case_fn = case_map.get("fn")
        if not callable(case_fn):
            raise ValueError(f"Differential test case '{case_id}' is missing required callable field 'fn'")
        seen_case_ids.add(case_id)
        if isinstance(raw_inputs, tuple):
            normalized_inputs: tuple[object, ...] | list[object] = tuple(cast(tuple[object, ...], raw_inputs))
        else:
            normalized_inputs = list(cast(list[object], raw_inputs))
        records.append({"id": case_id, "inputs": normalized_inputs, "fn": case_fn})
    if not records:
        raise ValueError("Differential test hook 'build_differential_test_cases' returned no cases")
    return records


def _run_import_only_standalone_test(
    test_file: Path,
    operator_file: Path,
) -> ResultPayload:
    prev = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        test_path = test_file.resolve()
        operator_path = operator_file.resolve()
        metadata = parse_test_metadata(test_path)
        with _temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
            test_module = _load_module(test_path, f"standalone_test_{test_path.stem}")
            main_fn = _require_callable(test_module, "main", test_path, kind="Standalone test module")
            operator_module = _load_module(operator_path, f"standalone_operator_{operator_path.stem}")
            operator_api = _resolve_operator_api(operator_module, metadata, operator_path)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                main_fn(operator_api)
                _maybe_synchronize_torch()
        return make_result(
            return_code=0,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )
    except Exception:
        return make_result(
            return_code=1,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue() + traceback.format_exc(),
        )
    finally:
        if prev is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev


def _run_declarative_differential_test(
    test_file: Path,
    operator_file: Path,
    archive_path: Path,
) -> ResultPayload:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        return make_result(
            return_code=1,
            stdout="",
            stderr=f"Missing differential test dependency: {exc}",
        )
    prev = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        metadata = parse_test_metadata(test_file)
        compute = _compute_flag_from_metadata(metadata)
        cases = load_differential_test_cases(test_file, operator_file)
        records: list[dict[str, object]] = []
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            for case in cases:
                records.append(
                    {
                        "id": case.case_id,
                        "inputs": case.inputs,
                        "result": case.fn(),
                    }
                )
                _synchronize(torch)
        torch.save({"compute": compute, "cases": records}, archive_path)
        return make_result(
            return_code=0,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
        )
    except Exception:
        return make_result(
            return_code=1,
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue() + traceback.format_exc(),
        )
    finally:
        if prev is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev


def _filter_result_payload(result: ResultPayload, *, verbose: bool) -> ResultPayload:
    if verbose:
        return result
    filtered_stdout = _filter_known_warning_lines(str(result["stdout"]))
    filtered_stderr = _filter_known_warning_lines(str(result["stderr"]))
    if filtered_stdout == result["stdout"] and filtered_stderr == result["stderr"]:
        return result
    return make_result(
        return_code=int(result["return_code"]),
        stdout=filtered_stdout,
        stderr=filtered_stderr,
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )


def _filter_known_warning_lines(text: str) -> str:
    filtered_lines = [
        line
        for line in text.splitlines(keepends=True)
        if not line.rstrip("\r\n").startswith(_WARNING_PREFIX)
    ]
    return "".join(filtered_lines)


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{module_name}_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_callable(
    module: object,
    name: str,
    source_path: Path,
    *,
    kind: str = "Differential test module",
) -> Callable[..., Any]:
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"{kind} missing required hook '{name}': {source_path}")
    return cast(Callable[..., Any], candidate)


def _resolve_operator_api(operator_module: object, metadata: Mapping[str, str], operator_path: Path) -> object:
    api_name = metadata.get("api-name")
    api_kind = metadata.get("api-kind")
    if not api_name:
        raise ValueError(f"Test metadata is missing required 'api-name' entry: {operator_path}")
    if api_kind not in {"triton-wrapper", "torch-function", "torch-module"}:
        raise ValueError(f"Test metadata is missing required 'api-kind' entry: {operator_path}")
    candidate = getattr(operator_module, api_name, None)
    if candidate is None:
        raise ValueError(f"Runtime operator file is missing required API '{api_name}': {operator_path}")
    if api_kind == "torch-module":
        if not callable(candidate):
            raise ValueError(f"Runtime operator API '{api_name}' is not callable: {operator_path}")
        try:
            return candidate()
        except TypeError as exc:
            raise RuntimeError(
                "torch-module entrypoints must support no-argument construction; "
                "constructor arguments are not supported in generated harnesses"
            ) from exc
    return candidate


def _compute_flag_from_metadata(metadata: Mapping[str, str]) -> bool:
    return _parse_compute_kind(metadata.get("compute-kind"))


def _maybe_synchronize_torch() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    _synchronize(torch)


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
    maybe_print_visible_devices()
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    remote_test = f"{remote_workspace}/{test_file.name}"
    remote_operator = f"{remote_workspace}/{operator_file.name}"
    remote_compare_helper = f"{remote_workspace}/npu_compare.py"
    try:
        copy_file_to_remote(spec, test_file, remote_test, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, operator_file, remote_operator, verbose=verbose, stderr=stderr)
        copy_file_to_remote(
            spec,
            SCRIPT_DIR / "npu_compare.py",
            remote_compare_helper,
            verbose=verbose,
            stderr=stderr,
        )
        extra_env = {"TRITON_ALWAYS_COMPILE": "1"}
        if test_mode == "standalone":
            result = run_remote_command_streaming(
                spec,
                remote_workspace,
                _build_remote_standalone_command(test_file.name, operator_file.name),
                stall_timeout_seconds=_test_timeout(),
                verbose=verbose,
                stderr=stderr,
                extra_env=extra_env,
            )
            return _filter_result_payload(result, verbose=verbose), None, remote_workspace
        if test_mode == "differential":
            archive_path = _differential_archive_path(operator_file)
            result = run_remote_command_streaming(
                spec,
                remote_workspace,
                _build_remote_differential_command(test_file.name, operator_file.name),
                stall_timeout_seconds=_test_timeout(),
                verbose=verbose,
                stderr=stderr,
                extra_env=extra_env,
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
            return _filter_result_payload(result, verbose=verbose), archived_result, remote_workspace
        raise ValueError(f"Unsupported test mode: {test_mode}")
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _copy_remote_differential_archive(
    spec: RemoteSpec,
    remote_workspace: str,
    archive_path: Path,
    *,
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


def _build_remote_standalone_command(test_name: str, operator_name: str) -> list[str]:
    remote_script = f"""
import importlib.util
import sys
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

def _parse_metadata(test_file):
    metadata = {{}}
    for line in Path(test_file).read_text(encoding="utf-8").splitlines():
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

def _require_callable(module, name, source_path):
    candidate = getattr(module, name, None)
    if not callable(candidate):
        raise ValueError(f"Standalone test module missing required hook '{{name}}': {{source_path}}")
    return candidate

def _resolve_operator_api(operator_module, metadata, operator_path):
    api_name = metadata.get("api-name")
    api_kind = metadata.get("api-kind")
    if not api_name:
        raise ValueError(f"Test metadata is missing required 'api-name' entry: {{operator_path}}")
    if api_kind not in {{"triton-wrapper", "torch-function", "torch-module"}}:
        raise ValueError(f"Test metadata is missing required 'api-kind' entry: {{operator_path}}")
    candidate = getattr(operator_module, api_name, None)
    if candidate is None:
        raise ValueError(f"Runtime operator file is missing required API '{{api_name}}': {{operator_path}}")
    if api_kind == "torch-module":
        return candidate()
    return candidate

def _maybe_synchronize():
    try:
        import torch
    except ImportError:
        return
    if hasattr(torch, "npu"):
        torch.npu.synchronize()

test_file = Path({test_name!r})
operator_file = Path({operator_name!r})
sys.path.insert(0, str(Path(".").resolve()))
stdout_buffer = StringIO()
stderr_buffer = StringIO()
try:
    metadata = _parse_metadata(test_file)
    test_module = _load_module(test_file, f"standalone_test_{{test_file.stem}}")
    main_fn = _require_callable(test_module, "main", test_file)
    operator_module = _load_module(operator_file, f"standalone_operator_{{operator_file.stem}}")
    operator_api = _resolve_operator_api(operator_module, metadata, operator_file)
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        main_fn(operator_api)
        _maybe_synchronize()
    sys.stdout.write(stdout_buffer.getvalue())
    sys.stderr.write(stderr_buffer.getvalue())
except Exception:
    sys.stdout.write(stdout_buffer.getvalue())
    sys.stderr.write(stderr_buffer.getvalue())
    traceback.print_exc()
    raise SystemExit(1)
"""
    return ["python3", "-c", remote_script]


def _build_remote_differential_command(test_name: str, operator_name: str) -> list[str]:
    remote_script = f"""
import importlib
import importlib.util
import sys
import traceback
from collections.abc import Iterable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import cast

def _load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(f"{{module_name}}_{{Path(module_path).stem}}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {{module_path}}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _parse_metadata(test_file):
    metadata = {{}}
    for line in Path(test_file).read_text(encoding="utf-8").splitlines():
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

{_remote_function_source(_parse_compute_kind)}

{_remote_function_source(_normalize_differential_case_records)}

def _compute_flag(metadata):
    return _parse_compute_kind(metadata.get("compute-kind"))

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
sys.path.insert(0, str(Path(".").resolve()))
try:
    torch = importlib.import_module("torch")
    metadata = _parse_metadata(test_file)
    compute = _compute_flag(metadata)
    test_module = _load_module(test_file, f"differential_test_{{test_file.stem}}")
    build_operator_api = _require_callable(test_module, "build_operator_api", test_file)
    build_cases = _require_callable(test_module, "build_differential_test_cases", test_file)
    operator_module = _load_module(operator_file, f"differential_operator_{{operator_file.stem}}")
    operator_api = build_operator_api(operator_module)
    raw_cases = build_cases(operator_api)
    case_records = _normalize_differential_case_records(raw_cases)
    records = []
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        for case_record in case_records:
            case_fn = case_record["fn"]
            records.append({{"id": case_record["id"], "inputs": case_record["inputs"], "result": case_fn()}})
            _synchronize(torch)
    torch.save({{"compute": compute, "cases": records}}, archive_file)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
"""
    return ["python3", "-c", remote_script]


def _remote_function_source(function: Callable[..., Any]) -> str:
    return textwrap.dedent(inspect.getsource(function)).strip()


@contextmanager
def _temporary_sys_path_entries(*paths: Path) -> Iterator[None]:
    added: list[str] = []
    try:
        for path in paths:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
                added.append(text)
        yield
    finally:
        for path in reversed(added):
            if path in sys.path:
                sys.path.remove(path)


def _parse_compute_kind(raw_value: object) -> bool:
    if raw_value is None:
        return True
    if not isinstance(raw_value, str):
        raise ValueError("Test metadata 'compute-kind' must be 'compute' or 'non-compute'")
    normalized = raw_value.strip().lower()
    if normalized == "compute":
        return True
    if normalized == "non-compute":
        return False
    raise ValueError("Test metadata 'compute-kind' must be 'compute' or 'non-compute'")
