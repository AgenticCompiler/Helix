from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Any, cast

from env_registry import (
    HELIX_ACCURACY_MODE,
    HELIX_DTYPE_CLOSE_ATOL,
    HELIX_DTYPE_CLOSE_RTOL,
    TORCH_DEVICE_BACKEND_AUTOLOAD,
)
from torch_npu_warnings import suppress_torch_npu_owner_mismatch_warning


# Shared with run_test_execution.py so dynamically loaded tests can import bundled helpers.
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DifferentialTestCase:
    case_id: str
    inputs: tuple[object, ...] | list[object]
    fn: Callable[[], object]


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
    case_id: str | None = None,
) -> list[DifferentialTestCase]:
    test_path = test_file.resolve()
    operator_path = operator_file.resolve()
    bootstrap_torch_npu(test_path.parent, operator_path.parent)
    with temporary_sys_path_entries(test_path.parent, operator_path.parent, SCRIPT_DIR):
        test_module = load_module(test_path, f"differential_test_{test_path.stem}")
        build_operator_api = require_callable(test_module, "build_operator_api", test_path)
        build_cases = require_callable(test_module, "build_differential_test_cases", test_path)
        operator_module = load_module(operator_path, f"differential_operator_{operator_path.stem}")
        raw_cases = build_cases(build_operator_api(operator_module))
    return normalize_differential_cases(raw_cases, case_id=case_id)


def normalize_differential_cases(
    raw_cases: object,
    *,
    case_id: str | None = None,
) -> list[DifferentialTestCase]:
    return [
        DifferentialTestCase(
            case_id=cast(str, record["id"]),
            inputs=cast(Any, record["inputs"]),
            fn=cast(Callable[[], object], record["fn"]),
        )
        for record in select_differential_case_records(
            normalize_differential_case_records(raw_cases), case_id
        )
    ]


def normalize_differential_case_records(raw_cases: object) -> list[dict[str, object]]:
    if isinstance(raw_cases, (str, bytes)) or isinstance(raw_cases, Mapping) or not isinstance(raw_cases, Iterable):
        raise ValueError("Differential test hook 'build_differential_test_cases' must return an iterable of cases")
    records: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for raw_case in cast(Iterable[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError("Differential test cases must be mappings")
        case_map = cast(Mapping[str, object], raw_case)
        record_id = case_map.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("Differential test case is missing required string field 'id'")
        if record_id in seen_case_ids:
            raise ValueError(f"Duplicate differential test case id: {record_id}")
        raw_inputs = case_map.get("inputs")
        if not isinstance(raw_inputs, (list, tuple)):
            raise ValueError(
                f"Differential test case '{record_id}' is missing required list/tuple field 'inputs'"
            )
        case_fn = case_map.get("fn")
        if not callable(case_fn):
            raise ValueError(f"Differential test case '{record_id}' is missing required callable field 'fn'")
        seen_case_ids.add(record_id)
        inputs: tuple[object, ...] | list[object]
        if isinstance(raw_inputs, tuple):
            inputs = tuple(cast(tuple[object, ...], raw_inputs))
        else:
            inputs = list(cast(list[object], raw_inputs))
        records.append({"id": record_id, "inputs": inputs, "fn": case_fn})
    if not records:
        raise ValueError("Differential test hook 'build_differential_test_cases' returned no cases")
    return records


def select_differential_case_records(
    case_records: list[dict[str, object]], case_id: str | None
) -> list[dict[str, object]]:
    if case_id is None:
        return case_records
    for case_record in case_records:
        if case_record["id"] == case_id:
            return [case_record]
    available = ", ".join(cast(str, record["id"]) for record in case_records)
    raise ValueError(f"Unknown differential test case id '{case_id}'. Available case ids: {available}")


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"{module_name}_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def require_callable(
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


def resolve_operator_api(
    operator_module: object, metadata: Mapping[str, str], operator_path: Path
) -> object:
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


def compute_flag_from_metadata(metadata: Mapping[str, str]) -> bool:
    return parse_compute_kind(metadata.get("compute-kind"))


def parse_compute_kind(raw_value: object) -> bool:
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


def bootstrap_torch_npu(*import_paths: Path) -> None:
    suppress_torch_npu_owner_mismatch_warning()
    loaded_torch = sys.modules.get("torch")
    if loaded_torch is not None and hasattr(loaded_torch, "npu"):
        return
    previous = os.environ.get(TORCH_DEVICE_BACKEND_AUTOLOAD)
    os.environ[TORCH_DEVICE_BACKEND_AUTOLOAD] = "0"
    try:
        try:
            importlib.import_module("torch")
        except ImportError:
            with temporary_sys_path_entries(*(path.resolve() for path in import_paths)):
                importlib.import_module("torch")
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            pass
    finally:
        if previous is None:
            os.environ.pop(TORCH_DEVICE_BACKEND_AUTOLOAD, None)
        else:
            os.environ[TORCH_DEVICE_BACKEND_AUTOLOAD] = previous


def run_test_accuracy_env(accuracy_mode: str | None = None) -> dict[str, str]:
    extra_env: dict[str, str] = {}
    if accuracy_mode is not None:
        extra_env[HELIX_ACCURACY_MODE] = accuracy_mode
    for name in (HELIX_ACCURACY_MODE, HELIX_DTYPE_CLOSE_ATOL, HELIX_DTYPE_CLOSE_RTOL):
        if name not in extra_env and (value := os.environ.get(name)) is not None:
            extra_env[name] = value
    return extra_env


def normalize_payload_for_serialization(payload: object) -> object:
    try:
        torch_module = importlib.import_module("torch")
    except ImportError:
        torch_module = None

    def normalize(value: object) -> object:
        tensor_type = None if torch_module is None else getattr(torch_module, "Tensor", None)
        if isinstance(tensor_type, type) and isinstance(value, tensor_type):
            tensor_like = cast(Any, value)
            detached = cast(object, tensor_like.detach()) if hasattr(tensor_like, "detach") else value
            to_cpu = getattr(cast(Any, detached), "cpu", None)
            return to_cpu() if callable(to_cpu) else detached
        if isinstance(value, Mapping):
            return {key: normalize(item) for key, item in cast(Mapping[object, object], value).items()}
        if isinstance(value, tuple):
            tuple_value = cast(tuple[object, ...], value)
            items = tuple(normalize(item) for item in tuple_value)
            if hasattr(tuple_value, "_fields"):
                named_tuple_type = cast(Any, type(tuple_value))
                return cast(object, named_tuple_type(*items))
            return items
        if isinstance(value, list):
            return [normalize(item) for item in cast(list[object], value)]
        return value

    return normalize(payload)


def serialize_payload_object(payload: object) -> str:
    normalized = normalize_payload_for_serialization(payload)
    return base64.b64encode(pickle.dumps(normalized)).decode("ascii")


def deserialize_payload_object(serialized_payload: str) -> object:
    return pickle.loads(base64.b64decode(serialized_payload.encode("ascii")))


@contextmanager
def temporary_sys_path_entries(*paths: Path) -> Iterator[None]:
    added: list[str] = []
    try:
        for path in paths:
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
                added.append(text)
        yield
    finally:
        for text in reversed(added):
            if text in sys.path:
                sys.path.remove(text)
