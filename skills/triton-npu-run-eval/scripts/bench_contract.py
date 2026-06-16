from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelResolution:
    kernel_names: list[str]
    kernel_source: str


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in bench_file.read_text(encoding="utf-8").splitlines():
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
        key_stripped = key.strip()
        if key_stripped == "bench-mode":
            continue
        metadata[key_stripped] = value.strip()
    return metadata


def resolve_bench_kernel_names(
    bench_file: Path,
    operator_file: Path | None = None,
) -> list[str]:
    return resolve_bench_kernel_resolution(bench_file, operator_file).kernel_names


def resolve_bench_kernel_resolution(
    bench_file: Path,
    operator_file: Path | None = None,
) -> KernelResolution:
    metadata = parse_bench_metadata(bench_file)
    metadata_kernel_names = _parse_kernel_names(metadata, bench_file, allow_empty=True)
    operator_kernel_names = (
        _discover_operator_triton_kernels(operator_file) if operator_file is not None else []
    )
    kernel_names = _stable_kernel_union(metadata_kernel_names, operator_kernel_names)
    if not kernel_names:
        raise ValueError(
            f"Benchmark metadata and operator file did not resolve any Triton kernels: {bench_file}"
        )
    return KernelResolution(
        kernel_names=kernel_names,
        kernel_source=_describe_kernel_source(metadata_kernel_names, operator_kernel_names),
    )


def _parse_kernel_names(
    metadata: dict[str, str],
    bench_file: Path,
    *,
    allow_empty: bool = False,
) -> list[str]:
    kernels_value = metadata.get("kernels")
    if kernels_value is not None:
        kernel_names = [part.strip() for part in kernels_value.split(",") if part.strip()]
    else:
        kernel_name = (metadata.get("kernel") or "").strip()
        kernel_names = [kernel_name] if kernel_name else []
    if not kernel_names and not allow_empty:
        raise ValueError(
            f"Benchmark metadata is missing required 'kernels' entry: {bench_file}"
        )
    return kernel_names


def _discover_operator_triton_kernels(operator_file: Path) -> list[str]:
    try:
        tree = ast.parse(operator_file.read_text(encoding="utf-8"), filename=str(operator_file))
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse operator file for Triton kernels: {operator_file}") from exc
    kernels: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _is_triton_jit_decorator(decorator) for decorator in node.decorator_list
        ):
            kernels.append(node.name)
    return kernels


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        return _is_triton_jit_decorator(node.func)
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr == "jit"
    return isinstance(node, ast.Name) and node.id == "jit"


def _stable_kernel_union(primary: list[str], secondary: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for kernel_name in [*primary, *secondary]:
        if kernel_name in seen:
            continue
        seen.add(kernel_name)
        merged.append(kernel_name)
    return merged


def _describe_kernel_source(metadata_kernels: list[str], operator_kernels: list[str]) -> str:
    if metadata_kernels and operator_kernels:
        return "metadata+operator"
    if metadata_kernels:
        return "metadata"
    return "operator"
