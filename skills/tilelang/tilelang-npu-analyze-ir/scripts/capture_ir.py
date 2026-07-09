#!/usr/bin/env python3
"""Capture AscendC source code from a TileLang compiled operator."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _cache_cleanup_guidance() -> str:
    return (
        'Clear workspace "__pycache__" directories and stale TileLang memoize files '
        '(for example `.pkl_memoize_py3`) before retrying the capture.'
    )


def _no_kernels_found_message(operator_file: Path) -> str:
    return (
        f"No compiled TileLang kernels found in {operator_file}. "
        "Add a module-level call that triggers @tilelang.jit compilation during import, "
        "for example `compiled_kernel = kernel_func(...)`, and expose the compiled kernel "
        "through a name that does not start with `_`."
    )


def _compilation_failure_message(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {exc}. {_cache_cleanup_guidance()}"


def _resolve_existing_path(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")
    return path


def _load_operator_module(operator_file: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"_capture_op_{operator_file.stem}", operator_file
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load operator module: {operator_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _is_tilelang_kernel(obj: Any) -> bool:
    return callable(getattr(obj, "get_kernel_source", None))


def _find_kernels(module: Any) -> dict[str, Any]:
    kernels: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if _is_tilelang_kernel(obj):
            kernels[name] = obj
    return kernels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print AscendC source from a TileLang compiled operator.",
        allow_abbrev=False,
    )
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--kernel", help="Specific kernel name (default: all)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operator_file = _resolve_existing_path(args.operator_file, "Operator file")

    try:
        module = _load_operator_module(operator_file)
    except Exception as exc:
        print(_compilation_failure_message("Failed to load operator", exc), file=sys.stderr)
        return 1

    kernels = _find_kernels(module)
    if not kernels:
        print(_no_kernels_found_message(operator_file), file=sys.stderr)
        return 1

    if args.kernel:
        if args.kernel not in kernels:
            print(
                f"Kernel '{args.kernel}' not found. Available: {', '.join(sorted(kernels))}",
                file=sys.stderr,
            )
            return 1
        kernels = {args.kernel: kernels[args.kernel]}

    for name, kernel in sorted(kernels.items()):
        if len(kernels) > 1:
            print(f"// === {name} ===")
        try:
            print(kernel.get_kernel_source())
        except Exception as exc:
            print(_compilation_failure_message(f"Compilation failed for kernel '{name}'", exc), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
