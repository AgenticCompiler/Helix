#!/usr/bin/env python3
"""Verify pattern-validation workspaces before optimize-batch."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_TRITON_KERNEL_RE = re.compile(
    r"@triton\.jit[^\n]*\n(?:@[^\n]+\n)*def\s+(\w+)\s*\(",
    re.MULTILINE,
)
_DEF_RE = re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)
_LAUNCH_GRID_RE = re.compile(r"\b(\w+)\s*\[")

from batch_layout import list_active_validation_workspaces
from workspace_deps import verify_dependency_issues

_DEFAULT_DEPENDENCY_DIR = "deps"
_ROOT_ALLOWED_NAMES = frozenset({"__init__.py", "conftest.py"})
_ROOT_ALLOWED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that validation workspaces follow workspace-scaffold-contract Step 2b "
            "before running optimize-batch."
        ),
    )
    parser.add_argument("--batch-root", required=True, help="Batch root directory.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report.",
    )
    args = parser.parse_args(argv)

    batch_root = Path(args.batch_root).expanduser().resolve()
    workspaces = list_active_validation_workspaces(batch_root)
    if not workspaces:
        print(f"verify_batch_scaffold: no active workspaces under {batch_root}", file=sys.stderr)
        return 1

    metas = [_load_meta(path) for path in workspaces]
    shared_sources = _shared_source_paths(metas)
    reports = [
        verify_workspace(workspace, meta, shared_sources=shared_sources)
        for workspace, meta in zip(workspaces, metas, strict=True)
    ]
    failed = [report for report in reports if report["issues"]]

    payload = {
        "batch_root": batch_root.as_posix(),
        "workspace_count": len(reports),
        "failed_count": len(failed),
        "reports": reports,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(_render_report(report))
        print(
            f"Summary: {len(reports) - len(failed)} ok, {len(failed)} failed scaffold checks",
        )

    return 1 if failed else 0


def verify_workspace(
    workspace: Path,
    meta: dict[str, Any],
    *,
    shared_sources: dict[str, list[str]],
) -> dict[str, Any]:
    issues: list[str] = []
    kernel_name = str(meta.get("kernel_name", "")).strip()
    operator_name = str(meta.get("operator_filename", "")).strip()
    if kernel_name and workspace.name != kernel_name:
        issues.append(
            f"workspace directory {workspace.name!r} must match validation-meta kernel_name "
            f"{kernel_name!r}"
        )
    if kernel_name and operator_name and operator_name != f"{kernel_name}.py":
        issues.append(
            f"operator_filename {operator_name!r} must be {kernel_name}.py when kernel_name is set"
        )
    if not operator_name:
        issues.append("validation-meta.json missing operator_filename")
        operator_path = None
        operator_text = ""
    else:
        operator_path = workspace / operator_name
        if not operator_path.is_file():
            issues.append(f"operator file not found: {operator_name}")
            operator_text = ""
        else:
            operator_text = operator_path.read_text(encoding="utf-8")

    source_path = str(meta.get("source_path", "")).strip()
    validation_target = str(meta.get("validation_target", "")).strip()
    split_from = str(meta.get("split_from", "")).strip()
    shared_source = source_path and len(shared_sources.get(source_path, [])) > 1

    if shared_source or split_from:
        if not validation_target:
            issues.append(
                "validation_target is required when split_from is set or multiple workspaces "
                f"share source_path {source_path!r} (Step 2b manual split)"
            )
        if shared_source and not split_from:
            issues.append(
                f"split_from is required when multiple workspaces share source_path {source_path!r}"
            )
        if shared_source and not meta.get("included_symbols"):
            issues.append(
                "included_symbols is required when multiple workspaces share the same source_path"
            )

    if validation_target and operator_text and validation_target not in operator_text:
        issues.append(
            f"validation_target {validation_target!r} not found in operator file; "
            "operator may be an un-split whole-file copy"
        )

    for symbol in _string_list(meta.get("included_symbols")):
        if operator_text and symbol not in operator_text:
            issues.append(f"included_symbols entry {symbol!r} missing from operator file")

    for symbol in _string_list(meta.get("excluded_targets")):
        if operator_text and symbol in operator_text:
            issues.append(
                f"excluded_targets entry {symbol!r} still present in operator file; "
                "Step 2b extract likely incomplete"
            )

    issues.extend(
        _operator_extract_issues(
            operator_text,
            kernels_in_operator=_string_list(meta.get("kernels_in_operator")),
            launch_functions=_string_list(meta.get("launch_functions")),
        ),
    )

    extra_root_py = _extra_root_py_files(workspace, operator_name)
    if extra_root_py:
        issues.append(
            "optimize-batch allows only one operator .py at workspace root; move helper "
            f"modules under {_DEFAULT_DEPENDENCY_DIR}/: {', '.join(extra_root_py)}"
        )

    dependency_dir = str(meta.get("dependency_dir", _DEFAULT_DEPENDENCY_DIR)).strip() or _DEFAULT_DEPENDENCY_DIR
    for entry in _string_list(meta.get("copied_dependencies")):
        normalized = entry.replace("\\", "/").lstrip("./")
        if "/" not in normalized:
            issues.append(
                f"copied_dependencies entry {entry!r} must live under {dependency_dir}/ "
                f"(for example {dependency_dir}/{Path(normalized).name})"
            )
            continue
        if not normalized.startswith(f"{dependency_dir}/"):
            issues.append(
                f"copied_dependencies entry {entry!r} must be under workspace {dependency_dir}/"
            )
        elif not (workspace / normalized).is_file():
            issues.append(f"copied_dependencies entry missing on disk: {entry}")

    issues.extend(
        verify_dependency_issues(
            workspace,
            meta,
            operator_text=operator_text,
            operator_filename=operator_name,
        ),
    )

    return {
        "workspace": workspace.name,
        "source_path": source_path or None,
        "validation_target": validation_target or None,
        "split_from": split_from or None,
        "shared_source_path": shared_source,
        "kernel_name": kernel_name or None,
        "dependency_dir": dependency_dir,
        "extra_root_py": extra_root_py,
        "issues": issues,
        "passed": not issues,
    }


def _operator_extract_issues(
    operator_text: str,
    *,
    kernels_in_operator: list[str],
    launch_functions: list[str],
) -> list[str]:
    if not operator_text:
        return []

    issues: list[str] = []
    found_kernels = _triton_kernel_names(operator_text)

    if kernels_in_operator:
        allowed = set(kernels_in_operator)
        extra_kernels = [name for name in found_kernels if name not in allowed]
        missing_kernels = [name for name in kernels_in_operator if name not in found_kernels]
        if extra_kernels:
            issues.append(
                "operator still defines Triton kernels outside kernels_in_operator: "
                f"{', '.join(extra_kernels)}; trim the extract (Step 2b), do not copy the whole "
                "source_path file"
            )
        if missing_kernels:
            issues.append(
                "kernels_in_operator not present in operator file: "
                f"{', '.join(missing_kernels)}"
            )
    elif launch_functions and len(found_kernels) > 1:
        issues.append(
            f"operator defines {len(found_kernels)} @triton.jit kernels "
            f"({', '.join(found_kernels)}) but kernels_in_operator is unset; copy "
            "kernels_in_operator from workspace-plan.json and trim unrelated kernels"
        )

    if launch_functions:
        launch_map = _launch_functions_in_source(operator_text)
        for launch_name in launch_functions:
            if launch_name not in operator_text:
                issues.append(
                    f"launch_functions entry {launch_name!r} not found in operator file",
                )
        extra_launches = [name for name in launch_map if name not in launch_functions]
        if extra_launches:
            issues.append(
                "operator still defines host launch functions outside launch_functions: "
                f"{', '.join(extra_launches)}; trim the extract (Step 2b)"
            )

    return issues


def _triton_kernel_names(source_text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in _TRITON_KERNEL_RE.findall(source_text):
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _launch_functions_in_source(source_text: str) -> dict[str, list[str]]:
    kernel_names = set(_TRITON_KERNEL_RE.findall(source_text))
    if not kernel_names:
        return {}

    current_host: str | None = None
    host_is_triton = False
    launch_to_kernels: dict[str, list[str]] = {}

    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@triton.jit"):
            host_is_triton = True
            continue
        def_match = _DEF_RE.match(line)
        if def_match:
            current_host = def_match.group(1)
            host_is_triton = False
            continue
        if current_host is None or host_is_triton:
            continue
        for launch_match in _LAUNCH_GRID_RE.finditer(line):
            callee = launch_match.group(1)
            if callee not in kernel_names:
                continue
            kernels = launch_to_kernels.setdefault(current_host, [])
            if callee not in kernels:
                kernels.append(callee)

    return launch_to_kernels


def _extra_root_py_files(workspace: Path, operator_filename: str) -> list[str]:
    if not operator_filename:
        return []
    extras: list[str] = []
    for path in sorted(workspace.iterdir()):
        if not path.is_file() or path.suffix != ".py":
            continue
        if path.name == operator_filename:
            continue
        if path.name in _ROOT_ALLOWED_NAMES:
            continue
        if path.name.startswith(_ROOT_ALLOWED_PREFIXES):
            continue
        extras.append(path.name)
    return extras


def _load_meta(workspace: Path) -> dict[str, Any]:
    meta_path = workspace / "validation-meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"verify_batch_scaffold: missing validation-meta.json: {workspace}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"verify_batch_scaffold: invalid validation-meta.json: {meta_path}")
    return payload


def _shared_source_paths(metas: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for meta in metas:
        source_path = str(meta.get("source_path", "")).strip()
        workspace = str(meta.get("workspace", "")).strip()
        if not source_path or not workspace:
            continue
        grouped.setdefault(source_path, []).append(workspace)
    return grouped


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _render_report(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [f"[{status}] {report['workspace']}"]
    if report.get("source_path"):
        lines.append(f"  source_path: {report['source_path']}")
    if report.get("validation_target"):
        lines.append(f"  validation_target: {report['validation_target']}")
    if report.get("shared_source_path"):
        lines.append("  shared_source_path: yes")
    for issue in report["issues"]:
        lines.append(f"  issue: {issue}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
