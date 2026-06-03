#!/usr/bin/env python3
"""Build a kernel-scoped workspace plan from PERF_KNOWLEDGE_BASE.md and repo sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_FILE_SECTION_RE = re.compile(r"^###\s+(\S+)\s*$", re.MULTILINE)
_COMMIT_BLOCK_RE = re.compile(r"^#####\s+([0-9a-f]{7,40})\s+(.+)$", re.MULTILINE)
_TRITON_KERNEL_RE = re.compile(r"@triton\.jit[^\n]*\n(?:@[^\n]+\n)*def\s+(\w+)\s*\(", re.MULTILINE)
_DEF_RE = re.compile(r"^def\s+(\w+)\s*\(", re.MULTILINE)
_LAUNCH_GRID_RE = re.compile(r"\b(\w+)\s*\[")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse PERF_KNOWLEDGE_BASE.md per-kernel lessons and map launch functions "
            "from repo sources into a workspace plan JSON."
        ),
    )
    parser.add_argument("--knowledge", required=True, help="Path to PERF_KNOWLEDGE_BASE.md")
    parser.add_argument("--repo", required=True, help="Repository root for source scanning")
    parser.add_argument("--output", required=True, help="Output workspace-plan.json path")
    parser.add_argument(
        "--base",
        default="",
        help="Optional base revision string copied into the plan for scaffold Git steps.",
    )
    args = parser.parse_args(argv)

    knowledge_path = Path(args.knowledge).expanduser().resolve()
    repo_root = Path(args.repo).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not knowledge_path.is_file():
        print(f"plan_workspaces_from_knowledge: missing knowledge file: {knowledge_path}", file=sys.stderr)
        return 2
    if not repo_root.is_dir():
        print(f"plan_workspaces_from_knowledge: repo is not a directory: {repo_root}", file=sys.stderr)
        return 2

    knowledge_text = knowledge_path.read_text(encoding="utf-8")
    file_sections = _parse_knowledge_file_sections(knowledge_text)
    workspaces: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source_path, section in file_sections.items():
        kernel_lessons = _kernel_lessons_from_section(section)
        source_file = repo_root / source_path
        if not source_file.is_file():
            warnings.append(f"source_path not found in repo, skipping scan: {source_path}")
            continue
        source_text = source_file.read_text(encoding="utf-8")
        launch_map = _map_launch_functions(source_text)
        if not launch_map:
            warnings.append(f"no launch functions discovered in {source_path}")
            continue

        for launch_fn, kernels_launched in launch_map.items():
            primary_kernel = kernels_launched[0]
            lesson_shas = sorted(
                {
                    sha
                    for kernel in kernels_launched
                    for sha in kernel_lessons.get(kernel, [])
                },
            )
            workspaces.append(
                {
                    "workspace": primary_kernel,
                    "kernel_name": primary_kernel,
                    "operator_filename": f"{primary_kernel}.py",
                    "launch_functions": [launch_fn],
                    "kernels_in_operator": kernels_launched,
                    "source_path": source_path,
                    "knowledge_lessons": lesson_shas,
                    "merge_launch_functions": len(kernels_launched) > 1,
                    "notes": (
                        f"Launch {launch_fn} calls {', '.join(kernels_launched)}; "
                        f"workspace named after primary kernel {primary_kernel}."
                    ),
                },
            )

        mentioned_kernels = set(kernel_lessons)
        launched_kernels = {kernel for kernels in launch_map.values() for kernel in kernels}
        orphan_lessons = sorted(mentioned_kernels - launched_kernels)
        if orphan_lessons:
            warnings.append(
                f"{source_path}: knowledge base mentions kernels without a launch mapping: "
                f"{', '.join(orphan_lessons)} — assign manually in workspace-plan.json",
            )

    # Deduplicate by workspace name; later entries override with a warning.
    deduped: dict[str, dict[str, Any]] = {}
    for entry in workspaces:
        name = str(entry["workspace"])
        if name in deduped:
            warnings.append(
                f"duplicate workspace {name!r} from {entry['source_path']!r}; "
                f"keeping first plan entry",
            )
            continue
        deduped[name] = entry

    payload = {
        "schema_version": 1,
        "knowledge_path": knowledge_path.as_posix(),
        "repo": repo_root.as_posix(),
        "base_revision": str(args.base).strip() or None,
        "workspace_count": len(deduped),
        "warnings": warnings,
        "workspaces": list(deduped.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path.as_posix())
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


def _parse_knowledge_file_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(_FILE_SECTION_RE.finditer(text))
    for index, match in enumerate(matches):
        source_path = match.group(1).strip()
        if not source_path.endswith(".py"):
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[source_path] = text[start:end]
    return sections


def _kernel_lessons_from_section(section_text: str) -> dict[str, list[str]]:
    lessons: dict[str, list[str]] = defaultdict(list)
    commit_starts = list(_COMMIT_BLOCK_RE.finditer(section_text))
    for index, match in enumerate(commit_starts):
        sha = match.group(1)
        start = match.end()
        end = commit_starts[index + 1].start() if index + 1 < len(commit_starts) else len(section_text)
        block = section_text[start:end]
        what_changed = _extract_field(block, "What changed")
        symbols = _extract_kernel_symbols(what_changed + "\n" + match.group(2))
        for symbol in symbols:
            if sha not in lessons[symbol]:
                lessons[symbol].append(sha)
    return dict(lessons)


def _extract_field(block: str, field_name: str) -> str:
    marker = f"- {field_name}:"
    for line in block.splitlines():
        if line.strip().startswith(marker):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_kernel_symbols(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    patterns = [
        re.compile(r"`([^`]+)`"),
        re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*_kernel(?:_[A-Za-z0-9_]+)?)\b"),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            symbol = match.group(1).strip()
            if not symbol or symbol in seen:
                continue
            if not _looks_like_kernel_symbol(symbol):
                continue
            seen.add(symbol)
            found.append(symbol)
    return found


def _looks_like_kernel_symbol(symbol: str) -> bool:
    if "_kernel" in symbol:
        return True
    if symbol.endswith("_fwd") or symbol.endswith("_bwd"):
        return True
    return bool(re.search(r"(?:fwd|bwd|kernel)", symbol))


def _map_launch_functions(source_text: str) -> dict[str, list[str]]:
    kernel_names = set(_TRITON_KERNEL_RE.findall(source_text))
    if not kernel_names:
        return {}

    lines = source_text.splitlines()
    current_host: str | None = None
    host_is_triton = False
    launch_to_kernels: dict[str, list[str]] = {}

    for line in lines:
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


if __name__ == "__main__":
    raise SystemExit(main())
