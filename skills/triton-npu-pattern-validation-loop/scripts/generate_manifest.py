#!/usr/bin/env python3
"""Optional heuristic manifest builder when synthesis matches strict table format.

Primary workflow: the orchestrating agent reads PERF_PATTERN_SYNTHESIS.md and authors
manifest.json manually. See references/workspace-scaffold-contract.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


class ManifestError(RuntimeError):
    pass

_ITEM_ROW = re.compile(
    r"^\|\s*(G\d+-I\d+)\s*\|\s*[^|]+\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*$"
)
_BACKTICK_ID = re.compile(r"`([a-z0-9-]+)`")
_SUPPORTING_FILES = re.compile(r"^- Supporting files:\s*(.+)$", re.MULTILINE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build validation manifest JSON from synthesis report.")
    parser.add_argument("--repo", required=True, help="Git repository path.")
    parser.add_argument("--synthesis", default="PERF_PATTERN_SYNTHESIS.md", help="Synthesis report path.")
    parser.add_argument("--base", default="origin/main", help="Base revision for scaffold.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    args = parser.parse_args(argv)
    try:
        repo = Path(args.repo).expanduser().resolve()
        synthesis_path = Path(args.synthesis).expanduser()
        if not synthesis_path.is_absolute():
            synthesis_path = (repo / synthesis_path).resolve()
        if not synthesis_path.is_file():
            raise ManifestError(f"synthesis report not found: {synthesis_path}")
        manifest = build_manifest(
            repo=repo,
            synthesis_text=synthesis_path.read_text(encoding="utf-8"),
            base_revision=str(args.base),
            synthesis_path=synthesis_path,
        )
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = (repo / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(output_path.as_posix())
        return 0
    except ManifestError as exc:
        print(f"generate_manifest: {exc}", file=sys.stderr)
        return 2


def build_manifest(
    *,
    repo: Path,
    synthesis_text: str,
    base_revision: str,
    synthesis_path: Path,
) -> dict[str, object]:
    item_patterns = parse_item_patterns(synthesis_text)
    file_to_patterns: dict[str, set[str]] = {}
    for item_id, related, recommendation in item_patterns:
        if recommendation in {"local-only", "reject", "no-change"}:
            continue
        patterns = extract_pattern_ids(related)
        if not patterns:
            continue
        source_files = find_source_files_for_item(synthesis_text, item_id)
        for source_file in source_files:
            file_to_patterns.setdefault(source_file, set()).update(patterns)

    if not file_to_patterns:
        file_to_patterns = parse_supporting_files_fallback(synthesis_text)

    operators: list[dict[str, object]] = []
    for source_path in sorted(file_to_patterns):
        stem = Path(source_path).stem
        operators.append(
            {
                "name": stem,
                "source_path": source_path,
                "operator_filename": Path(source_path).name,
                "test_paths": [],
                "expected_patterns": sorted(file_to_patterns[source_path]),
                "notes": f"auto-generated from {synthesis_path.name}",
            }
        )
    if not operators:
        raise ManifestError("no operators inferred from synthesis; edit manifest manually")

    try:
        synthesis_ref = synthesis_path.relative_to(repo).as_posix()
    except ValueError:
        synthesis_ref = synthesis_path.as_posix()

    return {
        "repo": repo.as_posix(),
        "base_revision": base_revision,
        "synthesis_report": synthesis_ref,
        "operators": operators,
    }


def parse_item_patterns(text: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        match = _ITEM_ROW.match(line.strip())
        if not match:
            continue
        item_id, _group, related, recommendation, _rationale = match.groups()
        rows.append((item_id.strip(), related.strip(), recommendation.strip()))
    return rows


def find_source_files_for_item(text: str, item_id: str) -> list[str]:
    section_pattern = re.compile(
        rf"^#####\s+{re.escape(item_id)}\b[^\n]*\n(?:.*?\n)*?(?=^#####\s|\Z)",
        re.MULTILINE,
    )
    match = section_pattern.search(text)
    if not match:
        return []
    block = match.group(0)
    source_match = re.search(r"^- Source file:\s*(.+)$", block, re.MULTILINE)
    if not source_match:
        return []
    raw = source_match.group(1).strip()
    paths: list[str] = []
    for part in re.split(r",\s*", raw):
        part = part.strip()
        if part.endswith(".py"):
            paths.append(part)
    return paths


def extract_pattern_ids(related: str) -> set[str]:
    ids = set(_BACKTICK_ID.findall(related))
    cleaned = {item for item in ids if item not in {"novel", "matches", "extends"}}
    if "none (novel)" in related.lower() or related.strip().lower().startswith("none"):
        return set()
    return cleaned


def parse_supporting_files_fallback(text: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for match in _SUPPORTING_FILES.finditer(text):
        files = [part.strip() for part in match.group(1).split(",") if part.strip().endswith(".py")]
        for file_name in files:
            mapping.setdefault(file_name, set())
    if not mapping:
        return mapping
    default_patterns = set(_BACKTICK_ID.findall(text))
    for key in mapping:
        mapping[key] = set(default_patterns)
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())
