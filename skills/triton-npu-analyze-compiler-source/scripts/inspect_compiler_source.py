#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEARCH_AREAS: tuple[tuple[str, str, str], ...] = (
    ("docs", "docs/source/en/developer_guide/passes", "*.md"),
    ("docs", "docs/source/zh_cn/developer_guide/passes", "*.md"),
    ("docs", "docs/source/en/developer_guide/features", "*.md"),
    ("docs", "docs/source/zh_cn/developer_guide/features", "*.md"),
    ("lib", "bishengir/lib/Conversion", "*"),
    ("lib", "bishengir/lib/Dialect", "*"),
    ("lib", "bishengir/lib/Transforms", "*"),
    ("include", "bishengir/include/bishengir", "*"),
)

HINT_CHOICES: tuple[str, ...] = (
    "pass",
    "feature",
    "dialect",
    "conversion",
    "layout",
    "memory",
    "pipeline",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect compiler source for likely navigation targets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    locate = subparsers.add_parser("locate")
    locate.add_argument("--source-root", required=True)
    locate.add_argument("--term", action="append", required=True)
    locate.add_argument("--hint", choices=HINT_CHOICES)
    locate.add_argument("--limit", type=int, default=10)
    locate.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def locate_payload(
    source_root: str | Path,
    *,
    terms: list[str],
    hint: str | None = None,
    limit: int = 10,
) -> dict[str, list[dict[str, object]]]:
    root = Path(source_root).expanduser().resolve()
    lowered_terms = [term.lower() for term in terms if term.strip()]
    grouped: dict[str, list[dict[str, object]]] = {"docs": [], "lib": [], "include": []}

    for area, relative_root, pattern in SEARCH_AREAS:
        candidate_root = root / relative_root
        if not candidate_root.exists():
            continue
        for path in sorted(candidate_root.rglob(pattern)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            score, matched_terms, why = _score_candidate(
                path=path,
                text=text,
                root=root,
                area=area,
                hint=hint,
                lowered_terms=lowered_terms,
            )
            if score <= 0:
                continue
            grouped[area].append(
                {
                    "area": area,
                    "path": str(path),
                    "score": score,
                    "matched_terms": matched_terms,
                    "why": why,
                }
            )

    for area, items in grouped.items():
        grouped[area] = sorted(
            items,
            key=lambda item: (-int(item["score"]), str(item["path"])),
        )[:limit]
    return grouped


def locate_text(
    source_root: str | Path,
    *,
    terms: list[str],
    hint: str | None = None,
    limit: int = 10,
) -> str:
    payload = locate_payload(source_root, terms=terms, hint=hint, limit=limit)
    lines = ["Compiler source candidates:"]
    for area in ("docs", "lib", "include"):
        lines.append("")
        lines.append(f"{area}:")
        if not payload[area]:
            lines.append("  (no matches)")
            continue
        for item in payload[area]:
            matched_terms = item["matched_terms"]
            assert isinstance(matched_terms, list)
            lines.append(
                "  "
                f"{item['path']}  score={item['score']}  "
                f"matched_terms={','.join(matched_terms)}  why={item['why']}"
            )
    return "\n".join(lines) + "\n"


def _score_candidate(
    *,
    path: Path,
    text: str,
    root: Path,
    area: str,
    hint: str | None,
    lowered_terms: list[str],
) -> tuple[int, list[str], str]:
    relative_path = str(path.relative_to(root)).lower()
    lowered_text = text.lower()
    matched_terms = [
        term for term in lowered_terms if term in relative_path or term in lowered_text
    ]
    if not matched_terms:
        return 0, [], ""

    path_hits = sum(relative_path.count(term) for term in matched_terms)
    text_hits = sum(lowered_text.count(term) for term in matched_terms)
    score = path_hits * 6 + min(text_hits, 5) * 2

    if area == "docs":
        score += 2
    elif area == "lib":
        score += 1

    if hint == "pass":
        if area == "docs":
            score += 5
        if "passes" in relative_path:
            score += 3
        if area == "include" and "passes" in relative_path:
            score += 2
    elif hint == "feature":
        if area == "docs" and "features" in relative_path:
            score += 5
    elif hint == "conversion":
        if area == "lib" and "conversion" in relative_path:
            score += 4
        if area == "include" and "conversion" in relative_path:
            score += 2
    elif hint == "dialect":
        if area == "lib" and "dialect" in relative_path:
            score += 4
    elif hint in {"layout", "memory", "pipeline"}:
        if area == "lib":
            score += 3
        if area == "docs":
            score += 2

    why = _build_why(area=area, relative_path=relative_path, hint=hint)
    return score, matched_terms, why


def _build_why(*, area: str, relative_path: str, hint: str | None) -> str:
    if area == "docs":
        if "passes" in relative_path:
            return "pass docs match"
        if "features" in relative_path:
            return "feature docs match"
        return "docs match"
    if area == "lib":
        if "conversion" in relative_path:
            return "conversion implementation match"
        if "dialect" in relative_path:
            return "dialect implementation match"
        if "transforms" in relative_path:
            return "transform implementation match"
        return "implementation match"
    if "passes" in relative_path:
        return "generated pass declaration match"
    if hint:
        return f"{hint} declaration match"
    return "declaration match"


def main() -> int:
    args = build_parser().parse_args()
    if args.command != "locate":
        raise ValueError(f"Unsupported command: {args.command}")
    if args.format == "json":
        print(
            json.dumps(
                locate_payload(
                    args.source_root,
                    terms=args.term,
                    hint=args.hint,
                    limit=args.limit,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            locate_text(
                args.source_root,
                terms=args.term,
                hint=args.hint,
                limit=args.limit,
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
