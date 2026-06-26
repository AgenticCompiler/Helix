from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SECTIONS = ("Summary", "Use When")
VALID_PRIORITIES = ("high", "normal")
_FRONTMATTER_BOUNDARY = "---"
_SECTION_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)


@dataclass
class PatternCard:
    identifier: str
    title: str
    priority: str
    summary: str
    source_path: Path


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    stripped = text.lstrip()
    if not stripped.startswith(_FRONTMATTER_BOUNDARY):
        return {}, text

    lines = stripped.splitlines()
    if len(lines) < 3 or lines[0].strip() != _FRONTMATTER_BOUNDARY:
        return {}, text

    metadata: dict[str, str] = {}
    end_index = None
    for index, raw_line in enumerate(lines[1:], start=1):
        if raw_line.strip() == _FRONTMATTER_BOUNDARY:
            end_index = index
            break
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        metadata[key.strip()] = value.strip()

    if end_index is None:
        return {}, text

    body = "\n".join(lines[end_index + 1:]).lstrip("\n")
    return metadata, body


def _top_level_sections(body: str) -> dict[str, str]:
    matches = list(_SECTION_HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[title] = body[start:end].strip()
    return sections


def _first_nonempty_paragraph(section_text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section_text) if part.strip()]
    if not paragraphs:
        return ""
    return " ".join(line.strip() for line in paragraphs[0].splitlines())


def _fallback_title(metadata: dict[str, str], source_path: Path, body: str) -> str:
    if metadata.get("title"):
        return metadata["title"]
    first_line = body.strip().splitlines()[0].strip() if body.strip() else ""
    if first_line.startswith("# "):
        return first_line[2:].strip()
    return source_path.stem


def _parse_priority(metadata: dict[str, str], source_path: Path) -> str:
    priority = metadata.get("priority", "normal").strip() or "normal"
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"{source_path.name} has invalid priority {priority!r}")
    return priority


def parse_pattern_card(path: Path) -> PatternCard:
    metadata, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    sections = _top_level_sections(body)
    missing = [name for name in REQUIRED_SECTIONS if not sections.get(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required section(s): {names}")

    return PatternCard(
        identifier=metadata.get("id", path.stem),
        title=_fallback_title(metadata, path, body),
        priority=_parse_priority(metadata, path),
        summary=_first_nonempty_paragraph(sections["Summary"]),
        source_path=path,
    )


def _iter_pattern_card_paths(patterns_dir: Path) -> list[Path]:
    ignored_names = {"README.md", "index.md"}
    return [
        path
        for path in sorted(patterns_dir.glob("*.md"))
        if path.name not in ignored_names
    ]


def list_pattern_cards(patterns_dir: Path) -> list[PatternCard]:
    return [parse_pattern_card(path) for path in _iter_pattern_card_paths(patterns_dir)]


def list_high_priority_pattern_cards(patterns_dir: Path) -> list[PatternCard]:
    return [card for card in list_pattern_cards(patterns_dir) if card.priority == "high"]


def build_high_priority_reminder_lines(patterns_dir: Path) -> list[str]:
    cards = list_high_priority_pattern_cards(patterns_dir)
    if not cards:
        return []
    return [f"`{card.identifier}`: {card.summary}" for card in cards]


def build_index_text(patterns_dir: Path) -> str:
    cards = list_pattern_cards(patterns_dir)
    high = [c for c in cards if c.priority == "high"]
    lines = [
        "# Optimization Pattern Index",
        "",
        "Use this file to choose optimization directions before reading detailed pattern references.",
        "",
        "Read this generated index first. Then read only the most relevant detailed pattern files.",
        "",
        "## High Priority Patterns",
        "",
    ]
    if high:
        for card in high:
            lines.append(f"### `{card.identifier}`")
            lines.append("")
            lines.append(f"- Summary: {card.summary}")
            lines.append(f"- Source: [{card.source_path.name}](patterns/{card.source_path.name})")
            lines.append("")
    else:
        lines.append("- None.")
        lines.append("")

    lines.extend(["## All Patterns", ""])
    if cards:
        for card in cards:
            lines.append(f"### `{card.identifier}`")
            lines.append("")
            lines.append(f"- Summary: {card.summary}")
            lines.append(f"- Priority: {card.priority}")
            lines.append(f"- Source: [{card.source_path.name}](patterns/{card.source_path.name})")
            lines.append("")
    else:
        lines.append("- No patterns yet.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the optimize pattern index from pattern cards.")
    parser.add_argument("--patterns-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    rendered = build_index_text(Path(args.patterns_dir))
    output_path = Path(args.output)
    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            print(f"Pattern index is out of date: {output_path}")
            return 1
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
