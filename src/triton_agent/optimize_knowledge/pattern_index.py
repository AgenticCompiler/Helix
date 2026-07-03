from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PatternIndexStyle = Literal["default", "extended", "tilelang"]

REQUIRED_SECTIONS = ("Summary", "Use When")
VALID_PRIORITIES = ("high", "normal")
_FRONTMATTER_BOUNDARY = "---"
_SECTION_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_SUBSECTION_HEADING_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^(?:-\s+|\d+\.\s+)(?P<item>.+?)\s*$")


@dataclass
class PatternCard:
    identifier: str
    title: str
    priority: str
    summary: str
    use_when: list[str]
    avoid_when: list[str]
    signals_code: list[str]
    signals_profile: list[str]
    signals_ir: list[str]
    related_patterns: list[str]
    verify_after_applying: list[str]
    source_path: Path


def infer_pattern_index_style(knowledge_dir: Path) -> PatternIndexStyle:
    name = knowledge_dir.name
    if name.startswith("tilelang-"):
        return "tilelang"
    if name.endswith("-v2") or name.endswith("-v3"):
        return "extended"
    return "default"


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
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

    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
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


def _subsections(section_text: str) -> dict[str, str]:
    matches = list(_SUBSECTION_HEADING_RE.finditer(section_text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section_text)
        sections[title] = section_text[start:end].strip()
    return sections


def _extract_bullets(section_text: str) -> list[str]:
    bullets: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        match = _LIST_ITEM_RE.match(line)
        if match:
            bullets.append(match.group("item").strip())
    return bullets


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
    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    sections = _top_level_sections(body)
    missing = [name for name in REQUIRED_SECTIONS if not sections.get(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required section(s): {names}")

    signals = _subsections(sections.get("Signals", ""))
    return PatternCard(
        identifier=metadata.get("id", path.stem),
        title=_fallback_title(metadata, path, body),
        priority=_parse_priority(metadata, path),
        summary=_first_nonempty_paragraph(sections["Summary"]),
        use_when=_extract_bullets(sections["Use When"]),
        avoid_when=_extract_bullets(sections.get("Avoid When", "")),
        signals_code=_extract_bullets(signals.get("Code", "")),
        signals_profile=_extract_bullets(signals.get("Profile", "")),
        signals_ir=_extract_bullets(signals.get("IR", "")),
        related_patterns=_extract_bullets(sections.get("Related Patterns", "")),
        verify_after_applying=_extract_bullets(
            sections.get("What To Verify After Applying", "")
        ),
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
    return [
        f"`{card.identifier}`: {card.summary}"
        for card in list_high_priority_pattern_cards(patterns_dir)
    ]


def _render_bullets(items: list[str]) -> list[str]:
    return [f"  - {item}" for item in items]


def _render_detailed_index(cards: list[PatternCard], *, include_extended_fields: bool) -> str:
    high_priority_cards = [card for card in cards if card.priority == "high"]
    lines = [
        "# Optimization Pattern Index",
        "",
        "Use this file to choose optimization directions before reading any detailed pattern reference.",
        "",
        "Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.",
        "",
        "Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.",
        "",
        "## High Priority Patterns",
        "",
    ]
    if high_priority_cards:
        for card in high_priority_cards:
            lines.append(f"### `{card.identifier}`")
            lines.append("")
            lines.append(f"- Summary: {card.summary}")
            lines.append(f"- Source: [{card.source_path.name}](patterns/{card.source_path.name})")
            lines.append("")
    else:
        lines.append("- None.")
        lines.append("")

    lines.extend(["## Generated Pattern Summaries", ""])
    for card in cards:
        lines.append(f"### `{card.identifier}`")
        lines.append("")
        lines.append(f"- Summary: {card.summary}")
        lines.append(f"- Source: [{card.source_path.name}](patterns/{card.source_path.name})")
        if card.use_when:
            lines.append("- Use When:")
            lines.extend(_render_bullets(card.use_when))
        if include_extended_fields:
            if card.avoid_when:
                lines.append("- Avoid When:")
                lines.extend(_render_bullets(card.avoid_when))
            if card.signals_code:
                lines.append("- Signals / Code:")
                lines.extend(_render_bullets(card.signals_code))
            if card.signals_profile:
                lines.append("- Signals / Profile:")
                lines.extend(_render_bullets(card.signals_profile))
            if card.signals_ir:
                lines.append("- Signals / IR:")
                lines.extend(_render_bullets(card.signals_ir))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_tilelang_index(cards: list[PatternCard]) -> str:
    high_priority_cards = [card for card in cards if card.priority == "high"]
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
    if high_priority_cards:
        for card in high_priority_cards:
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


def build_index_text(
    patterns_dir: Path,
    *,
    style: PatternIndexStyle = "default",
) -> str:
    cards = list_pattern_cards(patterns_dir)
    if style == "tilelang":
        return _render_tilelang_index(cards)
    return _render_detailed_index(cards, include_extended_fields=style == "extended")


def write_index(
    *,
    patterns_dir: Path,
    output: Path,
    style: PatternIndexStyle = "default",
    check: bool = False,
) -> int:
    rendered = build_index_text(patterns_dir, style=style)
    if check:
        current = output.read_text(encoding="utf-8")
        if current != rendered:
            print(f"Pattern index is out of date: {output}")
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the optimize pattern index from pattern cards.")
    parser.add_argument("--patterns-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--style",
        choices=("default", "extended", "tilelang"),
        default="default",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    return write_index(
        patterns_dir=Path(args.patterns_dir),
        output=Path(args.output),
        style=args.style,
        check=bool(args.check),
    )


if __name__ == "__main__":
    raise SystemExit(main())

