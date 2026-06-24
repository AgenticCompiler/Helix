from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REQUIRED_SECTIONS = ("Summary", "Evidence To Confirm", "Candidate Pattern Directions")
_SECTION_HEADING_RE = re.compile(r"^##\s+(?P<title>.+?)\s*$", flags=re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^(?:-\s+|\d+\.\s+)(?P<item>.+?)\s*$")


@dataclass
class SymptomCard:
    identifier: str
    summary: str
    evidence_to_confirm: list[str]
    candidate_pattern_directions: list[str]
    common_non_matches: list[str]
    source_path: Path


def _top_level_sections(body: str) -> dict[str, str]:
    matches = list(_SECTION_HEADING_RE.finditer(body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[title] = body[start:end].strip()
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


def parse_symptom_card(path: Path) -> SymptomCard:
    body = path.read_text(encoding="utf-8")
    sections = _top_level_sections(body)
    missing = [name for name in REQUIRED_SECTIONS if not sections.get(name)]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required section(s): {names}")

    return SymptomCard(
        identifier=path.stem,
        summary=_first_nonempty_paragraph(sections["Summary"]),
        evidence_to_confirm=_extract_bullets(sections["Evidence To Confirm"]),
        candidate_pattern_directions=_extract_bullets(
            sections["Candidate Pattern Directions"]
        ),
        common_non_matches=_extract_bullets(sections.get("Common Non-Matches", "")),
        source_path=path,
    )


def render_index(cards: list[SymptomCard]) -> str:
    lines = [
        "# Symptom Index",
        "",
        "Use this file after structured profile or IR evidence already exists.",
        "",
        "Read this generated index first. Then read only the one or two most relevant detailed symptom cards before returning to detailed pattern references.",
        "",
        "## Generated Symptom Summaries",
        "",
    ]
    for card in cards:
        lines.append(f"### `{card.identifier}`")
        lines.append("")
        lines.append(f"- Summary: {card.summary}")
        lines.append(f"- Source: [{card.source_path.name}](symptoms/{card.source_path.name})")
        if card.evidence_to_confirm:
            lines.append("- Evidence To Confirm:")
            lines.extend(f"  - {item}" for item in card.evidence_to_confirm)
        if card.candidate_pattern_directions:
            lines.append("- Candidate Pattern Directions:")
            lines.extend(f"  - {item}" for item in card.candidate_pattern_directions)
        if card.common_non_matches:
            lines.append("- Common Non-Matches:")
            lines.extend(f"  - {item}" for item in card.common_non_matches)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_index_text(symptoms_dir: Path) -> str:
    cards = [
        parse_symptom_card(path)
        for path in sorted(symptoms_dir.glob("*.md"))
        if path.name not in {"README.md", "index.md"}
    ]
    return render_index(cards)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the optimize symptom index from symptom cards.")
    parser.add_argument("--symptoms-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    rendered = build_index_text(Path(args.symptoms_dir))
    output_path = Path(args.output)
    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != rendered:
            print(f"Symptom index is out of date: {output_path}")
            return 1
        return 0

    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
