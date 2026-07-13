from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from helix.paths import skills_root


def _resolve_skill_source_dir(skill_name: str) -> Path:
    root = skills_root()
    flat_path = root / skill_name
    if flat_path.is_dir():
        return flat_path
    for group_dir in root.iterdir():
        if not group_dir.is_dir():
            continue
        grouped_path = group_dir / skill_name
        if grouped_path.is_dir():
            return grouped_path
    raise FileNotFoundError(f"Skill not found: {skill_name}")


BASELINE_CONTRACT_PATH = (
    _resolve_skill_source_dir("ascend-npu-optimize-state")
    / "references"
    / "baseline-contract.json"
)
ROUND_CONTRACT_PATH = (
    _resolve_skill_source_dir("ascend-npu-optimize-state")
    / "references"
    / "round-contract.json"
)
ARTIFACTS_PATH = (
    _resolve_skill_source_dir("triton-npu-optimize") / "references" / "artifacts.md"
)

BASELINE_BEGIN = "<!-- BEGIN GENERATED BASELINE STATE CONTRACT -->"
BASELINE_END = "<!-- END GENERATED BASELINE STATE CONTRACT -->"
ROUND_BEGIN = "<!-- BEGIN GENERATED ROUND STATE CONTRACT -->"
ROUND_END = "<!-- END GENERATED ROUND STATE CONTRACT -->"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return cast(dict[str, object], payload)


def _render_json_block(payload: dict[str, object]) -> str:
    return "```json\n" + json.dumps(payload, indent=2) + "\n```"


def _render_baseline_section(contract: dict[str, object]) -> str:
    field_map_object = contract["baseline_state_fields"]
    if not isinstance(field_map_object, dict):
        raise ValueError("baseline contract must define baseline_state_fields as an object")
    field_map = cast(dict[str, object], field_map_object)
    lines = [
        BASELINE_BEGIN,
        "`baseline/state.json` required fields:",
        "",
        _render_json_block(field_map),
        "",
        "Path-bearing fields in `baseline/state.json` must be written relative to the directory that contains `baseline/state.json`.",
        "",
        "Set `baseline_established` to `true` only after correctness passed, benchmark passed, and the canonical baseline artifacts are in place.",
        BASELINE_END,
    ]
    return "\n".join(lines)


def _render_round_section(contract: dict[str, object]) -> str:
    required_map_object = contract["round_state_required_fields"]
    optional_map_object = contract["round_state_optional_fields"]
    if not isinstance(required_map_object, dict):
        raise ValueError("round contract must define round_state_required_fields as an object")
    if not isinstance(optional_map_object, dict):
        raise ValueError("round contract must define round_state_optional_fields as an object")
    required_map = cast(dict[str, object], required_map_object)
    optional_map = cast(dict[str, object], optional_map_object)
    lines = [
        ROUND_BEGIN,
        "`round-state.json` required fields:",
        "",
        _render_json_block(required_map),
        "",
        "`round-state.json` optional fields when present:",
        "",
        _render_json_block(optional_map),
        "",
        "Path-bearing fields in `round-state.json` must be written relative to the directory that contains `round-state.json`.",
        ROUND_END,
    ]
    return "\n".join(lines)


def _replace_section(content: str, *, begin: str, end: str, replacement: str) -> str:
    start = content.find(begin)
    if start == -1:
        raise ValueError(f"missing marker: {begin}")
    finish = content.find(end, start)
    if finish == -1:
        raise ValueError(f"missing marker: {end}")
    finish += len(end)
    return content[:start] + replacement + content[finish:]


def main() -> int:
    baseline_contract = _load_json(BASELINE_CONTRACT_PATH)
    round_contract = _load_json(ROUND_CONTRACT_PATH)
    artifacts = ARTIFACTS_PATH.read_text(encoding="utf-8")
    artifacts = _replace_section(
        artifacts,
        begin=BASELINE_BEGIN,
        end=BASELINE_END,
        replacement=_render_baseline_section(baseline_contract),
    )
    artifacts = _replace_section(
        artifacts,
        begin=ROUND_BEGIN,
        end=ROUND_END,
        replacement=_render_round_section(round_contract),
    )
    ARTIFACTS_PATH.write_text(artifacts, encoding="utf-8")
    print(ARTIFACTS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
