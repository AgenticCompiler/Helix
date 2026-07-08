"""Structured issue detection: open issue_type strings, validation/coercion.

issue_type is an **open string** (not a closed enum) — the agent writes any
descriptive label (typically a pattern name or short description) in
stage-verdict.json. Routing is done by the agent (issues are placed under the
stage the agent judges them to belong to), not by a CLI routing table. The
scanner also emits issue_type strings for display in the injection.

Adding a new pattern only requires editing ``stages.json`` (add the pattern to
a stage's ``patterns`` list) — no code changes, no enum updates, no routing
table updates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from triton_agent.optimize.stages import Stage, StageGraph, default_stage_graph


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Issue:
    """A single detected optimization issue.

    ``issue_type`` is an open string (e.g. a pattern name like
    ``"flat-index-decode-tiling"`` or a description like ``"missing_autotune"``).
    The agent places the issue under the stage it judges correct in
    stage-verdict.json; no CLI routing is needed.
    """

    issue_type: str
    severity: int = 3  # 1 (low) .. 5 (high)
    location: str = ""
    description: str = ""
    suggested_fix: str = ""
    suggested_stage: Stage | None = None  # only honored for open_ended issues

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "location": self.location,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
        }
        if self.suggested_stage is not None:
            payload["suggested_stage"] = self.suggested_stage.value
        return payload


def _parse_stage(value: Any) -> Stage | None:
    if value is None:
        return None
    try:
        return Stage(str(value))
    except ValueError:
        logger.warning("unknown suggested_stage %r; ignoring", value)
        return None


def validate_and_coerce(raw: Any) -> Issue | None:
    """Coerce a raw agent/scanner item into a typed ``Issue``.

    Accepts any non-empty ``issue_type`` string (open vocabulary). Drops items
    that are not dicts, have empty issue_type, or are otherwise unusable.
    """
    if isinstance(raw, Issue):
        return raw

    if not isinstance(raw, dict):
        logger.warning("dropping non-dict issue item: %r", raw)
        return None

    issue_type = str(raw.get("issue_type", "")).strip()
    if not issue_type:
        logger.warning("dropping issue item with empty issue_type: %r", raw)
        return None

    try:
        severity = int(raw.get("severity", 3))
    except (TypeError, ValueError):
        severity = 3
    severity = max(1, min(5, severity))

    suggested_stage = _parse_stage(raw.get("suggested_stage"))
    # suggested_stage is only meaningful for open_ended discoveries.
    if issue_type != "open_ended" and suggested_stage is not None:
        suggested_stage = None

    return Issue(
        issue_type=issue_type,
        severity=severity,
        location=str(raw.get("location", "")),
        description=str(raw.get("description", "")),
        suggested_fix=str(raw.get("suggested_fix", "")),
        suggested_stage=suggested_stage,
    )


def merge_issues(*issue_lists: Iterable[Issue]) -> list[Issue]:
    """Concatenate issue lists from multiple channels (scanner + agent)."""
    merged: list[Issue] = []
    for issues in issue_lists:
        merged.extend(issues)
    return merged


def write_issues_json(issues: Iterable[Issue], path: Path) -> None:
    """Persist issues to a JSON array (used by the scanner or tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [issue.to_dict() for issue in issues]
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage verdict (agent writes this BEFORE optimizing; the gate reads it)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageVerdictEntry:
    """One stage's verdict from the stage-determination pass."""

    stage: Stage
    has_issues: bool  # True = issues, False = clean
    issues: tuple[Issue, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage.value,
            "verdict": "issues" if self.has_issues else "clean",
            "rationale": self.rationale,
        }
        if self.has_issues:
            payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


@dataclass(frozen=True)
class StageVerdict:
    """Per-stage verdict produced by the agent (current round)."""

    round_name: str
    entries: tuple[StageVerdictEntry, ...]
    determined_stage: Stage | None = None  # advisory; gate re-checks via deps

    def entry(self, stage: Stage) -> StageVerdictEntry | None:
        for entry in self.entries:
            if entry.stage == stage:
                return entry
        return None

    @property
    def clean_stages(self) -> tuple[Stage, ...]:
        return tuple(e.stage for e in self.entries if not e.has_issues)

    @property
    def issues_by_stage(self) -> dict[Stage, list[Issue]]:
        return {
            e.stage: list(e.issues) for e in self.entries if e.has_issues
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "round": self.round_name,
            "verdicts": [e.to_dict() for e in self.entries],
        }
        if self.determined_stage is not None:
            payload["determined_stage"] = self.determined_stage.value
        return payload


def _parse_stage(value: Any) -> Stage | None:  # type: ignore[no-redef]
    if value is None:
        return None
    try:
        return Stage(str(value))
    except ValueError:
        logger.warning("unknown stage in stage-verdict.json: %r", value)
        return None


def coerce_stage_verdict(raw: Any, *, round_name: str = "") -> StageVerdict | None:
    """Coerce a raw stage-verdict.json payload into a typed StageVerdict."""
    if not isinstance(raw, dict):
        logger.warning("stage-verdict payload is not an object: %r", raw)
        return None
    raw_entries = raw.get("verdicts")
    if not isinstance(raw_entries, list):
        logger.warning("stage-verdict.json missing 'verdicts' array")
        return None
    entries: list[StageVerdictEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        stage = _parse_stage(raw_entry.get("stage"))
        if stage is None:
            continue
        verdict_str = str(raw_entry.get("verdict", "")).strip().lower()
        rationale = str(raw_entry.get("rationale", ""))
        if verdict_str == "clean":
            entries.append(
                StageVerdictEntry(stage, has_issues=False, rationale=rationale)
            )
        else:
            raw_issues = raw_entry.get("issues", [])
            if not isinstance(raw_issues, list):
                raw_issues = []
            issues = [
                issue
                for issue in (validate_and_coerce(ri) for ri in raw_issues)
                if issue is not None
            ]
            entries.append(
                StageVerdictEntry(
                    stage,
                    has_issues=bool(issues) or verdict_str == "issues",
                    issues=tuple(issues),
                    rationale=rationale,
                )
            )
    if not entries:
        return None
    determined = _parse_stage(raw.get("determined_stage"))
    return StageVerdict(
        round_name=str(raw.get("round", round_name)),
        entries=tuple(entries),
        determined_stage=determined,
    )


def load_stage_verdict(round_dir: Path) -> StageVerdict | None:
    """Load and coerce ``opt-round-N/stage-verdict.json``."""
    verdict_path = Path(round_dir) / "stage-verdict.json"
    if not verdict_path.is_file():
        return None
    try:
        raw = json.loads(verdict_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("malformed %s: %s", verdict_path, exc)
        return None
    return coerce_stage_verdict(raw)


def write_stage_verdict(verdict: StageVerdict, path: Path) -> None:
    """Persist a StageVerdict (used by tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verdict.to_dict(), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
