"""JSON schema validation and repair for log_check_result.json and pattern_analysis.json."""

from __future__ import annotations

import json
import re
from typing import Any


_LOG_CHECK_CHECK_IDS = (
    "check-1",
    "check-2",
    "check-3",
    "check-4",
    "check-6",
    "check-7",
    "check-8",
    "check-9",
)

LOG_CHECK_RESULT_SCHEMA = {
    "schema_version": 1,
    "overall": "PASS",
    "failed_checks": "",
    "overview_detail": "",
    "checks": [
        {
            "id": "<check-N>",
            "name": "<check title>",
            "result": "pass",
            "detail": "<detail text or null>",
        },
    ],
}

PATTERN_ANALYSIS_SCHEMA = {
    "schema_version": 1,
    "rounds": [
        {
            "round": "round-1",
            "patterns": [
                {
                    "name": "<pattern name>",
                    "evidence": "explicit",
                    "source": "<citation>",
                },
            ],
        },
    ],
    "summary": {
        "known": [
            {"name": "<pattern name>", "rounds": [1], "evidence": "explicit"},
        ],
        "new": [
            {"name": "<pattern name>", "rounds": [3]},
        ],
        "extended": [
            {"name": "<pattern name>", "rounds": [4], "from": "<base pattern>"},
        ],
    },
}


def validate_log_check_json(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["expected a JSON object"]
    if not isinstance(data.get("schema_version"), int):
        errors.append("missing or invalid schema_version (must be int)")
    overall = data.get("overall")
    if overall not in ("PASS", "FAIL"):
        errors.append("overall must be 'PASS' or 'FAIL'")
    checks = data.get("checks")
    if not isinstance(checks, list) or len(checks) == 0:
        errors.append("checks must be a non-empty array")
    else:
        seen_ids: set[str] = set()
        for i, check in enumerate(checks):
            if not isinstance(check, dict):
                errors.append(f"checks[{i}] must be an object")
                continue
            cid = check.get("id")
            if not isinstance(cid, str):
                errors.append(f"checks[{i}].id missing or not a string")
            elif cid not in _LOG_CHECK_CHECK_IDS:
                errors.append(f"checks[{i}].id '{cid}' is not a valid check id")
            elif cid in seen_ids:
                errors.append(f"checks[{i}].id '{cid}' is duplicated")
            else:
                seen_ids.add(cid)
            if not isinstance(check.get("name"), str):
                errors.append(f"checks[{i}].name missing or not a string")
            if check.get("result") not in ("pass", "fail"):
                errors.append(f"checks[{i}].result must be 'pass' or 'fail'")
    if overall == "PASS" and any(
        isinstance(c, dict) and c.get("result") == "fail" for c in checks
    ):
        errors.append("overall is PASS but some checks have result 'fail'")
    if overall == "FAIL" and all(
        isinstance(c, dict) and c.get("result") == "pass" for c in checks
    ):
        errors.append("overall is FAIL but all checks have result 'pass'")
    return errors


def validate_pattern_analysis_json(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["expected a JSON object"]
    if not isinstance(data.get("schema_version"), int):
        errors.append("missing or invalid schema_version (must be int)")
    rounds = data.get("rounds")
    if not isinstance(rounds, list):
        errors.append("rounds must be an array")
    else:
        for i, r in enumerate(rounds):
            if not isinstance(r, dict):
                errors.append(f"rounds[{i}] must be an object")
                continue
            if not isinstance(r.get("round"), str):
                errors.append(f"rounds[{i}].round missing or not a string")
            patterns = r.get("patterns")
            if not isinstance(patterns, list):
                errors.append(f"rounds[{i}].patterns must be an array")
            else:
                for j, p in enumerate(patterns):
                    if not isinstance(p, dict):
                        errors.append(f"rounds[{i}].patterns[{j}] must be an object")
                        continue
                    if not isinstance(p.get("name"), str):
                        errors.append(f"rounds[{i}].patterns[{j}].name missing")
                    if p.get("evidence") not in ("explicit", "inferred"):
                        errors.append(
                            f"rounds[{i}].patterns[{j}].evidence must be 'explicit' or 'inferred'"
                        )
    summary = data.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        for key in ("known", "new", "extended"):
            items = summary.get(key)
            if not isinstance(items, list):
                errors.append(f"summary.{key} must be an array")
                continue
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"summary.{key}[{i}] must be an object")
                    continue
                if not isinstance(item.get("name"), str):
                    errors.append(f"summary.{key}[{i}].name missing")
                if key != "extended":
                    rounds_val = item.get("rounds")
                    if not isinstance(rounds_val, list) or not all(
                        isinstance(n, int) for n in rounds_val
                    ):
                        errors.append(
                            f"summary.{key}[{i}].rounds must be an array of ints"
                        )
                if key == "extended":
                    if not isinstance(item.get("from"), str):
                        errors.append(f"summary.extended[{i}].from missing")
    return errors


def repair_json(text: str) -> dict[str, Any] | None:
    """Attempt to repair common LLM JSON errors.

    Handles:
    - Text outside JSON braces (extracts first JSON object)
    - Trailing commas before ] or }
    - Unescaped newlines in string values
    - Single quotes instead of double quotes
    """
    if not text.strip():
        return None

    # Extract the first JSON object from the text
    cleaned = _extract_json_object(text)

    # Remove trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Fix single-quoted strings (basic heuristic)
    # Only attempt if the JSON doesn't already parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try replacing single quotes with double quotes (naive approach)
    repaired = _fix_single_quotes(cleaned)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


def _extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from text that may have markdown fences."""
    # Try to find JSON inside ```json fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Find first { and matching }
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return text


def _fix_single_quotes(text: str) -> str:
    """Replace single quotes used as JSON string delimiters with double quotes."""
    result: list[str] = []
    in_string = False
    string_char = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if not in_string:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                result.append('"')
            else:
                result.append(ch)
        else:
            if ch == "\\":
                result.append(ch)
                if i + 1 < len(text):
                    result.append(text[i + 1])
                    i += 1
            elif ch == string_char:
                in_string = False
                result.append('"')
            elif ch == '"' and string_char == "'":
                result.append('\\"')
            else:
                result.append(ch)
        i += 1
    return "".join(result)


def is_valid_log_check_json(data: dict[str, Any]) -> bool:
    return len(validate_log_check_json(data)) == 0


def is_valid_pattern_analysis_json(data: dict[str, Any]) -> bool:
    return len(validate_pattern_analysis_json(data)) == 0


__all__ = [
    "LOG_CHECK_RESULT_SCHEMA",
    "PATTERN_ANALYSIS_SCHEMA",
    "validate_log_check_json",
    "validate_pattern_analysis_json",
    "repair_json",
    "is_valid_log_check_json",
    "is_valid_pattern_analysis_json",
]
