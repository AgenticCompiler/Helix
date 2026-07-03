from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_json_object(path: Path, *, display_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {display_name} in {path.parent}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {display_name} in {path.parent}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{display_name} must contain an object in {path.parent}")
    return cast(dict[str, Any], payload)


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
