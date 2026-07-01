from __future__ import annotations

import sys
from pathlib import Path


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(str(bundle_root)).resolve()
        return Path(sys.executable).resolve().parent
    root = Path(__file__).resolve().parents[2]
    if (root / "skills").is_dir():
        return root
    return root


def skills_root() -> Path:
    return application_root() / "skills"
