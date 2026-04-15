from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_skill_root() -> Path:
    return repo_root() / "skills" / "operator-eval"


def run_skill_script_path(script_name: str) -> Path:
    path = run_skill_root() / "scripts" / f"{script_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Run skill script does not exist: {path}")
    return path


@lru_cache(maxsize=None)
def load_run_skill_module(script_name: str) -> ModuleType:
    path = run_skill_script_path(script_name)
    module_name = f"run_skill_{script_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load run skill script: {path}")
    module = importlib.util.module_from_spec(spec)
    script_dir = str(path.parent)
    added = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        added = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(script_dir)
    return module
