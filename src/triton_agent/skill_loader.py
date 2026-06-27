from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from triton_agent.resources import application_root
from triton_agent.skill_catalog import resolve_skill_source_dir


def repo_root() -> Path:
    return application_root()


def skill_script_root(skill_name: str) -> Path:
    return resolve_skill_source_dir(skill_name)


def skill_script_path(skill_name: str, script_name: str) -> Path:
    relative = Path(script_name + ".py")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid skill script path: {script_name!r}")
    path = skill_script_root(skill_name) / "scripts" / relative
    if not path.exists():
        raise FileNotFoundError(f"Skill script does not exist: {path}")
    return path


def operator_eval_skill_root() -> Path:
    return skill_script_root("ascend-npu-run-eval")


def operator_eval_script_path(script_name: str) -> Path:
    return skill_script_path("ascend-npu-run-eval", script_name)


@lru_cache(maxsize=None)
def load_skill_script_module(skill_name: str, script_name: str) -> ModuleType:
    path = skill_script_path(skill_name, script_name)
    module_name = (
        f"skill_{skill_name.replace('-', '_')}_"
        f"{script_name.replace('/', '_').replace('-', '_')}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load skill script: {path}")
    module = importlib.util.module_from_spec(spec)
    scripts_root = str(skill_script_root(skill_name) / "scripts")
    added = False
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
        added = True
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if added:
            sys.path.remove(scripts_root)
    return module


def load_operator_eval_script_module(script_name: str) -> ModuleType:
    return load_skill_script_module("ascend-npu-run-eval", script_name)
