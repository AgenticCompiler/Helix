from __future__ import annotations

from pathlib import Path
from types import ModuleType

from hook_runtime import skill_loader as _runtime_skill_loader


def repo_root() -> Path:
    return _runtime_skill_loader.runtime_root()


def skill_script_root(skill_name: str) -> Path:
    return _runtime_skill_loader.skill_script_root(skill_name)


def skill_script_path(skill_name: str, script_name: str) -> Path:
    return _runtime_skill_loader.skill_script_path(skill_name, script_name)


def operator_eval_skill_root() -> Path:
    return _runtime_skill_loader.operator_eval_skill_root()


def operator_eval_script_path(script_name: str) -> Path:
    return _runtime_skill_loader.operator_eval_script_path(script_name)


def load_skill_script_module(skill_name: str, script_name: str) -> ModuleType:
    return _runtime_skill_loader.load_skill_script_module(skill_name, script_name)


def load_operator_eval_script_module(script_name: str) -> ModuleType:
    return _runtime_skill_loader.load_operator_eval_script_module(script_name)
