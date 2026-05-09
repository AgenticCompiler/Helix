from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


_CODEX_HOOK_DIR = Path(".codex") / "triton-agent-hooks"
_CODEX_HOOKS_JSON = Path(".codex") / "hooks.json"
_CODEX_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect staged skill implementation files under .codex/skills/*/scripts/. "
    "Use the skill instructions and documented command interface instead."
)
_CODEX_DENY_READ_GLOBS = (Path(".codex") / "skills" / "*" / "scripts" / "**",)


@dataclass(frozen=True)
class AgentHookState:
    created_paths: list[Path]


class AgentHookManager:
    def __init__(self, hooks_root: Path) -> None:
        self.hooks_root = hooks_root

    def prepare_hooks(self, backend: str, workdir: Path) -> AgentHookState:
        if backend.lower() != "codex":
            return AgentHookState(created_paths=[])
        return self._prepare_codex_hooks(workdir)

    def cleanup(self, state: AgentHookState) -> list[str]:
        warnings: list[str] = []
        for path in reversed(state.created_paths):
            try:
                if not path.exists() and not path.is_symlink():
                    continue
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except OSError as exc:
                warnings.append(f"Failed to clean up staged agent hook path {path}: {exc}")
        return warnings

    def describe_prepare(self, state: AgentHookState) -> list[str]:
        if not state.created_paths:
            return ["No backend-specific hooks staged."]
        return [f"Staged agent hooks: {', '.join(str(path) for path in state.created_paths)}"]

    def describe_cleanup(self, state: AgentHookState) -> list[str]:
        if not state.created_paths:
            return ["No backend-specific hooks to clean up."]
        return [f"Cleaning up staged agent hooks: {', '.join(str(path) for path in state.created_paths)}"]

    def _prepare_codex_hooks(self, workdir: Path) -> AgentHookState:
        workspace = workdir.absolute()
        policy_workspace = workspace.resolve()
        template_dir = self.hooks_root / "codex"
        if not template_dir.is_dir():
            raise RuntimeError(f"Codex hook template directory does not exist: {template_dir}")

        hooks_json = workspace / _CODEX_HOOKS_JSON
        hook_dir = workspace / _CODEX_HOOK_DIR
        if hooks_json.exists() or hooks_json.is_symlink():
            raise RuntimeError(f"Existing Codex hooks config must not be overwritten: {hooks_json}")
        if hook_dir.exists() or hook_dir.is_symlink():
            raise RuntimeError(f"Existing Codex hook directory must not be overwritten: {hook_dir}")

        created_paths: list[Path] = []
        state = AgentHookState(created_paths=created_paths)
        try:
            hooks_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template_dir / "hooks.json", hooks_json)
            created_paths.append(hooks_json)

            shutil.copytree(
                template_dir,
                hook_dir,
                ignore=lambda _directory, names: {"hooks.json"} if "hooks.json" in names else set(),
            )
            created_paths.append(hook_dir)
            self._write_codex_policy(hook_dir / "policy.json", policy_workspace)
        except Exception:
            self.cleanup(state)
            raise

        return state

    def _write_codex_policy(self, policy_path: Path, workspace: Path) -> None:
        policy = {
            "workspace_root": str(workspace),
            "allow_read_roots": [str(workspace)],
            "deny_read_globs": [str(workspace / pattern) for pattern in _CODEX_DENY_READ_GLOBS],
            "deny_message": _CODEX_DENY_MESSAGE,
        }
        policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
