from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from triton_agent.backends.hook_common import HookStageOptions, HookStageState, cleanup_hook_stage


_OPENCODE_HOOK_DIR = Path(".opencode") / "triton-agent-hooks"
_OPENCODE_PLUGIN_FILE = Path(".opencode") / "plugins" / "triton-agent-hook-guard.js"
_OPENCODE_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected files (staged skill implementation files under "
    ".opencode/skills/*/scripts/ or triton-agent-logs/ output). "
    "Use the skill instructions and documented command interface instead."
)
_SHARED_DENY_READ_GLOBS = (Path("triton-agent-logs") / "**",)
_OPENCODE_DENY_READ_GLOBS = _SHARED_DENY_READ_GLOBS + (Path(".opencode") / "skills" / "*" / "scripts" / "**",)


def prepare_opencode_hooks(
    hooks_root: Path,
    workdir: Path,
    options: HookStageOptions | None = None,
    *,
    extra_allowed_read_roots: Sequence[Path] = (),
) -> HookStageState:
    options = options or HookStageOptions(guard_enabled=True)
    if not options.trace_enabled and not options.guard_enabled:
        return HookStageState(created_paths=[])

    workspace = workdir.absolute()
    policy_workspace = workspace.resolve()
    template_dir = hooks_root / "opencode"
    if not template_dir.is_dir():
        raise RuntimeError(f"OpenCode hook template directory does not exist: {template_dir}")

    plugin_file = workspace / _OPENCODE_PLUGIN_FILE
    hook_dir = workspace / _OPENCODE_HOOK_DIR
    if plugin_file.exists() or plugin_file.is_symlink():
        raise RuntimeError(f"Existing OpenCode hook plugin must not be overwritten: {plugin_file}")
    if hook_dir.exists() or hook_dir.is_symlink():
        raise RuntimeError(f"Existing OpenCode hook directory must not be overwritten: {hook_dir}")

    created_paths: list[Path] = []
    state = HookStageState(created_paths=created_paths)
    try:
        plugin_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_dir / "triton-agent-hook-guard.js", plugin_file)
        created_paths.append(plugin_file)

        hook_dir.mkdir(parents=True)
        created_paths.append(hook_dir)
        _write_opencode_policy(
            hook_dir / "policy.json",
            policy_workspace,
            options,
            extra_allowed_read_roots=extra_allowed_read_roots,
        )
    except Exception:
        cleanup_hook_stage(state)
        raise

    return state


def _write_opencode_policy(
    policy_path: Path,
    workspace: Path,
    options: HookStageOptions,
    *,
    extra_allowed_read_roots: Sequence[Path] = (),
) -> None:
    allow_read_roots = _allow_read_roots(workspace, extra_allowed_read_roots)
    protected_script_roots = [str((workspace / ".opencode" / "skills").resolve())]
    policy = {
        "workspace_root": str(workspace),
        "trace": _trace_policy(options),
        "guard": {
            "enabled": options.guard_enabled,
            "allow_read_roots": allow_read_roots,
            "protected_script_roots": protected_script_roots,
            "deny_read_globs": [str(workspace / pattern) for pattern in _OPENCODE_DENY_READ_GLOBS],
            "deny_message": _OPENCODE_DENY_MESSAGE,
        },
        "allow_read_roots": allow_read_roots,
        "protected_script_roots": protected_script_roots,
        "deny_read_globs": [str(workspace / pattern) for pattern in _OPENCODE_DENY_READ_GLOBS],
        "deny_message": _OPENCODE_DENY_MESSAGE,
    }
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def _allow_read_roots(workspace: Path, extra_allowed_read_roots: Sequence[Path]) -> list[str]:
    roots = [str(workspace)]
    seen = {workspace}
    for root in extra_allowed_read_roots:
        resolved_root = root.expanduser().resolve()
        if resolved_root in seen:
            continue
        roots.append(str(resolved_root))
        seen.add(resolved_root)
    return roots


def _trace_policy(options: HookStageOptions) -> dict[str, str | bool | None]:
    return {
        "enabled": options.trace_enabled,
        "path": str(options.trace_path) if options.trace_path is not None else None,
        "run_id": options.run_id,
    }
