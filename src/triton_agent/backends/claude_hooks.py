from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from triton_agent.backends.hook_common import (
    HookStageOptions,
    HookStageState,
    cleanup_hook_stage,
    replace_string_placeholder,
)


_CLAUDE_HOOK_DIR = Path(".claude") / "triton-agent-hooks"
_CLAUDE_SETTINGS_JSON = _CLAUDE_HOOK_DIR / "settings.json"
_CLAUDE_PROJECT_DIR = "${CLAUDE_PROJECT_DIR}"
_CLAUDE_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected runner-managed files (temporary optimize runtime files, "
    "staged skill implementation files under .claude/skills/*/scripts/, or triton-agent-logs/ "
    "output). "
    "Use the skill instructions and documented command interface instead."
)
_SHARED_DENY_READ_GLOBS = (
    Path(".triton-agent"),
    Path(".triton-agent") / "**",
    Path("triton-agent-logs") / "**",
)
_CLAUDE_DENY_READ_GLOBS = _SHARED_DENY_READ_GLOBS + (
    Path(".claude") / "triton-agent-hooks",
    Path(".claude") / "triton-agent-hooks" / "**",
    Path(".claude") / "skills" / "*" / "scripts" / "**",
)


def prepare_claude_hooks(
    hooks_root: Path,
    workdir: Path,
    options: HookStageOptions | None = None,
    *,
    extra_allowed_read_roots: Sequence[Path] = (),
) -> HookStageState:
    options = options or HookStageOptions(guard_enabled=True)
    if not options.guard_enabled:
        return HookStageState(created_paths=[])

    workspace = workdir.absolute()
    policy_workspace = workspace.resolve()
    settings_template = hooks_root / "claude" / "settings.json"
    guard_template = hooks_root / "claude" / "pretooluse_guard.py"
    hook_runtime_template = hooks_root.parent / "src" / "hook_runtime"
    if not settings_template.is_file():
        raise RuntimeError(f"Claude hook settings template does not exist: {settings_template}")
    if not guard_template.is_file():
        raise RuntimeError(f"Claude hook guard template does not exist: {guard_template}")
    if not hook_runtime_template.is_dir():
        raise RuntimeError(f"Hook runtime template directory does not exist: {hook_runtime_template}")

    hook_dir = workspace / _CLAUDE_HOOK_DIR
    settings_json = workspace / _CLAUDE_SETTINGS_JSON
    if hook_dir.exists() or hook_dir.is_symlink():
        raise RuntimeError(f"Existing Claude hook directory must not be overwritten: {hook_dir}")

    created_paths: list[Path] = []
    state = HookStageState(created_paths=created_paths)
    try:
        hook_dir.parent.mkdir(parents=True, exist_ok=True)
        hook_dir.mkdir(parents=True)
        shutil.copy2(guard_template, hook_dir / "pretooluse_guard.py")
        shutil.copytree(hook_runtime_template, hook_dir / "hook_runtime")
        _write_claude_policy(
            hook_dir / "policy.json",
            policy_workspace,
            options,
            extra_allowed_read_roots=extra_allowed_read_roots,
        )
        _write_claude_settings(settings_template, settings_json, policy_workspace)
        created_paths.append(settings_json)
        created_paths.append(hook_dir)
    except Exception:
        cleanup_hook_stage(state)
        raise

    return state


def _write_claude_policy(
    policy_path: Path,
    workspace: Path,
    options: HookStageOptions,
    *,
    extra_allowed_read_roots: Sequence[Path] = (),
) -> None:
    allow_read_roots = _allow_read_roots(workspace, extra_allowed_read_roots)
    policy = {
        "workspace_root": str(workspace),
        "trace": _trace_policy(options),
        "guard": {
            "enabled": options.guard_enabled,
            "allow_read_roots": allow_read_roots,
            "deny_read_globs": [str(workspace / pattern) for pattern in _CLAUDE_DENY_READ_GLOBS],
            "deny_message": _CLAUDE_DENY_MESSAGE,
        },
        "allow_read_roots": allow_read_roots,
        "deny_read_globs": [str(workspace / pattern) for pattern in _CLAUDE_DENY_READ_GLOBS],
        "deny_message": _CLAUDE_DENY_MESSAGE,
    }
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def _write_claude_settings(settings_template: Path, settings_path: Path, project_dir: Path) -> None:
    settings = json.loads(settings_template.read_text(encoding="utf-8"))
    settings = replace_string_placeholder(settings, _CLAUDE_PROJECT_DIR, str(project_dir))
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


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
