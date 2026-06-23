from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from triton_agent.backends.hook_common import HookStageOptions, HookStageState, cleanup_hook_stage
from triton_agent.otel_trace import append_trace_event, utc_timestamp


_CODEX_HOOK_DIR = Path(".codex") / "triton-agent-hooks"
_CODEX_HOOKS_JSON = Path(".codex") / "hooks.json"
_CODEX_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected files (staged skill implementation files under "
    ".codex/skills/*/scripts/ or triton-agent-logs/ output). "
    "Use the skill instructions and documented command interface instead."
)
_SHARED_DENY_READ_GLOBS = (Path("triton-agent-logs") / "**",)
_CODEX_DENY_READ_GLOBS = _SHARED_DENY_READ_GLOBS + (Path(".codex") / "skills" / "*" / "scripts" / "**",)


def prepare_codex_hooks(
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
    template_dir = hooks_root / "codex"
    shared_template_dir = hooks_root / "shared"
    if not template_dir.is_dir():
        raise RuntimeError(f"Codex hook template directory does not exist: {template_dir}")
    if not shared_template_dir.is_dir():
        raise RuntimeError(f"Shared hook template directory does not exist: {shared_template_dir}")
    guard_template = template_dir / "pretooluse_guard.py"
    policy_engine_template = shared_template_dir / "tool_use_guard_policy.py"
    if not guard_template.is_file():
        raise RuntimeError(f"Codex hook guard template does not exist: {guard_template}")
    if not policy_engine_template.is_file():
        raise RuntimeError(f"Shared guard policy template does not exist: {policy_engine_template}")

    hooks_json = workspace / _CODEX_HOOKS_JSON
    hook_dir = workspace / _CODEX_HOOK_DIR
    if hooks_json.exists() or hooks_json.is_symlink():
        raise RuntimeError(f"Existing Codex hooks config must not be overwritten: {hooks_json}")
    if hook_dir.exists() or hook_dir.is_symlink():
        raise RuntimeError(f"Existing Codex hook directory must not be overwritten: {hook_dir}")

    created_paths: list[Path] = []
    state = HookStageState(created_paths=created_paths)
    try:
        hooks_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_dir / "hooks.json", hooks_json)
        created_paths.append(hooks_json)

        hook_dir.mkdir(parents=True)
        shutil.copy2(guard_template, hook_dir / "pretooluse_guard.py")
        shutil.copy2(policy_engine_template, hook_dir / "tool_use_guard_policy.py")
        shutil.copy2(template_dir / "tool_trace_hook.py", hook_dir / "tool_trace_hook.py")
        created_paths.append(hook_dir)
        _write_codex_policy(
            hook_dir / "policy.json",
            policy_workspace,
            options,
            extra_allowed_read_roots=extra_allowed_read_roots,
        )

        if options.trace_enabled and options.trace_path is not None:
            _write_trace_setup_event(options, hook_dir)
    except Exception:
        cleanup_hook_stage(state)
        raise

    return state


def _write_codex_policy(
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
            "deny_read_globs": [str(workspace / pattern) for pattern in _CODEX_DENY_READ_GLOBS],
            "deny_message": _CODEX_DENY_MESSAGE,
        },
        "allow_read_roots": allow_read_roots,
        "deny_read_globs": [str(workspace / pattern) for pattern in _CODEX_DENY_READ_GLOBS],
        "deny_message": _CODEX_DENY_MESSAGE,
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


def _write_trace_setup_event(options: HookStageOptions, hook_dir: Path) -> None:
    if options.trace_path is None:
        return
    append_trace_event(options.trace_path, {
        "schema_version": 1,
        "type": "diagnostic",
        "phase": "instant",
        "code": "trace_setup",
        "detail": f"Codex trace hooks staged: hook_dir={hook_dir}, run_id={options.run_id}",
        "source": "codex_hook",
        "confidence": "high",
        "run_id": options.run_id or "",
        "timestamp": utc_timestamp(),
    })
