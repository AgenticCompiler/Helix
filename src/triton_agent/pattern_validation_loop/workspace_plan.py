from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from triton_agent.pattern_validation_loop.paths import resolve_repo_path
from triton_agent.skill_loader import load_skill_script_module

DEFAULT_KNOWLEDGE_FILE = "PERF_KNOWLEDGE_BASE.md"
DEFAULT_WORKSPACE_PLAN_NAME = "workspace-plan.json"
_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"
_PLAN_SCRIPT = "plan_workspaces_from_knowledge"


def resolve_knowledge_base_path(repo_root: Path, knowledge_output: str = DEFAULT_KNOWLEDGE_FILE) -> Path:
    return resolve_repo_path(repo_root, knowledge_output)


def default_workspace_plan_path(batch_root: Path) -> Path:
    return batch_root / DEFAULT_WORKSPACE_PLAN_NAME


def generate_workspace_plan(
    *,
    repo_root: Path,
    knowledge_path: Path,
    output_path: Path,
    base_revision: str = "",
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build workspace-plan.json from PERF_KNOWLEDGE_BASE.md. Returns payload and warnings."""
    knowledge_path = knowledge_path.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not knowledge_path.is_file():
        return None, [f"knowledge base not found: {knowledge_path}"]

    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, _PLAN_SCRIPT)
    argv = [
        "--knowledge",
        knowledge_path.as_posix(),
        "--repo",
        repo_root.as_posix(),
        "--output",
        output_path.as_posix(),
    ]
    if base_revision.strip():
        argv.extend(["--base", base_revision.strip()])

    exit_code = int(module.main(argv))
    if exit_code != 0:
        raise RuntimeError(
            f"plan_workspaces_from_knowledge failed with exit code {exit_code} "
            f"(knowledge={knowledge_path.as_posix()})",
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    warnings = [str(item) for item in payload.get("warnings", [])]
    return payload, warnings


def generate_workspace_plan_if_present(
    *,
    repo_root: Path,
    batch_root: Path,
    knowledge_output: str = DEFAULT_KNOWLEDGE_FILE,
    base_revision: str = "",
    output_path: Path | None = None,
    stream: Any = None,
) -> tuple[Path | None, list[str]]:
    """Generate plan when the knowledge base file exists; otherwise return (None, [])."""
    knowledge_path = resolve_knowledge_base_path(repo_root, knowledge_output)
    if not knowledge_path.is_file():
        return None, []

    plan_path = output_path or default_workspace_plan_path(batch_root)
    out = stream or sys.stderr
    print(
        f"[pattern-validation-loop] generating workspace plan from {knowledge_path.as_posix()}",
        file=out,
        flush=True,
    )
    payload, warnings = generate_workspace_plan(
        repo_root=repo_root,
        knowledge_path=knowledge_path,
        output_path=plan_path,
        base_revision=base_revision,
    )
    if payload is not None:
        count = int(payload.get("workspace_count", 0))
        print(
            f"[pattern-validation-loop] workspace plan: {plan_path.as_posix()} ({count} workspaces)",
            file=out,
            flush=True,
        )
    return plan_path, warnings
