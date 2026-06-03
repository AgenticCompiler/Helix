from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triton_agent.skill_loader import load_skill_script_module

_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"
_AUDIT_SCRIPT = "audit_batch"


def collect_batch_evidence(
    batch_root: Path,
    *,
    output_path: Path | None = None,
    include_completed: bool = False,
) -> dict[str, Any]:
    """Collect optimize-round evidence JSON for agent review."""
    batch_root = batch_root.expanduser().resolve()
    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, _AUDIT_SCRIPT)
    argv = ["--batch-root", batch_root.as_posix(), "--json"]
    if include_completed:
        argv.append("--include-completed")
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        argv.extend(["--output", output_path.as_posix()])

    exit_code = int(module.main(argv))
    if exit_code != 0:
        raise RuntimeError(f"audit_batch evidence collection failed with exit code {exit_code}")

    if output_path is not None:
        return json.loads(output_path.read_text(encoding="utf-8"))
    raise RuntimeError("audit_batch --json without --output did not persist a report path")


def reset_active_workspace_rounds(batch_root: Path) -> None:
    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, "reset_workspace_rounds")
    exit_code = int(module.main(["--batch-root", batch_root.expanduser().resolve().as_posix()]))
    if exit_code != 0:
        raise RuntimeError(f"reset_workspace_rounds failed with exit code {exit_code}")
