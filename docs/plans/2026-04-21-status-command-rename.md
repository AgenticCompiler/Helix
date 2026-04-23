# Status Command Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `optimize-status` to `status`, move its implementation into a dedicated `status/` package, and give it its own command handler without changing status behavior.

**Architecture:** Keep status behavior unchanged while relocating its command and runtime code out of `optimize/`. Split the work into CLI surface updates, runtime package extraction, command-handler ownership cleanup, and focused docs plus regression coverage so the repository ends up with `commands/status.py` and `status/{core,render}.py` as the canonical home for status reporting.

**Tech Stack:** Python 3.12, `argparse`, `pathlib`, existing optimize status helpers, Python `unittest`

---

### Task 1: Rename the CLI surface from `optimize-status` to `status`

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/__init__.py`

- [ ] **Step 1: Write the failing parser and help tests**

Update CLI tests so they assert the new command surface:

```python
cases = [
    ("gen_eval_batch", CommandKind.GEN_EVAL_BATCH),
    ("status", CommandKind.STATUS),
    ("verify_batch", CommandKind.VERIFY_BATCH),
]

args = parser.parse_args(["status", "-i", "kernels"])
self.assertEqual(args.command_kind, CommandKind.STATUS)

help_text = parser.format_help()
self.assertIn("Status:", help_text)
self.assertIn("status", help_text)
self.assertNotIn("optimize-status", help_text)
self.assertNotIn("optimize_status", help_text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_cli.PathResolutionTests.test_main_optimize_status_rejects_missing_root -v`
Expected: FAIL because the parser still exposes `optimize-status`, still uses `CommandKind.OPTIMIZE_STATUS`, and still groups the command under optimization.

- [ ] **Step 3: Write minimal implementation**

Rename the command kind and parser registration:

```python
class CommandKind(str, Enum):
    ...
    STATUS = "status"
    OPTIMIZE = "optimize"
    OPTIMIZE_BATCH = "optimize-batch"
```

Update the CLI command spec and help groups:

```python
from triton_agent.commands.status import handle_status

CommandKind.STATUS: _CommandSpec(
    handler=handle_status,
    help_group="Status",
    help_summary="Show optimization status for one workspace.",
    description="Show optimization status for one workspace.",
    has_output=False,
    has_format=True,
),
```

Remove the old alias and keep examples current:

```python
_TOP_LEVEL_EXAMPLES = (
    "triton-agent gen-test -i kernel.py",
    "triton-agent verify -i .",
    "triton-agent status -i .",
    "triton-agent optimize -i kernel.py --agent codex",
)

aliases = {
    "gen_eval": "gen-eval",
    "verify_batch": "verify-batch",
    "optimize_batch": "optimize-batch",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_cli.PathResolutionTests.test_main_optimize_status_rejects_missing_root -v`
Expected: PASS

### Task 2: Extract status runtime into a dedicated package

**Files:**
- Create: `src/triton_agent/status/__init__.py`
- Create: `src/triton_agent/status/core.py`
- Create: `src/triton_agent/status/render.py`
- Modify: `src/triton_agent/optimize/render.py`
- Delete: `src/triton_agent/optimize/status.py`
- Rename: `tests/test_optimize_status.py` to `tests/test_status.py`
- Rename: `tests/test_optimize_render.py` to `tests/test_status_render.py`

- [ ] **Step 1: Write the failing import-path tests**

Update the status-focused tests to import from the new package paths:

```python
from triton_agent.status.core import (
    inspect_optimize_status_workspace,
    parse_logged_best_round,
    workspace_has_optimize_artifacts,
)
```

```python
from triton_agent.optimize.models import OptimizeStatusWorkspace
from triton_agent.status.render import render_optimize_status_results
```

Keep the behavior assertions unchanged so the move stays behavior-preserving.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_status tests.test_status_render -v`
Expected: FAIL because `triton_agent.status` does not exist yet and the render helpers still live under `optimize/render.py`.

- [ ] **Step 3: Write minimal implementation**

Create the new status package and move the status-only code:

```python
# src/triton_agent/status/__init__.py
from triton_agent.status.core import (
    inspect_optimize_status_workspace,
    scan_optimize_status_workspaces,
    workspace_has_optimize_artifacts,
)
from triton_agent.status.render import render_optimize_status_results

__all__ = [
    "inspect_optimize_status_workspace",
    "scan_optimize_status_workspaces",
    "workspace_has_optimize_artifacts",
    "render_optimize_status_results",
]
```

```python
# src/triton_agent/optimize/render.py
from triton_agent.optimize.models import BatchOptimizeResult

def render_batch_optimize_results(
    results: list[BatchOptimizeResult],
    stdout: TextIO | None = None,
) -> int:
    ...
```

Copy the existing status inspection logic into `status/core.py`, copy the existing status rendering logic into `status/render.py`, and remove `optimize/status.py` after imports are updated.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_status tests.test_status_render -v`
Expected: PASS

### Task 3: Give status its own command handler and update dependent imports

**Files:**
- Create: `src/triton_agent/commands/status.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/commands/__init__.py`
- Modify: `src/triton_agent/verification/core.py`
- Modify: `tests/test_verify.py`
- Modify: `tests/test_verify_batch.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing handler-ownership and import tests**

Update tests so verification and CLI paths use the new status modules:

```python
from triton_agent.status.core import inspect_optimize_status_workspace

args = parser.parse_args(["status", "-i", "workspace"])
self.assertEqual(args.command_kind, CommandKind.STATUS)
```

Add or adjust a CLI-level assertion so the status command resolves through `commands/status.py`, not `commands/optimize.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_cli tests.test_verify tests.test_verify_batch -v`
Expected: FAIL because the status handler still lives in `commands/optimize.py` and verification still imports `triton_agent.optimize.status`.

- [ ] **Step 3: Write minimal implementation**

Create a dedicated status handler:

```python
from triton_agent.status.core import (
    inspect_optimize_status_workspace,
    scan_optimize_status_workspaces,
    workspace_has_optimize_artifacts,
)
from triton_agent.status.render import render_optimize_status_results

def handle_status(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    ...
```

Update verification to use the new import path:

```python
from triton_agent.status.core import inspect_optimize_status_workspace
```

Remove the status handler from `commands/optimize.py` once the CLI imports `handle_status` from `commands/status.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_cli tests.test_verify tests.test_verify_batch -v`
Expected: PASS

### Task 4: Update docs and run focused regression coverage

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-04-21-status-command-rename-design.md`
- Modify: `docs/specs/2026-04-21-verification-command-rename-design.md`
- Modify: `docs/plans/2026-04-21-status-command-rename.md`

- [ ] **Step 1: Update command references in active docs**

Replace user-facing command examples with `status`:

```md
- `status`: summarize optimization progress across many workspaces.

uv run triton-agent status --input operators_root
uv run triton-agent status --input .
uv run triton-agent status --input operators_root --format markdown
```

Keep explanatory text explicit that `status` still reports optimization progress.

- [ ] **Step 2: Run focused regression checks**

Run: `uv run python -m unittest tests.test_cli tests.test_status tests.test_status_render tests.test_verify tests.test_verify_batch -v`
Expected: PASS

- [ ] **Step 3: Run repository verification for the changed surface**

Run:
- `uv run --group dev ruff check src tests`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

Expected:
- Ruff: `All checks passed!`
- Pyright: `0 errors, 0 warnings, 0 informations`
- Unittest: all tests PASS
