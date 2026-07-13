# Optimize Session Artifacts Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `OptimizeGuidanceManager` with a thinner `OptimizeSessionArtifactsManager` facade and split optimize artifact handling by domain without changing optimize behavior.

**Architecture:** Move the current mixed responsibilities in `src/helix/optimize/guidance.py` into three focused modules: `memory_file`, `runtime_handoff`, and `archive`. Keep one facade in `session_artifacts.py` so `execution.py` can keep a compact call surface while domain modules own rendering, runtime files, and archive/session-record logic separately.

**Tech Stack:** Python 3.12, `dataclasses`, `pathlib`, `textwrap.dedent`, existing optimize execution plumbing, Python `unittest`

---

### Task 1: Introduce Memory File Module And Preserve Guidance Behavior

**Files:**
- Create: `src/helix/optimize/memory_file.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `src/helix/optimize/guidance.py` only until the facade migration is complete

- [ ] **Step 1: Write failing tests that target a dedicated memory-file module**

Add imports and tests in `tests/test_optimize_guidance.py` for:

```python
from helix.optimize.memory_file import (
    MemoryFileState,
    MemoryFileManager,
)


def test_memory_file_manager_selects_agents_by_default(self) -> None:
    manager = MemoryFileManager()

    self.assertEqual(manager.guidance_filename("codex"), "AGENTS.md")


def test_memory_file_manager_selects_claude_memory_file(self) -> None:
    manager = MemoryFileManager()

    self.assertEqual(manager.guidance_filename("claude"), "CLAUDE.md")
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `python3 -m unittest tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_memory_file_manager_selects_agents_by_default tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_memory_file_manager_selects_claude_memory_file -v`

Expected: FAIL because `helix.optimize.memory_file` does not exist yet.

- [ ] **Step 3: Create the memory-file module with minimal state and filename logic**

Create `src/helix/optimize/memory_file.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MemoryFileState:
    guidance_path: Path
    backup_path: Optional[Path]
    created_guidance: bool


class MemoryFileManager:
    def guidance_filename(self, agent_name: str) -> str:
        if agent_name == "claude":
            return "CLAUDE.md"
        return "AGENTS.md"
```

- [ ] **Step 4: Re-run the targeted tests to verify they pass**

Run: `python3 -m unittest tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_memory_file_manager_selects_agents_by_default tests.test_optimize_guidance.OptimizeGuidanceManagerTests.test_memory_file_manager_selects_claude_memory_file -v`

Expected: PASS.

- [ ] **Step 5: Move temporary memory-file rendering and lifecycle into `MemoryFileManager`**

Move these responsibilities out of `guidance.py` and into `memory_file.py`:

- render unsupervised/shared memory-file text
- backup path calculation
- backup/write preparation
- restore/delete cleanup
- verbose describe helpers for memory-file preparation and cleanup

Preserve the current rendered `AGENTS.md` / `CLAUDE.md` content exactly.

- [ ] **Step 6: Run the full guidance test suite**

Run: `python3 -m unittest tests.test_optimize_guidance -v`

Expected: PASS with no guidance-content regressions.

- [ ] **Step 7: Commit the memory-file extraction**

```bash
git add src/helix/optimize/memory_file.py tests/test_optimize_guidance.py src/helix/optimize/guidance.py
git commit -m "refactor: extract optimize memory file manager"
```

### Task 2: Extract Runtime Handoff Lifecycle From Mixed Guidance Logic

**Files:**
- Create: `src/helix/optimize/runtime_handoff.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `src/helix/optimize/guidance.py` only until the facade migration is complete

- [ ] **Step 1: Write failing tests for runtime handoff state and file creation**

Add tests that import:

```python
from helix.optimize.runtime_handoff import (
    RuntimeHandoffManager,
    RuntimeHandoffState,
)
```

and verify a new manager can create:

- `.helix/round-brief.md`
- `.helix/supervisor-report.md`
- `.helix/history/`

inside a temporary workspace.

- [ ] **Step 2: Run targeted tests to verify failure**

Run the new runtime-handoff tests directly with `python3 -m unittest ... -v`.

Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Create the runtime handoff module**

Create `src/helix/optimize/runtime_handoff.py` with:

- `RuntimeHandoffState`
- runtime-root creation
- initial `round-brief.md` / `supervisor-report.md` seeding
- runtime-tree cleanup for supervised sessions

Move only `.helix/` ownership into this module. Do not move archive logic here.

- [ ] **Step 4: Wire the existing runtime cleanup through the new module**

Update the interim manager/facade path so supervised preparation and cleanup go through `RuntimeHandoffManager`.

- [ ] **Step 5: Run the focused and integration suites**

Run:

```bash
python3 -m unittest tests.test_optimize_guidance -v
python3 -m unittest tests.test_optimize_runtime -v
```

Expected: PASS. Runtime handoff behavior should remain unchanged.

- [ ] **Step 6: Commit the runtime handoff extraction**

```bash
git add src/helix/optimize/runtime_handoff.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py src/helix/optimize/guidance.py
git commit -m "refactor: extract optimize runtime handoff manager"
```

### Task 3: Extract Archive And Session Recording Responsibilities

**Files:**
- Create: `src/helix/optimize/archive.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/helix/optimize/guidance.py` only until the facade migration is complete

- [ ] **Step 1: Write failing tests for archive state and session recording helpers**

Add tests for:

```python
from helix.optimize.archive import ArchiveManager, ArchiveState
```

Cover:

- `agent-sessions.jsonl` append behavior
- archive directory creation under `optimize-logs/helix/<run-id>/`
- shared-guidance snapshot writing
- `final/` and `history/` output copying

- [ ] **Step 2: Run targeted tests to verify failure**

Run the new archive tests directly with `python3 -m unittest ... -v`.

Expected: FAIL because `archive.py` does not exist yet.

- [ ] **Step 3: Create the archive module**

Create `src/helix/optimize/archive.py` with:

- `ArchiveState`
- run-id generation
- archive directory layout
- archive copy logic
- `record_agent_session()`

Keep archive layout and JSONL schema unchanged.

- [ ] **Step 4: Rewire supervised and unsupervised flows to use `ArchiveManager`**

Move archive creation and session-record calls out of the mixed manager path and through the new module.

- [ ] **Step 5: Run the relevant regression suites**

Run:

```bash
python3 -m unittest tests.test_optimize_guidance -v
python3 -m unittest tests.test_optimize_runtime -v
```

Expected: PASS. Archive output and session-recording behavior should remain unchanged.

- [ ] **Step 6: Commit the archive extraction**

```bash
git add src/helix/optimize/archive.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py src/helix/optimize/guidance.py
git commit -m "refactor: extract optimize archive manager"
```

### Task 4: Replace The Mixed Manager With A Session-Artifacts Facade

**Files:**
- Create: `src/helix/optimize/session_artifacts.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_optimize_guidance.py`
- Delete: `src/helix/optimize/guidance.py` or reduce it to a compatibility shim only if necessary

- [ ] **Step 1: Write failing import/update tests for the renamed facade**

Add or update tests so runtime code expects:

```python
from helix.optimize.session_artifacts import OptimizeSessionArtifactsManager
```

Use the existing optimize runtime tests to catch import-path breakage.

- [ ] **Step 2: Run the affected runtime tests to verify failure**

Run the optimize runtime tests that exercise manager construction.

Expected: FAIL until the new facade exists and callers are updated.

- [ ] **Step 3: Create `session_artifacts.py` as the thin facade**

Implement:

- `SharedOptimizeSessionArtifactsState`
- `OptimizeSessionArtifactsState`
- `OptimizeSessionArtifactsManager`

The facade should compose:

- `MemoryFileManager`
- `RuntimeHandoffManager`
- `ArchiveManager`

and expose the small set of session-level prepare/cleanup/describe methods still needed by `execution.py`.

- [ ] **Step 4: Update optimize callers to use the new facade**

Replace `OptimizeGuidanceManager` imports/usages in:

- `src/helix/optimize/execution.py`
- `src/helix/optimize/orchestration.py`

Keep runtime behavior unchanged.

- [ ] **Step 5: Remove or shrink the old `guidance.py`**

If no callers remain, delete `guidance.py`. If a temporary compatibility shim is needed, keep it tiny and re-export only the new facade during the migration.

- [ ] **Step 6: Run the final verification set**

Run:

```bash
python3 -m unittest tests.test_optimize_guidance -v
python3 -m unittest tests.test_optimize_runtime -v
```

Expected: PASS with no optimize behavior regressions.

- [ ] **Step 7: Commit the facade migration**

```bash
git add src/helix/optimize/session_artifacts.py src/helix/optimize/memory_file.py src/helix/optimize/runtime_handoff.py src/helix/optimize/archive.py src/helix/optimize/execution.py src/helix/optimize/orchestration.py tests/test_optimize_guidance.py tests/test_optimize_runtime.py
git commit -m "refactor: split optimize session artifacts by domain"
```

## Self-Review

- Spec coverage: Task 1 extracts memory-file ownership, Task 2 extracts runtime handoff ownership, Task 3 extracts archive/session-record ownership, and Task 4 introduces the renamed facade and caller migration.
- Placeholder scan: Every task names exact files, target behaviors, and verification commands. No `TODO`, `TBD`, or implicit “clean up later” steps remain.
- Type consistency: The plan consistently uses `MemoryFileState`, `RuntimeHandoffState`, `ArchiveState`, `SharedOptimizeSessionArtifactsState`, `OptimizeSessionArtifactsState`, and `OptimizeSessionArtifactsManager`.
