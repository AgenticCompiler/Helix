# Clean Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `clean` CLI subcommand that removes known generated artifacts from one workspace or a batch root, with optional `--deep` cleanup that also removes generated test and benchmark cases.

**Architecture:** Add a thin CLI handler plus a focused cleanup module that owns workspace discovery, known-artifact selection, and deletion. Reuse existing optimize operator resolution and `status`-style workspace-or-batch-root behavior while widening single-workspace detection to include non-optimize generated artifacts such as `triton_<op>.py`, `PROF_*`, and `extra-info.json`.

**Tech Stack:** Python 3.12, `argparse`, `pathlib`, `unittest`

---

### Task 1: Add parser coverage for `clean`

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `src/helix/models.py`

- [ ] **Step 1: Write the failing parser tests**

```python
    def test_clean_maps_to_command_kind(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["clean", "-i", "workspace"])
        self.assertEqual(args.command, "clean")
        self.assertEqual(args.command_kind, CommandKind.CLEAN)
        self.assertFalse(args.deep)
        self.assertFalse(hasattr(args, "agent"))

    def test_clean_accepts_deep_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["clean", "-i", "workspace", "--deep", "--verbose"])
        self.assertTrue(args.deep)
        self.assertTrue(args.verbose)
```

- [ ] **Step 2: Run the focused parser tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_clean_maps_to_command_kind tests.test_cli.CliParserTests.test_clean_accepts_deep_option -v`

Expected: FAIL because `clean` / `CommandKind.CLEAN` do not exist yet.

- [ ] **Step 3: Add the command kind placeholder**

```python
class CommandKind(str, Enum):
    ...
    REPORT = "report"
    REPORT_BATCH = "report-batch"
    CLEAN = "clean"
```

```python
COMMAND_TO_SKILL = {
    ...
    CommandKind.REPORT: "",
    CommandKind.REPORT_BATCH: "",
    CommandKind.CLEAN: "",
}
```

- [ ] **Step 4: Re-run the focused parser tests**

Run: `uv run python -m unittest tests.test_cli.CliParserTests.test_clean_maps_to_command_kind tests.test_cli.CliParserTests.test_clean_accepts_deep_option -v`

Expected: still FAIL because the parser does not register `clean` yet.

### Task 2: Implement cleanup discovery and deletion logic

**Files:**
- Create: `src/helix/cleaning.py`
- Test: `tests/test_cleaning.py`

- [ ] **Step 1: Write the failing cleanup-module tests**

```python
class WorkspaceCleaningTests(unittest.TestCase):
    def test_collect_workspace_cleanup_targets_preserves_operator_and_cases_by_default(self) -> None:
        ...

    def test_collect_workspace_cleanup_targets_includes_cases_for_deep_cleanup(self) -> None:
        ...

    def test_clean_workspace_removes_prof_and_extra_info(self) -> None:
        ...

    def test_clean_workspace_unlinks_symlink_artifact(self) -> None:
        ...

    def test_is_cleanable_workspace_detects_generated_only_workspace(self) -> None:
        ...
```

- [ ] **Step 2: Run the cleanup-module tests to verify they fail**

Run: `uv run python -m unittest tests.test_cleaning -v`

Expected: FAIL because `helix.cleaning` does not exist yet.

- [ ] **Step 3: Implement the cleanup module**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanupResult:
    workspace: Path
    removed: tuple[Path, ...]
    missing: tuple[Path, ...]


def is_cleanable_workspace(path: Path) -> bool: ...
def discover_clean_workspaces(root: Path) -> list[Path]: ...
def clean_workspace(workspace: Path, *, deep: bool) -> CleanupResult: ...
def clean_batch_root_artifacts(root: Path) -> CleanupResult: ...
```
```

- [ ] **Step 4: Re-run the cleanup-module tests**

Run: `uv run python -m unittest tests.test_cleaning -v`

Expected: PASS.

### Task 3: Wire the `clean` command into the CLI

**Files:**
- Create: `src/helix/commands/clean.py`
- Modify: `src/helix/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing handler tests**

```python
    def test_main_clean_single_workspace_preserves_cases_by_default(self) -> None:
        ...

    def test_main_clean_single_workspace_deep_removes_cases(self) -> None:
        ...

    def test_main_clean_batch_root_removes_batch_artifacts(self) -> None:
        ...

    def test_main_clean_batch_root_without_children_reports_status_style_error(self) -> None:
        ...
```

- [ ] **Step 2: Run the focused handler tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliMainTests.test_main_clean_single_workspace_preserves_cases_by_default tests.test_cli.CliMainTests.test_main_clean_single_workspace_deep_removes_cases tests.test_cli.CliMainTests.test_main_clean_batch_root_removes_batch_artifacts tests.test_cli.CliMainTests.test_main_clean_batch_root_without_children_reports_status_style_error -v`

Expected: FAIL because no `clean` handler or parser wiring exists yet.

- [ ] **Step 3: Implement the handler and parser wiring**

```python
def handle_clean(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    target = Path(args.input).expanduser().resolve()
    if not target.exists():
        parser.error(f"Input path does not exist: {target}")
    if not target.is_dir():
        parser.error(f"Input path is not a directory: {target}")
    ...
```

```python
from helix.commands.clean import handle_clean
...
CommandKind.CLEAN: _CommandSpec(
    handler=handle_clean,
    help_group="Status",
    help_summary="Remove known generated artifacts from one workspace or a batch root.",
    description="Remove known generated artifacts from one operator workspace or a batch root.",
)
```

```python
if command_kind == CommandKind.CLEAN:
    subparser.add_argument("--deep", action="store_true")
```

- [ ] **Step 4: Re-run the focused handler tests**

Run: `uv run python -m unittest tests.test_cli.CliMainTests.test_main_clean_single_workspace_preserves_cases_by_default tests.test_cli.CliMainTests.test_main_clean_single_workspace_deep_removes_cases tests.test_cli.CliMainTests.test_main_clean_batch_root_removes_batch_artifacts tests.test_cli.CliMainTests.test_main_clean_batch_root_without_children_reports_status_style_error -v`

Expected: PASS.

### Task 4: Document the new command

**Files:**
- Modify: `README.md`
- Test: none

- [ ] **Step 1: Add command-map and usage docs**

```markdown
- `clean`: remove known generated artifacts from one workspace or a batch root.
```

```markdown
## Clean Workspaces

Use `clean` when you want to remove known generated artifacts while keeping the original operator file.

```bash
uv run helix clean --input .
uv run helix clean --input operators_root
uv run helix clean --input . --deep
```
```

- [ ] **Step 2: Review the README section for consistency with the spec**

Run: `rg -n "clean|--deep" README.md`

Expected: shows the new command map and usage section with `--deep`.

### Task 5: Verify the end-to-end change set

**Files:**
- Modify: `docs/plans/2026-05-30-clean-subcommand.md`

- [ ] **Step 1: Run the cleanup-focused unit tests**

Run: `uv run python -m unittest tests.test_cleaning tests.test_cli -v`

Expected: PASS for the new cleanup tests and existing CLI coverage.

- [ ] **Step 2: Run the repository verification commands**

Run: `uv run --group dev ruff check`

Expected: PASS.

- [ ] **Step 3: Run type checking**

Run: `uv run pyright`

Expected: PASS.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS.
