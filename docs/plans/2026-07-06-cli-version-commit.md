# CLI Version Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add top-level `helix -v` and `helix --version` support that prints only the build commit display value and exits successfully.

**Architecture:** Keep the CLI thin by wiring the standard argparse version action into the existing root parser. Reuse `helix.build_info.get_build_info_display()` so version output matches the build-info value already shown in top-level help.

**Tech Stack:** Python 3.9, argparse, pytest, unittest.mock

---

### Task 1: Add failing CLI tests for version flags

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_main_prints_build_commit_for_long_version_flag(self) -> None:
        stdout = StringIO()
        with patch("helix.cli.get_build_info_display", return_value="deadbeefcafe"):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exc:
                    main(["--version"])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "deadbeefcafe\n")

    def test_main_prints_build_commit_for_short_version_flag(self) -> None:
        stdout = StringIO()
        with patch("helix.cli.get_build_info_display", return_value="deadbeefcafe"):
            with redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as exc:
                    main(["-v"])
        self.assertEqual(exc.exception.code, 0)
        self.assertEqual(stdout.getvalue(), "deadbeefcafe\n")
```

- [ ] **Step 2: Run the CLI version tests to verify they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k version_flag`

Expected: FAIL because the root parser does not yet accept `-v` or `--version`.

- [ ] **Step 3: Update help-style fixtures that depend on root usage text**

```python
        text = "usage: helix [-h] [-v] COMMAND ..."
```

Apply this exact usage string update to the root-parser fixture strings in `tests/test_help_style.py`.

- [ ] **Step 4: Run the focused help-style tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_help_style.py`

Expected: PASS if only the fixture strings changed correctly.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py tests/test_help_style.py
git commit -m "test: cover top-level version flags"
```

### Task 2: Add root parser version action

**Files:**
- Modify: `src/helix/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add the standard argparse version option to the root parser**

```python
    parser = TritonArgumentParser(
        prog="helix",
        usage="helix [-h] [-v] COMMAND ...",
        description=_TOP_LEVEL_DESCRIPTION,
        epilog=_build_top_level_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        env_var_names=_collect_env_var_names(),
        command_names=_collect_command_names(),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_build_info_display(),
    )
```

- [ ] **Step 2: Run the version-flag tests to verify they pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k version_flag`

Expected: PASS with both tests printing the mocked commit and exiting with status `0`.

- [ ] **Step 3: Run the help-style tests to verify the new root usage is stable**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_help_style.py`

Expected: PASS with the updated `usage: helix [-h] [-v] COMMAND ...` fixtures.

- [ ] **Step 4: Run the broader CLI test module**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/helix/cli.py tests/test_cli.py tests/test_help_style.py
git commit -m "feat: add top-level version flag"
```

### Task 3: Project verification

**Files:**
- Modify: `docs/specs/2026-07-06-cli-version-commit-design.md`
- Modify: `docs/plans/2026-07-06-cli-version-commit.md`

- [ ] **Step 1: Run Ruff**

Run: `uv run --group dev ruff check`

Expected: PASS with no lint errors.

- [ ] **Step 2: Run Pyright**

Run: `uv run pyright`

Expected: PASS with no type errors.

- [ ] **Step 3: Run focused pytest coverage for this change**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py tests/test_help_style.py tests/test_build_info.py`

Expected: PASS.

- [ ] **Step 4: Run a manual CLI verification**

Run: `uv run python -m helix.cli --version`

Expected: prints one line containing the current build commit display value or `unknown`.

- [ ] **Step 5: Commit**

```bash
git add docs/specs/2026-07-06-cli-version-commit-design.md docs/plans/2026-07-06-cli-version-commit.md src/helix/cli.py tests/test_cli.py tests/test_help_style.py
git commit -m "docs: record cli version flag plan"
```
