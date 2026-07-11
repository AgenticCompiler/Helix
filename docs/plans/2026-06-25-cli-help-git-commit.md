# CLI Help Git Commit Implementation Plan

> **For agent**: implement from top to bottom as bite-sized tasks, committing after each completed task.
> **Design spec**: `docs/specs/2026-06-25-cli-help-git-commit-design.md`

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/helix/build_info.py` | CREATE | Runtime resolver: source checkout detection, embedded metadata loading, display formatting |
| `src/helix/_setuptools_hooks.py` | CREATE | Setuptools `build_py` + `sdist` overrides that generate `_build_meta.json` |
| `pyproject.toml` | MODIFY | Register build command overrides via `[tool.setuptools.cmdclass]` |
| `packaging/helix.spec` | MODIFY | Include `_build_meta.json` in PyInstaller `a.datas` |
| `scripts/build-pyinstaller.py` | MODIFY | Generate `_build_meta.json` into source tree before invoking PyInstaller |
| `.gitignore` | MODIFY | Ignore generated `_build_meta.json` to prevent accidental commits |
| `src/helix/cli.py` | MODIFY | Add "Build info:" section to top-level epilog |
| `tests/test_build_info.py` | CREATE | Unit tests for resolver logic |
| `tests/test_cli.py` | MODIFY | Assert Build info section in help output |

## Tasks

### T1 — Create `build_info.py` module

File: `src/helix/build_info.py`

Functions:
- `_resolve_source_checkout_commit() -> str | None`: walk up from `__file__` looking for `.git` (dir or worktree file), then run `git rev-parse HEAD`. Return `None` on any failure.
- `_load_embedded_commit() -> str | None`: read `Path(__file__).with_name("_build_meta.json")`, parse JSON, return `data.get("git_commit")`. Return `None` on any failure (missing file, invalid JSON, missing key).
- `get_build_commit() -> str | None`: resolution order — source checkout → embedded → `None`. Memoize result with `functools.lru_cache` (process-local).
- `get_build_info_display() -> str`: return first 12 chars of full commit, or `"unknown"`.

### T2 — Unit tests for `build_info.py`

File: `tests/test_build_info.py`

Coverage:
- Source checkout resolution (mock `.git/HEAD` or `git rev-parse` output)
- Worktree indirection (`.git` file instead of directory)
- Embedded metadata resolution (valid JSON with `git_commit` key)
- Missing embedded metadata file
- Corrupted JSON
- Missing `git_commit` key in valid JSON
- `unknown` fallback (no `.git`, no metadata)
- Display shortening (40-char → 12-char, `None` → `"unknown"`)
- Memoization (same result on second call without re-reading)

### T3 — CLI integration

File: `src/helix/cli.py`

- Import `get_build_info_display` from `build_info`
- In `_build_top_level_epilog()`, prepend a "Build info:" section before "Command groups:":

```python
def _build_top_level_epilog() -> str:
    from helix.build_info import get_build_info_display
    lines = ["Build info:"]
    lines.append(f"  Git commit: {get_build_info_display()}")
    lines.append("")
    lines.append("Command groups:")
    # ... rest unchanged
```

### T4 — Extend CLI help tests

File: `tests/test_cli.py`

- In `test_top_level_help_groups_commands_and_examples`, add assertions:
  - `self.assertIn("Build info:", help_text)`
  - `self.assertIn("Git commit:", help_text)`

### T5 — Setuptools build command override

File: `src/helix/_setuptools_hooks.py`

- Define `BuildPyWithMeta` that overrides `setuptools.command.build_py.build_py`
- `run()`: resolve commit (env var → git fallback → "unknown"), write `_build_meta.json` to `build_lib/helix/`, then call `super().run()`
- Define `SdistWithMeta` that overrides `setuptools.command.sdist.sdist`
- `make_distribution()`: generate metadata into source tree before packaging (same resolve logic), call `super().make_distribution()`

File: `pyproject.toml`

```toml
[tool.setuptools.cmdclass]
build_py = "helix._setuptools_hooks:BuildPyWithMeta"
sdist = "helix._setuptools_hooks:SdistWithMeta"
```

### T6 — PyInstaller integration

File: `scripts/build-pyinstaller.py`

- Before calling `run_command(["pyinstaller", ...])`, run metadata generation:
  - Resolve commit (env var → git → "unknown")
  - Write `src/helix/_build_meta.json`
  - Add cleanup (optional, since .gitignored)

File: `packaging/helix.spec`

- After `collect_skills()`, add metadata tuple:
```python
meta_path = ROOT / "src" / "helix" / "_build_meta.json"
if meta_path.is_file():
    datas.append((str(meta_path), "helix"))
```

### T7 — `.gitignore`

Append `src/helix/_build_meta.json` to `.gitignore`.

### T8 — Full verification

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_build_info.py tests/test_cli.py`
- Manual: `uv run python -m helix.cli --help` shows Build info section
