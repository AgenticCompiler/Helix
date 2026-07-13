# Prepare Skills Temporary Git Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `prepare_skills()` initialize a temporary local git repo at the workspace root when `workdir/.git` is missing, then remove only that temporary `.git` during cleanup.

**Architecture:** Keep the behavior inside `SkillLinkManager` so every existing staging caller gets the same repository boundary semantics. Extend `SkillLinkSet` with temporary git metadata, create the boundary before skill copying, and roll it back on failure paths inside `prepare_skills()` itself.

**Tech Stack:** Python 3.12, `pathlib`, `subprocess`, `shutil`, `unittest`

---

### Task 1: Add failing temporary git boundary tests

**Files:**
- Modify: `tests/test_skills.py`
- Modify: `src/helix/skills.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_prepare_skills_creates_and_cleans_temporary_git_repo(self) -> None:
        ...

    def test_prepare_skills_preserves_existing_local_git_repo(self) -> None:
        ...

    def test_prepare_skills_creates_local_git_repo_even_under_parent_repo(self) -> None:
        ...

    def test_prepare_skills_rolls_back_temporary_git_repo_on_failure(self) -> None:
        ...
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_prepare_skills_creates_and_cleans_temporary_git_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_preserves_existing_local_git_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_creates_local_git_repo_even_under_parent_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_rolls_back_temporary_git_repo_on_failure -v`

Expected: FAIL because `SkillLinkManager` does not create, record, or clean up a temporary `.git`.

### Task 2: Implement temporary git boundary lifecycle in `SkillLinkManager`

**Files:**
- Modify: `src/helix/skills.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Add minimal metadata and helper methods**

```python
@dataclass
class SkillLinkSet:
    created_paths: List[Path]
    temporary_git_dir: Path | None = None
```

```python
def _ensure_local_git_boundary(self, workdir: Path) -> Path | None:
    git_path = workdir / ".git"
    if git_path.exists():
        return None
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workdir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return git_path
```

- [ ] **Step 2: Run the focused tests to verify they still fail for cleanup/reporting gaps**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_prepare_skills_creates_and_cleans_temporary_git_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_preserves_existing_local_git_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_creates_local_git_repo_even_under_parent_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_rolls_back_temporary_git_repo_on_failure -v`

Expected: still FAIL until `prepare_skills()` returns the metadata and `cleanup()` removes it.

- [ ] **Step 3: Wire creation, rollback, cleanup, and verbose descriptions**

```python
temporary_git_dir = self._ensure_local_git_boundary(workdir)
try:
    ...
    return SkillLinkSet(created, temporary_git_dir=temporary_git_dir)
except Exception:
    if temporary_git_dir is not None:
        shutil.rmtree(temporary_git_dir)
    raise
```

```python
if link_set.temporary_git_dir is not None:
    shutil.rmtree(link_set.temporary_git_dir)
```

- [ ] **Step 4: Re-run the focused tests**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_prepare_skills_creates_and_cleans_temporary_git_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_preserves_existing_local_git_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_creates_local_git_repo_even_under_parent_repo tests.test_skills.SkillLinkManagerTests.test_prepare_skills_rolls_back_temporary_git_repo_on_failure -v`

Expected: PASS.

### Task 3: Verify the complete change

**Files:**
- Modify: `docs/specs/2026-06-01-prepare-skills-temp-git-boundary-design.md`
- Modify: `docs/plans/2026-06-01-prepare-skills-temp-git-boundary.md`

- [ ] **Step 1: Run the full skill staging tests**

Run: `uv run python -m unittest tests.test_skills -v`

Expected: PASS.

- [ ] **Step 2: Run lint**

Run: `uv run --group dev ruff check`

Expected: PASS.

- [ ] **Step 3: Run type checking**

Run: `uv run pyright`

Expected: PASS.
