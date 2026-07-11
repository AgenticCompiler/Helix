# Skill Staging Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh backend-staged skill directories on repeated launches so updated repository skills are recopied even when the backend skills root already exists.

**Architecture:** Keep the current backend-root staging flow and cleanup model intact. Narrow the overwrite boundary to the selected staged skill leaf: when a selected `.<backend>/skills/<skill>` directory already exists as a normal directory, remove just that directory and copy the source skill back in fresh before continuing.

**Tech Stack:** Python, pathlib, shutil, unittest, uv

---

### Task 1: Lock In Repeated-Launch Refresh Behavior

**Files:**
- Modify: `tests/test_skills.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Write the failing refresh test**

```python
    def test_repeated_selected_skill_staging_refreshes_existing_skill_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()

            skill_dir = source / "common" / "ascend-npu-gen-test"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("version one\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            first_links = manager.prepare_skills("claude", workspace, skill_names=("ascend-npu-gen-test",))

            staged_skill = self._skills_target(workspace, "claude") / "ascend-npu-gen-test"
            self.assertEqual((staged_skill / "SKILL.md").read_text(encoding="utf-8"), "version one\n")

            (skill_dir / "SKILL.md").write_text("version two\n", encoding="utf-8")
            second_links = manager.prepare_skills("claude", workspace, skill_names=("ascend-npu-gen-test",))

            self.assertEqual((staged_skill / "SKILL.md").read_text(encoding="utf-8"), "version two\n")

            manager.cleanup(second_links)
            manager.cleanup(first_links)
```

- [ ] **Step 2: Extend the test to prove unrelated sibling directories survive refresh**

```python
            (self._skills_target(workspace, "claude") / "user-skill").mkdir()

            second_links = manager.prepare_skills("claude", workspace, skill_names=("ascend-npu-gen-test",))

            self.assertTrue((self._skills_target(workspace, "claude") / "user-skill").exists())
```

- [ ] **Step 3: Run the focused test and verify it fails for the stale-copy bug**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_repeated_selected_skill_staging_refreshes_existing_skill_dir -v`

Expected: FAIL because the second `prepare_skills()` call leaves the staged `SKILL.md` at `"version one\n"` instead of refreshing it to `"version two\n"`.

### Task 2: Implement Leaf-Directory Refresh In Staging

**Files:**
- Modify: `src/helix/skills/staging.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Replace the skip-on-existing branch with leaf-directory refresh**

```python
        for staged_name, skill_dir in self._iter_selected_skill_dirs(skill_names, skill_sources):
            staged_path = target / staged_name
            if staged_path.exists():
                if staged_path.is_symlink():
                    raise RuntimeError(f"Skill path already exists as a symlink: {staged_path}")
                if not staged_path.is_dir():
                    raise RuntimeError(f"Skill path already exists and is not a directory: {staged_path}")
                self._remove_path(staged_path)
            shutil.copytree(skill_dir, staged_path, symlinks=False)
            created.append(staged_path)
```

- [ ] **Step 2: Keep the rest of `prepare_skills()` unchanged**

```python
            created.extend(self._copy_selected_skill_dirs(target, skill_names, skill_sources))
            if not root_pre_existed:
                created.insert(0, backend_root_path)
            return SkillLinkSet(created, temporary_git_dir=temporary_git_dir)
```

This preserves the current backend-root creation behavior, temporary git rollback, and cleanup reporting while changing only the selected-skill overwrite semantics.

- [ ] **Step 3: Re-run the focused test and verify it passes**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests.test_repeated_selected_skill_staging_refreshes_existing_skill_dir -v`

Expected: PASS

- [ ] **Step 4: Run the full skill staging test module**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests -v`

Expected: PASS

- [ ] **Step 5: Run repository verification for the touched area**

Run: `uv run --group dev ruff check src/helix/skills/staging.py tests/test_skills.py`

Expected: PASS
