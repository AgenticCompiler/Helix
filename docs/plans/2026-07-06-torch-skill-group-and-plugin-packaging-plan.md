# Torch Skill Group And Claude Plugin Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `torch-npu-optimize-knowledge` into `skills/torch/`, keep main CLI operator-target staging unchanged, and make the Claude plugin builder a fixed Triton package that never bundles that skill.

**Architecture:** Keep the logical skill name unchanged and update only the physical ownership path in the repository catalog and path-based references. For the Claude plugin builder, stop exposing `optimize_target` and derive a fixed optimize payload by filtering the shared staging result down to the plugin-supported Triton bundle.

**Tech Stack:** Python, unittest, repository skill catalog/staging helpers, shell index-update script.

---

### Task 1: Lock in the new physical skill path contract

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `src/triton_agent/skills/catalog.py`
- Modify: `scripts/update-optimize-knowledge-indices.sh`
- Modify: `.codex/skills/create-optimize-pattern/SKILL.md`
- Create: `skills/torch/torch-npu-optimize-knowledge/`
- Delete: `skills/triton/torch-npu-optimize-knowledge/`

- [ ] **Step 1: Write the failing path-contract tests**

Add or update assertions in `tests/test_generation_contracts.py` so the Torch NPU knowledge skill is read from `skills/torch/torch-npu-optimize-knowledge/...` instead of `skills/triton/...`, and so path-based helper docs/scripts point to the new location.

```python
    def test_torch_npu_optimize_knowledge_skill_owns_operator_level_pattern_references(
        self,
    ) -> None:
        knowledge = _read("skills/torch/torch-npu-optimize-knowledge/SKILL.md")
        pattern_index = _read("skills/torch/torch-npu-optimize-knowledge/references/pattern_index.md")
        pattern = _read(
            "skills/torch/torch-npu-optimize-knowledge/references/patterns/argsort-avoid-aicpu-fallback.md"
        )
```

```python
    def test_agents_declares_knowledge_skill_as_generic_pattern_and_symptom_source(
        self,
    ) -> None:
        skill = _read(".codex/skills/create-optimize-pattern/SKILL.md")
        self.assertIn(
            "skills/torch/torch-npu-optimize-knowledge/references/patterns/",
            skill,
        )
```

```python
    def test_skill_points_to_shared_optimize_knowledge_index_update_script(self) -> None:
        script = _read("scripts/update-optimize-knowledge-indices.sh")
        self.assertIn(
            "skills/torch/torch-npu-optimize-knowledge/references/pattern_index.md",
            script,
        )
```

- [ ] **Step 2: Run the failing contract tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py -k "torch_npu_optimize_knowledge or create_optimize_pattern or index_update_script"`

Expected: FAIL because the repository still uses `skills/triton/torch-npu-optimize-knowledge/...`.

- [ ] **Step 3: Move the skill and update the physical catalog path**

Move the directory from `skills/triton/torch-npu-optimize-knowledge/` to `skills/torch/torch-npu-optimize-knowledge/`, then update `src/triton_agent/skills/catalog.py` so the logical skill still resolves but now points at the Torch group.

Use a catalog entry shaped like:

```python
_TORCH_SKILLS: tuple[SkillCatalogEntry, ...] = (
    SkillCatalogEntry(
        logical_name="torch-npu-optimize-knowledge",
        source_group="torch",
        physical_path="skills/torch/torch-npu-optimize-knowledge",
    ),
)

SKILL_CATALOG: tuple[SkillCatalogEntry, ...] = (
    _COMMON_SKILLS + _TRITON_SKILLS + _TORCH_SKILLS + _TILELANG_SKILLS
)
```

Keep `src/triton_agent/skills/selection.py` unchanged in this task.

- [ ] **Step 4: Update path-based repository references**

Update:

- `scripts/update-optimize-knowledge-indices.sh`
- `.codex/skills/create-optimize-pattern/SKILL.md`

with the new path:

```bash
uv run python -m triton_agent.optimize_knowledge.pattern_index \
  --patterns-dir skills/torch/torch-npu-optimize-knowledge/references/patterns \
  --output skills/torch/torch-npu-optimize-knowledge/references/pattern_index.md \
  --style default
```

and:

```md
| Torch NPU optimize pattern | `torch-npu-optimize-knowledge` | `skills/torch/torch-npu-optimize-knowledge/references/patterns/` |
```

- [ ] **Step 5: Run the contract tests again**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py -k "torch_npu_optimize_knowledge or create_optimize_pattern or index_update_script"`

Expected: PASS.

- [ ] **Step 6: Regenerate the Torch pattern index and verify the script still works**

Run: `bash scripts/update-optimize-knowledge-indices.sh`

Expected: command exits `0` and prints `Update optimize knowledge indices done.`

- [ ] **Step 7: Commit**

```bash
git add tests/test_generation_contracts.py src/triton_agent/skills/catalog.py scripts/update-optimize-knowledge-indices.sh .codex/skills/create-optimize-pattern/SKILL.md skills/torch/torch-npu-optimize-knowledge
git commit -m "refact: move torch optimize knowledge into torch group"
```

### Task 2: Keep main CLI staging behavior unchanged while proving the moved skill still resolves

**Files:**
- Modify: `tests/test_skill_staging.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/triton_agent/skills/catalog.py`

- [ ] **Step 1: Add a regression test that the moved skill still resolves through the catalog**

Add a focused test in `tests/test_skill_staging.py` that proves the logical skill name still stages for operator-target optimize and still does not stage for kernel-target optimize.

```python
    def test_resolve_staged_skills_for_optimize_operator_target_includes_torch_npu_knowledge(
        self,
    ) -> None:
        names, _ = resolve_staged_skills(
            CommandKind.OPTIMIZE,
            optimize_target="operator",
        )

        self.assertIn("torch-npu-optimize-knowledge", names or ())
```

Keep the existing kernel-target omission test as the paired guard.

- [ ] **Step 2: Add a runtime regression assertion**

Keep or tighten the existing assertion in `tests/test_optimize_runtime.py` so `build_optimize_request(...)` still includes `torch-npu-optimize-knowledge` for `optimize_target="operator"`.

```python
            self.assertIn(
                "torch-npu-optimize-knowledge",
                request.staged_skill_names or (),
            )
```

- [ ] **Step 3: Run the staging and runtime tests**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_staging.py tests/test_optimize_runtime.py -k "torch_npu or optimize_target"`

Expected: PASS. This confirms the physical move did not change main CLI behavior.

- [ ] **Step 4: Commit**

```bash
git add tests/test_skill_staging.py tests/test_optimize_runtime.py src/triton_agent/skills/catalog.py
git commit -m "test: preserve operator staging for torch optimize knowledge"
```

### Task 3: Remove hidden optimize-target support from the Claude plugin builder

**Files:**
- Modify: `tests/test_claude_optimize_plugin.py`
- Modify: `scripts/build-claude-optimize-plugin.py`

- [ ] **Step 1: Write failing plugin-builder tests for the fixed package contract**

Update `tests/test_claude_optimize_plugin.py` so the builder contract no longer compares directly against `resolve_staged_skills(CommandKind.OPTIMIZE)`; instead, it should expect the plugin optimize payload to exclude `torch-npu-optimize-knowledge` and the builder API to expose no `optimize_target`.

Add assertions like:

```python
        assets = build_claude_optimize_plugin_assets()

        self.assertNotIn("torch-npu-optimize-knowledge", assets.optimize_skill_names)
        self.assertNotIn("torch-npu-optimize-knowledge", assets.skill_names)
```

and in the built tree test:

```python
            self.assertFalse((built_dir / "skills" / "torch-npu-optimize-knowledge").exists())
```

Also add a lightweight signature check:

```python
        self.assertNotIn(
            "optimize_target",
            build_claude_optimize_plugin_assets.__code__.co_varnames[: build_claude_optimize_plugin_assets.__code__.co_argcount],
        )
```

- [ ] **Step 2: Run the plugin-builder tests to confirm they fail**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`

Expected: FAIL because the builder still exposes `optimize_target` and still packages the unfiltered optimize staging result.

- [ ] **Step 3: Implement the fixed optimize payload in the builder**

In `scripts/build-claude-optimize-plugin.py`:

- remove `optimize_target` from both public builder functions
- keep using `resolve_staged_skills(CommandKind.OPTIMIZE, language=language, enable_cann_ext_api=enable_cann_ext_api)`
- filter the optimize skill names before packaging

Use a helper like:

```python
def _select_plugin_optimize_skill_names(skill_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        skill_name
        for skill_name in skill_names
        if skill_name != "torch-npu-optimize-knowledge"
    )
```

Apply the same filtered list when rendering the optimize agent and when computing the final `skill_names` union.

Do not change the shared CLI staging helper and do not add plugin-only rename logic anywhere else.

- [ ] **Step 4: Run the plugin-builder tests again**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_claude_optimize_plugin.py scripts/build-claude-optimize-plugin.py
git commit -m "refact: fix claude plugin optimize skill packaging"
```

### Task 4: Run the full targeted verification set and reconcile any leftover path assumptions

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_claude_optimize_plugin.py`
- Modify: `docs/specs/2026-06-23-skill-layout-split-design.md`
- Modify: any other file from the targeted test failures only if it still hard-codes the old physical path

- [ ] **Step 1: Run the full targeted suite from the spec**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_staging.py tests/test_claude_optimize_plugin.py tests/test_generation_contracts.py`

Expected: PASS.

- [ ] **Step 2: If the suite exposes additional old-path references, fix only those exact references**

Likely candidates include historical design docs or remaining contract assertions that still mention:

```text
skills/triton/torch-npu-optimize-knowledge/
```

Only patch files that fail the targeted suite or are directly needed to restore the documented contract for this change.

- [ ] **Step 3: Re-run the targeted suite**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_staging.py tests/test_claude_optimize_plugin.py tests/test_generation_contracts.py`

Expected: PASS with no skipped repairs outstanding.

- [ ] **Step 4: Commit**

```bash
git add tests/test_skill_staging.py tests/test_claude_optimize_plugin.py tests/test_generation_contracts.py docs/specs/2026-06-23-skill-layout-split-design.md
git commit -m "test: align contracts for torch skill group split"
```
