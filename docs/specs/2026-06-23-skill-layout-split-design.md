# Skill Layout Split Design

## Goal

Restructure the repository skill source tree from one flat `skills/<skill-name>/` layout into two physical groups:

- `skills/common/`
- `skills/triton/`

This split should make backend-neutral Ascend NPU workflow skills reusable for future TileLang support while keeping Triton-specific skills isolated. The runtime-facing staged layout must remain flat, such as `.codex/skills/<skill-name>/`, so existing backend agents keep one-level skill discovery semantics.

In this repository, the `ascend-npu-` prefix means "shared across Ascend-platform workflows" rather than "shared across all accelerator backends". Both Triton Ascend and future TileLang Ascend flows target the same Ascend NPU platform, so `ascend-npu-*` is the shared namespace for backend-neutral skills within that Ascend-only scope. Supporting non-Ascend platforms is not a goal of this naming scheme.

This design covers:

- physical skill-directory relocation
- skill renaming for backend-neutral Ascend skills
- staged-skill flattening behavior
- CLI/runtime skill resolution updates
- skill-script path-resolution rules that work in both source and staged trees
- migration sequencing for an atomic hard-cut rename

This design does not introduce any backward-compatibility aliases for old skill names.

## User-Visible Semantics

- Source skills in this repository should no longer live directly under `skills/<skill-name>/`.
- The repository should instead store skill sources under `skills/common/<skill-name>/` or `skills/triton/<skill-name>/`.
- When the CLI stages skills into a target workspace, the staged layout must remain flat:
  - `.codex/skills/<skill-name>/`
  - `.opencode/skills/<skill-name>/`
  - similar layouts for other backends
- Users should interact only with the new skill names. The CLI, staged prompts, tests, and live documentation should stop referring to the old renamed `triton-npu-*` names for backend-neutral skills.
- No compatibility layer should translate old skill names to new names. A caller using an old renamed skill name should now fail explicitly.
- `triton-npu-optimize` remains a Triton skill and stays under `skills/triton/`.
- `torch-npu-optimize-knowledge` remains a Triton-side skill and stays under `skills/triton/`.
- `triton-npu-run-eval-mcp` moves together with `run-eval` into the common group and should be renamed consistently.
- Skill scripts must not hard-code `common/` or `triton/` into import paths, dynamic-load paths, or sibling-skill path probes. They must resolve other skills by logical skill name only, so the same code works from both the repository tree and flattened staged trees.

## Problem

The current repository assumes one physical invariant everywhere:

```text
skills/<skill-name>/
```

That assumption leaks into several layers:

- `src/triton_agent/skill_staging.py` stages skills by logical name with no notion of grouped source directories
- `src/triton_agent/skills.py` assumes the source tree is already flat and can sometimes copy the entire `skills/` root directly
- `src/triton_agent/skill_loader.py` assumes `repo_root()/skills/<skill-name>/scripts/...`
- multiple skill scripts dynamically load sibling skills by walking to a flat `skills/<skill-name>/scripts/` source path
- tests read live skill files through hard-coded flat paths

That coupling blocks the desired split:

- backend-neutral Ascend skills cannot be grouped separately from Triton-only skills
- future TileLang support would continue inheriting Triton-oriented names and path assumptions
- any naive move to `skills/common/` and `skills/triton/` would break staging, runtime loading, and cross-skill script resolution
- current "full-copy" staging behavior would incorrectly stage nested `common/` and `triton/` directories into workspaces instead of the required flat layout

The repository therefore needs one explicit logical skill catalog that separates:

- logical skill name
- physical source location
- staged destination name

## Skill Classification And Naming

### Common Skills

These skills move under `skills/common/` and are renamed from `triton-npu-*` to `ascend-npu-*`:

| Old logical name | New logical name | New physical source path |
| --- | --- | --- |
| `triton-npu-optimize-start-round` | `ascend-npu-optimize-start-round` | `skills/common/ascend-npu-optimize-start-round/` |
| `triton-npu-optimize-submit-baseline` | `ascend-npu-optimize-submit-baseline` | `skills/common/ascend-npu-optimize-submit-baseline/` |
| `triton-npu-optimize-submit-round` | `ascend-npu-optimize-submit-round` | `skills/common/ascend-npu-optimize-submit-round/` |
| `triton-npu-prepare-optimize-baseline` | `ascend-npu-prepare-optimize-baseline` | `skills/common/ascend-npu-prepare-optimize-baseline/` |
| `triton-npu-gen-test` | `ascend-npu-gen-test` | `skills/common/ascend-npu-gen-test/` |
| `triton-npu-gen-bench` | `ascend-npu-gen-bench` | `skills/common/ascend-npu-gen-bench/` |
| `triton-npu-gen-eval-suite` | `ascend-npu-gen-eval-suite` | `skills/common/ascend-npu-gen-eval-suite/` |
| `triton-npu-run-eval` | `ascend-npu-run-eval` | `skills/common/ascend-npu-run-eval/` |
| `triton-npu-run-eval-mcp` | `ascend-npu-run-eval-mcp` | `skills/common/ascend-npu-run-eval-mcp/` |
| `triton-npu-report` | `ascend-npu-report` | `skills/common/ascend-npu-report/` |
| `triton-npu-kernel-bench-logs` | superseded by `ascend-npu-distill-patterns` | `skills/common/ascend-npu-distill-patterns/` |
| `triton-npu-profile-operator` | `ascend-npu-profile-operator` | `skills/common/ascend-npu-profile-operator/` |
| `triton-npu-analyze-ir` | `ascend-npu-analyze-ir` | `skills/common/ascend-npu-analyze-ir/` |
| `triton-npu-analyze-round-performance` | `ascend-npu-analyze-round-performance` | `skills/common/ascend-npu-analyze-round-performance/` |
| `triton-npu-analyze-commit-perf` | `ascend-npu-analyze-commit-perf` | `skills/common/ascend-npu-analyze-commit-perf/` |

`ascend-npu-optimize-submit-baseline`, `ascend-npu-optimize-submit-round`, and `ascend-npu-optimize-start-round` remain allowed to reference the Triton optimize skill by logical skill name when they need shared optimize workflow helpers. The key constraint is that their script logic must not encode the physical group names `common/` or `triton/`.

`triton-npu-analyze-commit-perf` is currently not part of any `CommandKind` staging rule, but it is a real repository skill and is referenced directly by `src/triton_agent/distill/git_repo_workspaces.py`. It must still be cataloged, renamed, and moved under `skills/common/` so it does not remain as an uncategorized flat-root orphan after the split.

### Triton Skills

These skills move under `skills/triton/` and keep their current logical names:

| Logical name | New physical source path |
| --- | --- |
| `triton-npu-optimize` | `skills/triton/triton-npu-optimize/` |
| `triton-npu-convert-pytorch-operator` | `skills/triton/triton-npu-convert-pytorch-operator/` |
| `triton-npu-repair-guide` | `skills/triton/triton-npu-repair-guide/` |
| `triton-npu-analyze-compiler-source` | `skills/triton/triton-npu-analyze-compiler-source/` |
| `triton-npu-cann-ext-api-patterns` | `skills/triton/triton-npu-cann-ext-api-patterns/` |
| `triton-npu-optimize-knowledge` | `skills/triton/triton-npu-optimize-knowledge/` |
| `triton-npu-optimize-knowledge-v2` | `skills/triton/triton-npu-optimize-knowledge-v2/` |
| `triton-npu-optimize-knowledge-v3` | `skills/triton/triton-npu-optimize-knowledge-v3/` |
| `torch-npu-optimize-knowledge` | `skills/triton/torch-npu-optimize-knowledge/` |

`triton-npu-optimize` intentionally stays Triton-scoped in this iteration even though parts of its workflow are backend-neutral. That placement keeps the top-level optimize orchestration explicitly Triton-owned while common validation, execution, reporting, and profiling skills become reusable building blocks.

## Design

### Central Skill Catalog

Add one central catalog in `src/triton_agent/` that becomes the source of truth for repository-owned skills.

Each catalog entry should define at least:

- logical skill name
- source group, either `common` or `triton`
- physical source directory relative to repository root

The catalog should enforce repository-wide uniqueness of logical skill names. No two physical source directories may claim the same logical skill name, even across different groups.

The catalog should be used by:

- `skill_staging.py`
- `skills.py`
- `skill_loader.py`
- tests that need to locate live skill files
- any runtime helper that currently assumes `skills/<skill-name>/...`

This catalog should replace implicit discovery from `skills_root().iterdir()` for repository-owned skill staging and loading.

The catalog must not include compatibility aliases for old renamed skill names.

Tests should assert that every repository-owned skill directory appears exactly once in the catalog and that every logical skill name is unique.

### Physical Skills Root

`src/triton_agent/resources.py` should continue exposing the repository `skills/` root:

```text
<repo>/skills
```

But callers must stop assuming that direct children of that root are logical skills. After this change, direct children include grouping directories such as:

- `skills/common/`
- `skills/triton/`

The repository should treat `skills/` as a namespace container, not as the flat set of live skill directories.

### Staging Must Always Flatten

The staged workspace layout is a stable contract and must remain flat regardless of how the repository stores skill sources.

Therefore, `SkillLinkManager.prepare_skills(...)` must stop using raw whole-root copy semantics for repository skills. In particular:

- when staging a selected list of skills, the manager should resolve each logical skill through the catalog and copy that skill's source directory into a flat staged target named by the logical skill name
- when staging "all repository skills", the manager should iterate catalog entries and flatten them into the target one by one
- the staged destination name should always be the logical skill name, not the physical source path name with `common/` or `triton/`

Example:

- repository source: `skills/common/ascend-npu-run-eval/`
- staged destination: `.codex/skills/ascend-npu-run-eval/`

This rule also applies to source overrides such as the MCP replacement for run-eval. The override should swap one logical skill's source directory while preserving the same staged logical skill name.

This change must be applied to every `skills.py` path that currently assumes `self.skills_root` is already a flat set of skill directories. In practice that means replacing catalog-blind behavior in:

- `_iter_skill_dirs()`
- `_iter_selected_skill_dirs()`
- `_copy_selected_skill_dirs()`
- the `copy_root_when_missing and skill_names is None` branch that currently runs `shutil.copytree(self.skills_root, target, symlinks=False)`

After the split, none of those paths may copy or enumerate `skills/common/` and `skills/triton/` as if they were staged skills.

### Stage Rules And Command Ownership

`src/triton_agent/skill_staging.py` and any related command-to-skill mapping should be updated to the new logical names.

High-level command ownership after the rename should be:

- generation commands use `ascend-npu-gen-test`, `ascend-npu-gen-bench`, `ascend-npu-gen-eval-suite`, `ascend-npu-run-eval`, and `triton-npu-repair-guide`
- report uses `ascend-npu-report`
- log-check uses `triton-npu-optimize-knowledge`, `ascend-npu-optimize-submit-baseline`, and `ascend-npu-optimize-submit-round`
- optimize uses:
  - `triton-npu-optimize`
  - `triton-npu-optimize-knowledge`
  - `ascend-npu-prepare-optimize-baseline`
  - `ascend-npu-gen-test`
  - `ascend-npu-gen-bench`
  - `ascend-npu-run-eval`
  - `ascend-npu-optimize-submit-baseline`
  - `ascend-npu-optimize-submit-round`
  - `ascend-npu-optimize-start-round`
  - `ascend-npu-profile-operator`
  - `ascend-npu-analyze-round-performance`
  - `ascend-npu-analyze-ir`
  - `triton-npu-analyze-compiler-source`
  - `triton-npu-repair-guide`

The optimize-knowledge version switching logic should remain Triton-owned for now because the only versioned knowledge libraries in scope remain under `skills/triton/`.

Today that version-switching path lives in:

- `src/triton_agent/skill_staging.py::_resolve_skill_sources()`
- `src/triton_agent/optimize/pattern_reminders.py::resolve_generic_optimize_knowledge_skill_name()`

Those modules should keep owning Triton optimize-knowledge version selection in this iteration.

The MCP override path also needs an explicit rename at the source-override site. The current mapping:

```python
sources["triton-npu-run-eval"] = "triton-npu-run-eval-mcp"
```

in `src/triton_agent/skill_staging.py::_resolve_skill_sources()` must become:

```python
sources["ascend-npu-run-eval"] = "ascend-npu-run-eval-mcp"
```

while preserving the same "logical staged name, alternate source directory" semantics.

`ascend-npu-analyze-commit-perf` remains outside `CommandKind` stage rules in this iteration because no current user command stages it into agent workspaces. Its source lookup continues to be owned by the distill workflow, but that lookup must still move to the grouped-source catalog contract.

### Runtime Skill Loading

`src/triton_agent/skill_loader.py` should load skill scripts by logical skill name through the central catalog rather than directly concatenating `repo_root() / "skills" / skill_name`.

Required consequences:

- `skill_script_root(skill_name)` should resolve the physical skill source directory from the catalog
- `skill_script_path(skill_name, script_name)` should continue locating `scripts/<script_name>.py` under that resolved root
- `operator_eval_skill_root()` and `load_operator_eval_script_module()` should switch from `triton-npu-run-eval` to `ascend-npu-run-eval`
- runtime bridge calls such as `load_skill_script_module("...", "...")` should use the new logical names where applicable

### Skill-Script Cross-Skill Resolution Rules

Some skill scripts load helpers from other skills by walking the repository tree relative to `__file__`. That behavior should remain supported, but the resolution contract must become logical-name-based rather than flat-path-based.

Skill scripts must follow these rules:

1. They may locate another skill only by logical skill name.
2. They must not encode `common/` or `triton/` in the lookup path.
3. They must work from both:
   - the repository source tree
   - a flattened staged backend tree

The resolution algorithm should therefore be:

1. Start from `Path(__file__).resolve()`.
2. Walk upward until reaching the nearest ancestor named `skills`.
3. From that `skills` root, first check the flattened staged form `skills/<logical-skill-name>/` and accept it when `SKILL.md` exists there.
4. If the flattened staged form is absent, inspect only the direct child directories of `skills/` as source groups and collect candidates at `<group>/<logical-skill-name>/` when `SKILL.md` exists there.
5. Require exactly one grouped-source candidate.
6. Fail explicitly when zero candidates or more than one candidate are found.
7. Use the resolved logical skill directory for dynamic loading.

This keeps one script implementation valid in both environments:

- source-tree physical layout: `skills/common/<logical-skill-name>/` or `skills/triton/<logical-skill-name>/`
- staged layout: `skills/<logical-skill-name>/`

This algorithm deliberately avoids arbitrary-depth recursive scans. Source resolution remains bounded to one flat staged check plus one scan across the direct source-group directories, so startup cost stays predictable and the match contract stays deterministic.

The repository may implement this resolution as small skill-local helpers where needed, but it should not introduce source-path assumptions that mention `common/` or `triton/`.

### Skill Markdown Reference Rule

Each `SKILL.md` should only reference script paths within the same logical skill directory.

That means skill docs should prefer references like:

- `scripts/run-command.py`
- `scripts/optimize_submit_round.py`

and avoid cross-skill script path references such as:

- `../other-skill/scripts/...`

Skill-to-skill workflow references should use logical skill names instead of file paths. This keeps the skill contract independent of grouped source layout and consistent with staged flattening.

As of this review, the live documentation audit finds four `SKILL.md` files with cross-skill `../.../scripts/...` references that must be rewritten during the migration:

- `skills/triton-npu-gen-eval-suite/SKILL.md`
- `skills/triton-npu-analyze-round-performance/SKILL.md`
- `skills/triton-npu-profile-operator/SKILL.md`
- `skills/triton/triton-npu-optimize/SKILL.md`

There is also at least one current live reference document with the same pattern:

- `skills/triton/triton-npu-optimize/references/artifacts.md`

Those files should be updated as part of the rename-and-layout migration rather than left for a later doc-only cleanup.

## Triton-Owned Live Workflow References

`triton-npu-optimize` remains under `skills/triton/`, but it should update its live workflow references to the new common skill names it depends on:

- `ascend-npu-prepare-optimize-baseline`
- `ascend-npu-gen-test`
- `ascend-npu-gen-bench`
- `ascend-npu-run-eval`
- `ascend-npu-optimize-submit-baseline`
- `ascend-npu-optimize-submit-round`
- `ascend-npu-optimize-start-round`
- `ascend-npu-profile-operator`
- `ascend-npu-analyze-round-performance`
- `ascend-npu-analyze-ir`

Its top-level workflow remains Triton-owned in this iteration.

`ascend-npu-analyze-round-performance` may continue pointing to `triton-npu-optimize-knowledge` for current generic symptom or pattern references in this iteration. Splitting generic optimize knowledge into a backend-neutral library is out of scope for this change.

## Migration Sequencing And Rollback

This migration is a hard cut and should be treated as one atomic rename-and-layout change series, not as a partially shippable incremental rollout with runtime aliases.

Recommended branch implementation order:

1. Introduce the central skill catalog plus grouped-source-aware staging and loading helpers.
2. Update `skills.py`, `skill_loader.py`, and other runtime callers so the code can resolve logical skills from grouped repository sources while still producing flat staged copies.
3. Physically move and rename skill directories into `skills/common/` and `skills/triton/`.
4. Update all source references, tests, and live docs to the new logical names and grouped source paths.
5. Verify the final tree only after all of the above land together.

Current implementation blast radius is already large enough that this should be planned as one repository-wide migration batch:

- 23 `src/triton_agent/*.py` files currently contain skill-name or flat-skill-path assumptions
- 39 `tests/test_*.py` files currently contain repository skill-name or flat-skill-path assumptions

Merge and rollback rules:

- the final merged tree must be atomic because no old-name alias layer exists
- partial cherry-picks of the rename are unsupported
- rollback should revert the entire migration changeset rather than trying to restore selected old names by hand

## Follow-On TileLang Behavior Work

This layout-split design intentionally does not bundle TileLang functional behavior changes into the same implementation batch.

The grouped layout exists partly to prepare for later TileLang support, but behavior changes such as:

- adding `tilelang-wrapper`
- widening kernel discovery beyond Triton
- changing round continuity detection signals

should be specified and implemented in a follow-on TileLang support design after the layout and rename migration is stable.

The follow-on TileLang design should revisit at least:

- `ascend-npu-gen-test`
- `ascend-npu-gen-bench`
- `ascend-npu-gen-eval-suite`
- `ascend-npu-run-eval`
- `ascend-npu-optimize-submit-round`
- `ascend-npu-prepare-optimize-baseline`

## Tests And Documentation

### Live Tests

Tests under `tests/` should be updated to use the new logical names and grouped physical layout.

This includes at least:

- staging tests
- skill-loader tests
- runtime command mapping tests
- optimize contract tests
- skill-document contract tests
- any test that reads live skill files through hard-coded paths

New or strengthened coverage should verify:

- repository physical source paths now live under `skills/common/` or `skills/triton/`
- staged skill trees remain flat
- full-copy staging no longer reproduces the grouped repository source tree
- skill-loader resolution succeeds for renamed common skills
- skill-side logical-name lookup works from both grouped source and flattened staged copies

Current blast-radius summary for live tests:

- 39 test files currently reference repository skill names or flat skill paths
- those files fall into five rough buckets:
  - CLI, backend, and command behavior: 11 files
  - skill staging, loading, and contract behavior: 6 files
  - optimize workflow behavior: 8 files
  - feature/runtime helpers such as IR and remote execution: 4 files
  - other trace, hook, profiler, and doc-contract coverage: 10 files

At minimum, the following file groups should be expected to change together:

- CLI, backend, and command behavior:
  - `tests/test_backends_base.py`
  - `tests/test_bench_runner.py`
  - `tests/test_claude_runner.py`
  - `tests/test_cli.py`
  - `tests/test_codex_runner.py`
  - `tests/test_convert_commands.py`
  - `tests/test_generation_commands.py`
  - `tests/test_opencode_runner.py`
  - `tests/test_openhands_runner.py`
  - `tests/test_pi_runner.py`
  - `tests/test_traecli_runner.py`
- skill staging, loading, and contract behavior:
  - `tests/test_distill_knowledge_workspace.py`
  - `tests/test_distill_workflow.py`
  - `tests/test_run_skill_loader.py`
  - `tests/test_skill_command_script.py`
  - `tests/test_skill_staging.py`
  - `tests/test_skills.py`
- optimize workflow behavior:
  - `tests/test_optimize_baseline.py`
  - `tests/test_optimize_checks.py`
  - `tests/test_optimize_contract.py`
  - `tests/test_optimize_guidance.py`
  - `tests/test_optimize_pattern_tools.py`
  - `tests/test_optimize_round_contract.py`
  - `tests/test_optimize_runtime.py`
  - `tests/test_optimize_workflow_state.py`
- feature/runtime helpers:
  - `tests/test_ascend_operator_ir_analyzer.py`
  - `tests/test_inspect_compiler_source.py`
  - `tests/test_inspect_ir.py`
  - `tests/test_remote_execution.py`
- other trace, hook, profiler, and doc-contract coverage:
  - `tests/test_ascend_npu_operator_profiler.py`
  - `tests/test_codex_pretooluse_guard.py`
  - `tests/test_codex_trace.py`
  - `tests/test_generation_contracts.py`
  - `tests/test_log_check_launcher.py`
  - `tests/test_models.py`
  - `tests/test_msprof_parse_bin.py`
  - `tests/test_opencode_hook_guard.py`
  - `tests/test_subagents.py`
  - `tests/test_trace_analyze_analyzer.py`

### Live Documentation

The repository's current user-facing documentation should be updated to the new skill names and live source paths where applicable. This includes:

- `README.md`
- current `SKILL.md` files
- current `references/*.md` files
- tests that enforce documentation contracts

### Historical Documents

Historical design and review snapshots under:

- `docs/plans/`
- `docs/reviews/`

should not be mass-rewritten in this iteration unless a specific file is still treated as live contract surface by tests or current user-facing docs.

This keeps historical documents readable as snapshots instead of rewriting old plans to describe post-migration names they were never written against.

## Compatibility Policy

This migration is a hard cut.

The repository should not provide:

- old-to-new skill-name aliases
- old physical directory symlinks or duplicates
- runtime fallback lookup from old logical names to new ones

All repository-owned callers should move to the new names in the same implementation series as the directory split.

## Non-Goals

- making `triton-npu-optimize` itself backend-neutral in this iteration
- bundling TileLang functional behavior changes into the same implementation batch as the layout split
- splitting or renaming `triton-npu-optimize-knowledge`, `triton-npu-optimize-knowledge-v2`, `triton-npu-optimize-knowledge-v3`, or `torch-npu-optimize-knowledge`
- creating a full TileLang-specific optimize top-level skill set
- introducing a general backend-plugin architecture for skills
- rewriting historical design/review documents purely for naming consistency
