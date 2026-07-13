# Torch Skill Group And Claude Plugin Packaging Design

## Goal

Move the repository-owned `torch-npu-optimize-knowledge` skill out of the Triton source group into a new `skills/torch/` group, while keeping its logical skill name and main CLI staging behavior unchanged.

At the same time, remove the hidden `optimize_target` dimension from the Claude plugin builder so the generated `triton-optimizer` plugin remains a fixed Triton workflow package and never bundles `torch-npu-optimize-knowledge`.

## User-Visible Semantics

- The repository source path for `torch-npu-optimize-knowledge` moves from `skills/triton/torch-npu-optimize-knowledge/` to `skills/torch/torch-npu-optimize-knowledge/`.
- The logical skill name remains `torch-npu-optimize-knowledge`.
- Main CLI optimize behavior does not change:
  - `--optimize-target kernel` still does not stage `torch-npu-optimize-knowledge`.
  - `--optimize-target operator` still stages `torch-npu-optimize-knowledge`.
- Building the Claude plugin still produces the same Triton-focused plugin shape as today, but the builder contract becomes explicit: the plugin does not have an optimize-target variant and must not package `torch-npu-optimize-knowledge`.
- The plugin builder Python API no longer accepts `optimize_target`.

## Problem

Today `torch-npu-optimize-knowledge` is physically stored under `skills/triton/` even though its scope is Torch NPU and operator-level optimization guidance rather than Triton-kernel ownership.

That mismatch leaks into adjacent tooling:

- the skill catalog classifies it as a Triton-group source directory
- index-update scripts and helper skills point at a Triton physical path
- tests encode the current Triton physical location

Separately, the Claude plugin builder still exposes an `optimize_target` parameter in its Python API even though the shipped plugin does not expose optimize-target selection. That hidden parameter can produce an unsupported packaging variant that includes `torch-npu-optimize-knowledge`, which is not a real product mode for the plugin.

## Non-Goals

- Do not rename the logical skill `torch-npu-optimize-knowledge`.
- Do not change `helix optimize --optimize-target ...` semantics.
- Do not remove operator-target staging of `torch-npu-optimize-knowledge` from the main CLI.
- Do not add compatibility aliases or duplicate copies under both `skills/triton/` and `skills/torch/`.
- Do not redesign unrelated plugin-builder options.

## Design

### Physical Skill Ownership

Create a new physical source group under `skills/torch/` and move the existing skill directory to:

- `skills/torch/torch-npu-optimize-knowledge/`

The skill remains a normal repository-owned catalog skill. Only its physical source-group ownership changes.

The central skill catalog should continue resolving the logical name `torch-npu-optimize-knowledge`, but its `physical_path` should point to `skills/torch/torch-npu-optimize-knowledge`. The catalog structure should grow a Torch-specific section or equivalent grouping rather than continuing to classify this skill as part of the Triton physical group.

### Main CLI Staging Stays Target-Aware

`src/helix/skills/selection.py` should keep the current optimize-target behavior:

- kernel-target optimize excludes `torch-npu-optimize-knowledge`
- operator-target optimize includes `torch-npu-optimize-knowledge`

This is intentionally a physical-layout change, not a staging-semantics change.

All runtime prompts, memory-file guidance, subagent instructions, and workflow text that refer to the skill by logical name should continue to use `torch-npu-optimize-knowledge`. They should not be rewritten to a new logical identifier.

### Claude Plugin Builder Becomes Fixed-Contract Packaging

The Claude plugin does not expose optimize-target selection, so the builder should stop pretending that it can produce target-specific optimize payloads.

The builder contract should therefore change as follows:

- remove `optimize_target` from `build_claude_optimize_plugin_assets(...)`
- remove `optimize_target` from `build_claude_optimize_plugin(...)`
- keep the builder focused on the fixed Triton plugin workflow contract

Implementation may still reuse existing staging helpers, but the final optimize skill payload used by the builder must be target-independent from the plugin caller's perspective and must not contain `torch-npu-optimize-knowledge`.

This means the builder should explicitly verify the packaged optimize skill set excludes `torch-npu-optimize-knowledge`, and the built plugin `skills/` tree should also exclude it.

No user-visible manifest, hook, or agent-text expansion is needed for this change. The intended effect is contract simplification and packaging hardening, not a new plugin feature.

### Path-Based References Must Follow The Move

Any repository-owned asset that names the source-tree path of the Torch NPU knowledge skill must be updated to the new location. This includes:

- `scripts/update-optimize-knowledge-indices.sh`
- `.codex/skills/create-optimize-pattern/SKILL.md`
- tests that read or assert the physical repository path

References that are intentionally logical or staged-layout-oriented should stay logical. For example, user-facing guidance that says to use the staged `torch-npu-optimize-knowledge` skill should remain unchanged.

## Verification

The implementation should be considered complete only if all of the following are true:

- `skills/torch/torch-npu-optimize-knowledge/` exists as the live repository skill directory
- `skills/triton/torch-npu-optimize-knowledge/` is no longer a live repository skill directory
- `resolve_staged_skills(CommandKind.OPTIMIZE, optimize_target="kernel")` still excludes `torch-npu-optimize-knowledge`
- `resolve_staged_skills(CommandKind.OPTIMIZE, optimize_target="operator")` still includes `torch-npu-optimize-knowledge`
- the Claude plugin builder API no longer accepts `optimize_target`
- `build_claude_optimize_plugin_assets()` does not return `torch-npu-optimize-knowledge` in `optimize_skill_names` or the final `skill_names`
- a built plugin directory does not contain `skills/torch-npu-optimize-knowledge`

Recommended verification commands:

- `bash scripts/update-optimize-knowledge-indices.sh`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_staging.py tests/test_claude_optimize_plugin.py tests/test_generation_contracts.py`
