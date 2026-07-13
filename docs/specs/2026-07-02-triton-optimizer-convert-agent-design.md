# Triton Optimizer Convert Agent Design

## Goal

Rename the generated Claude Code plugin from `helix-optimize` to `triton-optimizer` and package both supported Helix workflows needed by this plugin:

- optimize, through the existing optimize agent and optimize skill set
- convert, through a new Triton-only convert agent and the Triton convert skill set

The plugin remains focused on Triton workflows. It must not package TileLang convert skills for the convert agent.

## User-Visible Semantics

Building the Claude plugin should produce a `triton-optimizer` plugin. The generated plugin should expose two Claude agents:

- `helix-optimize` for optimize sessions
- `helix-convert` for convert sessions

The optimize agent keeps its current workflow contract, including fixed optimize modes and plugin-managed optimize state hooks. The convert agent is a separate workflow agent. It converts one PyTorch operator into a Triton Ascend NPU-backed operator, keeps the original input immutable, writes to the requested output path, and validates through the convert skill's standalone or differential test flow.

## Skill Packaging

The builder should continue resolving optimize skills through the existing staging contract for `CommandKind.OPTIMIZE`.

The convert agent should resolve skills through the existing staging contract for `CommandKind.CONVERT` with `language="triton"`. The expected convert skill payload is:

- `triton-npu-convert-pytorch-operator`
- `ascend-npu-gen-test`
- `ascend-npu-run-eval`
- `triton-npu-repair-guide`

The builder should copy the union of optimize and convert skills into the plugin `skills/` directory. Duplicate skill names should be copied once. Skill source mappings should still be honored when present.

## Agent Content

The optimize agent renderer should remain narrow and continue to describe optimize-only state, baseline, and round rules.

The new convert agent renderer should mirror the optimize agent structure where useful:

- frontmatter with `name`, `description`, `model`, allowed tools, and bundled `skills`
- a primary workflow skill declaration using `triton-npu-convert-pytorch-operator`
- convert-specific rules that the original input file is immutable, the output path is the converted artifact, and validation must use the bundled test and run-eval skills

The convert agent must not mention optimize baseline submission, round lifecycle, or `.helix` workflow state as required convert behavior.

## Tests

Update the Claude plugin builder tests to verify:

- the manifest plugin name is `triton-optimizer`
- the generated text files include both agent files
- the optimize agent still contains the existing fixed optimize mode and round rules
- the convert agent names `triton-npu-convert-pytorch-operator` as the primary skill
- builder assets include the Triton convert staging skill set
- built plugin output contains `skills/triton-npu-convert-pytorch-operator`
- built plugin output does not contain `skills/tilelang-npu-convert-pytorch-operator`

Focused plugin builder tests are sufficient because the change is limited to packaging metadata, agent text, and skill selection.
