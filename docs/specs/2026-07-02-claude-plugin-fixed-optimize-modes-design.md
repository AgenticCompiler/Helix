# Claude Plugin Fixed Optimize Modes Design

## Goal

Make the generated Claude optimize plugin carry the same resolved optimize mode
contract that CLI-launched optimize sessions provide in their agent prompt.

## User-Visible Semantics

- The standalone Claude optimize plugin uses `differential` correctness mode.
- The standalone Claude optimize plugin uses `torch-npu-profiler` benchmark mode.
- The generated optimize agent states those fixed modes directly so Claude does
  not need to infer them from generic skill prose.
- Baseline preparation, harness generation, validation commands, and
  `baseline/state.json` should all use those resolved modes unless an existing
  valid baseline already records matching modes.
- The plugin does not add a runtime mode selector. Users who need alternate
  modes should use the CLI `optimize` command, where `--test-mode` and
  `--bench-mode` remain explicit options.

## Design

Keep the fixed plugin modes in `scripts/build-claude-optimize-plugin.py`, next
to the generated agent contract. The builder should render a short "Fixed
Optimize Modes" section into `agents/helix-optimize.md` before the
critical workflow rules.

This keeps the mode contract in the user-facing agent instructions, matching
where CLI optimize prompts already place resolved mode context. Hook runtime
state remains focused on temporary workflow lifecycle data, not static plugin
policy.

## Testing

- Add a builder unit test that asserts the generated agent text includes the
  fixed `test-mode` and `bench-mode` guidance.
- Keep the existing plugin tree test to verify the generated agent file is
  written.
- Run focused plugin builder tests, then the standard repository verification
  commands.
