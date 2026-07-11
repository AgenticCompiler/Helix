# Generation Prompt Option Implementation Plan

**Goal:** Add `--prompt` support to `gen-eval`, `gen-eval-batch`, `gen-test`, and `gen-bench` with the same prompt-append behavior already used by `convert` and `optimize`.

**Architecture:** Extend the shared generation option payload so parser-level `--prompt` values flow through existing generation request construction. Reuse the shared prompt append helper in generation orchestration so single and batch generation both get identical `Additional user instructions:` semantics.

**Tech Stack:** Python 3, `argparse`, `unittest`, existing generation/request builder modules

## Task 1: Cover Parsing And Prompt Composition

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_commands.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/generation/models.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `src/helix/generation/orchestration.py`

Steps:

- Add parser tests for `gen-eval`, `gen-eval-batch`, `gen-test`, and `gen-bench` accepting `--prompt`.
- Add request-builder tests that verify appended `Additional user instructions:` content for generation flows.
- Run the focused tests and confirm they fail before implementation.
- Implement the minimal parser/model/request wiring.
- Re-run the focused tests and confirm they pass.

## Task 2: Prove Batch Prompt Propagation

**Files:**
- Modify: `tests/test_generation_batch.py`
- Modify: `src/helix/generation/batch.py` only if shared request construction is insufficient

Steps:

- Add a batch test that captures each `gen-eval-batch` request prompt.
- Verify the test fails before prompt wiring exists.
- Keep batch behavior on shared `build_generation_request(...)` construction if possible.
- Re-run the batch test and confirm it passes.

## Final Verification

Run a focused regression slice covering:

- CLI parser behavior for the new prompt-enabled generation commands
- request-builder prompt composition
- `gen-eval` staged skill preservation
- batch prompt propagation
