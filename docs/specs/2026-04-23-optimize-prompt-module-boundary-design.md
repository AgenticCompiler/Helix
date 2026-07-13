# Optimize Prompt Module Boundary Design

## Summary

Move optimize-specific prompt construction helpers out of `src/helix/prompts.py` and into `src/helix/optimize/prompts.py`.

Keep the top-level prompt entrypoint stable for non-optimize commands and preserve existing optimize prompt behavior.

## Goals

- Put optimize-only prompt logic under the `optimize/` feature package.
- Keep `src/helix/prompts.py` focused on shared prompt assembly.
- Avoid behavior changes in optimize, optimize-batch, supervised optimize, or optimize resume flows.
- Minimize import churn for callers that still rely on top-level prompt helpers during the transition.

## Non-Goals

- Do not redesign prompt wording.
- Do not change CLI flags, optimize workflow semantics, or runner behavior.
- Do not split non-optimize prompt construction into additional modules in this change.

## Design

Create a new module at `src/helix/optimize/prompts.py` that owns:

- `strict_learned_lessons_lines`
- `layered_analysis_lines`
- `compiler_source_analysis_lines`
- `build_optimize_worker_prompt`
- `build_optimize_unsupervised_prompt`
- `build_optimize_supervisor_prompt`
- `build_optimize_resume_prompt`

Keep `src/helix/prompts.py` as the shared prompt entry module. It should continue to own:

- `PROMPT_INTROS`
- `append_additional_user_instructions`
- `build_prompt`
- non-optimize prompt assembly for generation and execution commands

`build_prompt` should delegate optimize-specific sections to `helix.optimize.prompts` instead of defining those helpers inline.

For compatibility, `src/helix/prompts.py` may re-export optimize prompt helpers so existing tests and any internal callers do not break during this refactor.

## Expected Call-Site Changes

- Update optimize-local modules such as guidance, execution, and run-loop to import optimize prompt helpers from `helix.optimize.prompts`.
- Leave shared callers on `helix.prompts.build_prompt`.

## Verification

Run focused unit tests that cover:

- optimize prompt construction
- optimize resume prompt construction
- optimize batch user prompt propagation
- direct imports used by optimize orchestration and runners
