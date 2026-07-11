# Optimize Guidance Template Refactor Design

## Context

`OptimizeGuidanceManager` currently renders temporary optimize guidance for `AGENTS.md` and `CLAUDE.md` through inline string concatenation in [src/helix/optimize/guidance.py](/Users/cdj/Projects/helix/src/helix/optimize/guidance.py).

That approach works, but the guidance text is now harder to read and maintain because:

- shared rules are mixed into long string assembly code
- unsupervised and shared guidance duplicate structural patterns
- adding or reordering top-level guidance lines requires editing several string fragments

The current behavior should stay the same. This change is about maintainability and readability, not workflow semantics.

## Decision

Refactor optimize guidance rendering inside `guidance.py` to use code-local multiline templates plus small rendering helpers.

The refactor should:

- keep temporary top-level guidance generated from code, not from external template files
- keep separate templates for unsupervised guidance and shared orchestration guidance
- extract shared block rendering for rule bullets, layered-analysis bullets, and optional compiler-source sections
- preserve existing rendered text semantics unless a line was intentionally added already as part of prior approved work

## Template Structure

Introduce a small internal rendering structure in `guidance.py`:

- one multiline template for unsupervised optimize guidance
- one multiline template for shared optimize orchestration guidance
- one helper that renders a bullet list block from `list[str]`
- one helper that renders an optional paragraph or block only when content exists

Each render function should mostly prepare named values, then fill the template. It should no longer manually concatenate the full document line by line.

## Scope

This refactor applies only to temporary optimize guidance written into:

- `AGENTS.md`
- `CLAUDE.md`

It includes:

- `OptimizeGuidanceManager._render_unsupervised_guidance()`
- `OptimizeGuidanceManager._render_shared_guidance()`
- nearby local helpers that support those renderers

It does not include:

- optimize launch prompts in `src/helix/prompts.py`
- skill content under `skills/`
- archive, cleanup, or runtime session-recording behavior

## Expected Result

After the refactor:

- a reader should be able to understand each rendered guidance document from one compact template
- shared guidance rules such as cautious file reading and strict user-instruction handling should stay easy to locate
- tests should continue validating both `AGENTS.md` and `CLAUDE.md` output without behavioral regressions

## Non-Goals

- changing optimize workflow semantics
- introducing external Markdown template files
- changing prompt-generation code outside `guidance.py`
- adding new runtime enforcement or read-audit mechanisms
