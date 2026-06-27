# Compiler Source Navigation Map

## Purpose

This reference helps narrow one performance-related compiler question to a small set of AscendNPU-IR source locations.

## Default Reading Order

`round evidence -> <compiler-source-dir>/docs -> <compiler-source-dir>/bishengir/lib -> <compiler-source-dir>/bishengir/include(when needed) -> <compiler-source-dir>/bishengir/test(rare fallback)`

## Directory Atlas

### `<compiler-source-dir>/docs/source/*/developer_guide/passes/`

- Read this first when the question is stage-oriented or pass-oriented.
- Use it to understand what a pass family is supposed to do.

### `<compiler-source-dir>/docs/source/*/developer_guide/features/`

- Read this first when the symptom looks like a feature or subsystem behavior.
- Use it to orient on pipeline or memory concepts before implementation reading.

### `<compiler-source-dir>/bishengir/lib/Conversion/`

- Default implementation root for lowering-path and pass-effect questions.
- Start here when the symptom looks tied to stage-to-stage IR transformation.

### `<compiler-source-dir>/bishengir/lib/Dialect/`

- Default implementation root for dialect-level behavior and op semantics.
- Start here when the symptom looks tied to operation meaning or inserted IR structure.

### `<compiler-source-dir>/bishengir/lib/Transforms/`

- Default implementation root for transform-heavy behavior not anchored to one dialect.
- Start here when the symptom feels like a general optimization pass rather than one conversion pipeline.

### `<compiler-source-dir>/bishengir/include/bishengir/`

- Use only when `Passes.td`, `Passes.h`, interface declarations, or registration surfaces are needed.
- Treat it as an aid for finding the right implementation file under `<compiler-source-dir>/bishengir/lib/`.

## Symptom To Subtree

- vectorization loss -> `<compiler-source-dir>/docs/.../passes/` then `<compiler-source-dir>/bishengir/lib/Conversion/` and `<compiler-source-dir>/bishengir/lib/Transforms/`
- copy or sync growth -> `<compiler-source-dir>/docs/.../features/` then `<compiler-source-dir>/bishengir/lib/Dialect/` and `<compiler-source-dir>/bishengir/lib/Conversion/`
- buffer expansion or layout churn -> `<compiler-source-dir>/docs/.../features/` then `<compiler-source-dir>/bishengir/lib/Dialect/` and `<compiler-source-dir>/bishengir/lib/Conversion/`
- suspicious stage transition -> `<compiler-source-dir>/docs/.../passes/` then the matching `<compiler-source-dir>/bishengir/lib/` subtree
- unclear pass naming or registration surface -> `<compiler-source-dir>/bishengir/include/bishengir/**/Passes.td` and `Passes.h`

## Search Recipes

```bash
rg -n "vector|vectorize|auto-vectorize" <source-root>/docs <source-root>/bishengir/lib
rg -n "copy|dma|wait|barrier|sync" <source-root>/docs <source-root>/bishengir/lib
rg -n "Passes\\.td|Passes\\.h" <source-root>/bishengir/include/bishengir
```

## Anti-Patterns

- Do not start with a whole-tree search across every directory.
- Do not read `<compiler-source-dir>/bishengir/include/` before you know which subsystem matters.
- Do not treat `<compiler-source-dir>/bishengir/test/` as the default path for performance analysis.
