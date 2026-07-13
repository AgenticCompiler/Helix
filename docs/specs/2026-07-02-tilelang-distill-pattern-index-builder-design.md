# TileLang Distill Pattern Index Builder Design

## User-Visible Behavior

Running `ta-cli distill --lang tilelang` should use the staged `tilelang-npu-optimize-knowledge` skill the same way Triton distill uses `triton-npu-optimize-knowledge`. When a distill agent edits or adds pattern cards, the CLI should regenerate the staged `references/pattern_index.md` instead of failing with a missing `scripts/build_pattern_index.py` error.

## Root Cause

The distill workflow calls `rebuild_pattern_index()` after successful distill or analysis agent steps. That helper expects every optimize knowledge skill to expose `scripts/build_pattern_index.py`. The TileLang knowledge skill already contained the parser, renderer, and CLI logic in `scripts/pattern_catalog.py`, but it did not expose the standard builder entrypoint.

## Design

Move generated-index implementation into runtime code under `src/helix/optimize_knowledge/`:

- `pattern_index.py` owns pattern-card parsing, high-priority reminder generation, and pattern-index rendering.
- `symptom_index.py` owns symptom-card parsing and symptom-index rendering.

The optimize knowledge skills keep authored cards and generated Markdown indexes, but they no longer own index-generation Python scripts. This matches the actual usage: agents consume staged Markdown references, while the CLI and repository maintenance commands regenerate indexes.

Adapt distill's `rebuild_pattern_index()` to call the `src` renderer directly instead of subprocess-running `skills/*/scripts/build_pattern_index.py` from the editable skill workspace. Runtime high-priority reminders should also call the same `src` renderer rather than loading a staged skill script.

Keep `scripts/update-optimize-knowledge-indices.sh` as the repository maintenance entrypoint, but make it call the `src` module CLI with explicit input and output paths.

Add a regression test that copies the bundled TileLang knowledge skill into a temporary editable distill workspace, removes any skill-local builder script, and calls `rebuild_pattern_index()` against it. The test should fail while distill still depends on a skill-local script and pass once distill uses the `src` renderer.

## Verification

- Run the new distill knowledge workspace regression test.
- Run strict skill-script pyright for the new TileLang builder script.
- Run the repository standard checks if the targeted verification is clean.
