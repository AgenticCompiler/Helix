# Helix Rename Implementation Plan

## Goal

Replace the repository's project name `triton-agent` / `triton_agent` with `helix` everywhere that represents this project's own identity.

## Steps

1. Rename tracked packages, service directories, hook files, and packaging assets whose paths contain the old project name.
2. Apply a mechanical text replacement across tracked source, tests, scripts, docs, and config for the old project-name variants and prefixes.
3. Fix targeted naming fallout such as class names, help text, packaging metadata, and runtime-path constants.
4. Run repository searches to confirm no tracked project-identity references remain.
5. Run lint, type-check, and tests; repair any breakages introduced by the rename.
