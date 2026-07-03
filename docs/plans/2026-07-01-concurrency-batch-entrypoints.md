# Concurrency Batch Entrypoints Implementation Plan

## Goal

Allow single-workspace commands to delegate to their existing batch implementations when `--concurrency` is explicitly provided, while keeping existing `*-batch` commands supported.

## Steps

1. Add parser tests proving single commands default `concurrency` to `None`, accept explicit `--concurrency`, and keep batch command defaults unchanged.
2. Add dispatch tests proving `gen-eval`, `convert`, `optimize`, `log-check`, `report`, and `verify` call their batch implementations when `--concurrency` is explicit.
3. Add parser support for optional concurrency on those single commands and move batch-only options onto the matching single commands where needed.
4. Update command handlers to delegate to existing batch handlers when `args.concurrency is not None`; `verify` prints a warning before delegation.
5. Update README examples and command descriptions to document both spellings.
6. Run focused tests first, then the repository standard verification commands.
