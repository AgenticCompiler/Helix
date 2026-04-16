# Inspect IR Ranking And Change Scan

## Goal

Improve `inspect_ir.py` so agents can more quickly prioritize which archived IR stages deserve attention and which adjacent compiler passes introduced the largest structural changes.

## User-visible behavior

- Extend `list-stages` with optional ranking support.
- Add a new `find-changes` subcommand that compares adjacent stages in archive order.
- Keep `--ir-dir` as the only user-facing IR selector; do not expose `bishengir_stages/`.
- Keep output terminal-oriented plain text.

## `list-stages` enhancement

Add an optional `--sort-by` flag with:

- `order`
  - default archive order
- `size`
  - largest file size first
- `lines`
  - largest line count first
- `interesting`
  - highest heuristic score first

The heuristic score should stay lightweight and explainable. A good first version is a weighted sum of existing keyword counts, with more weight on signals often relevant to performance investigations:

- `alloc`
- `copy`
- `vector`
- `load`
- `store`
- `dma`
- `wait`
- `set_flag`
- `barrier`
- `matmul`

When sorting by `interesting`, include the computed score in the rendered stage list.

## `find-changes` subcommand

Purpose:
- scan adjacent stages and rank the biggest textual and structural transitions

Input:
- `--ir-dir <path>`
- optional `--limit <N>`
- optional `--sort-by <metric>`

Suggested sort metrics:
- `score`
  - default combined change score
- `lines`
  - absolute line-count delta
- `size`
  - absolute byte delta

Output:
- one adjacent stage pair per block or line
- include:
  - `from` stage
  - `to` stage
  - line delta
  - size delta
  - keyword delta summary
  - combined score

The first version should compare only adjacent stages in archive order. It does not need to search all stage pairs.

## Design notes

- Reuse existing stage discovery, keyword counting, and selector logic.
- Keep scoring transparent and heuristic-based rather than claiming performance conclusions.
- The score is only for prioritization; the agent should still inspect summaries and diffs before drawing conclusions.
- Prefer stable output that agents can grep or copy into follow-up prompts.

## Documentation updates

- Update `skills/triton-npu-analyze-ir/SKILL.md` to mention stage ranking and adjacent change scanning as the preferred first pass over large archives.
- Update `README.md` with one short example for `find-changes`.

## Verification

- Add tests for:
  - `list-stages --sort-by interesting`
  - `list-stages --sort-by size`
  - `find-changes`
- Run targeted inspection tests plus the existing full validation commands.
