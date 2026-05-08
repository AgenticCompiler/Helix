---
name: triton-npu-kernel-bench-logs
description: Interprets NPUKernelBench-style operator workspaces with baseline, opt-round-N, PyTorch and Triton sources, and perf text logs. Use when reviewing archived optimization runs under trees such as workspace/NPUKernelBench_level_1_2_triton, comparing rounds without rerunning benchmarks, extracting PyTorch timings from raw-op-statistic-case lines, distilling lessons into triton-npu-optimize-v2 pattern cards, or tracking progress in PATTERN_AND_LOG_SYNC_PROGRESS.md.
---

# Kernel bench logs

## Goal

Read and summarize completed optimization evidence in an NPUKernelBench-style operator directory (for example `workspace/NPUKernelBench_level_1_2_triton/NN_OperatorName/`) without treating the tree as a live optimization workspace unless the user asks to continue rounds.

## When to use

- The user points at `NPUKernelBench_level_1_2_triton` or a sibling bench export and wants an overview, diff narrative, or perf story.
- You need to compare PyTorch reference numbers to initial Triton and final Triton using existing `*_perf.txt` files only.
- You need the fastest on-ramp to what happened across rounds (`opt-note.md`, `learned_lessons.md`, then selective `opt-round-*`).

## Directory map

Follow the canonical path and filename semantics in [references/kernel-bench-layout.md](references/kernel-bench-layout.md). That file lists operator-level paths, optional round artifacts, and how to read PyTorch timing lines.

## Locating bench trees (`workspace/` is gitignored)

This repository’s **`.gitignore` ignores the whole `workspace/` directory** (see repo root `.gitignore`). Tools that **honor Git ignore rules**—including **repository-scoped ripgrep**, many IDE “search in files” views, and **some workspace-wide glob or listing tools**—often **omit `workspace/` entirely**. An **empty search result does not prove** that `NPUKernelBench_level_1_2_triton` or `opt-round-*` artifacts are missing on disk.

**Always** discover or search bench logs using methods that **include gitignored paths**:

1. **Direct reads** — Open files by **full absolute path** with the **Read** tool (for example `…/workspace/NPUKernelBench_level_1_2_triton/NN_OperatorName/opt-round-1/attempts.md`). Path-based reads are not skipped just because `workspace/` is gitignored.
2. **`find` from the repo root** — POSIX `find` does **not** apply `.gitignore`; use it to list operators or rounds when you need a reliable inventory, for example:
   - `find workspace/NPUKernelBench_level_1_2_triton -maxdepth 1 -mindepth 1 -type d`
   - `find workspace/NPUKernelBench_level_1_2_triton -name 'attempts.md'`
3. **`rg --no-ignore` scoped to `workspace/`** — When ripgrep is installed and you need text search under the bench tree, disable ignore files so `.gitignore` does not hide `workspace/`:
   - `rg --no-ignore 'pattern' workspace/NPUKernelBench_level_1_2_triton`

If a **git-aware** search or glob returns nothing under `workspace/`, **do not conclude the tree is absent** until you confirm with **Read**, **`find`**, or **`rg --no-ignore`** as above. The same rule applies to sibling exports the user places under `workspace/`.

## Reading order

1. `opt-note.md` for the cross-round ledger and overall outcome.
2. `learned_lessons.md` for distilled, reusable rules (not round play-by-play).
3. `baseline/` versus top-level `triton_NN_OperatorName.py` and `triton_NN_OperatorName_perf.txt` to confirm the starting Triton snapshot and its benchmark.
4. Top-level `opt_triton_NN_OperatorName.py` and `opt_triton_NN_OperatorName_perf.txt` for the latest optimized snapshot as archived in that tree.
5. `opt-round-{i}/` in ascending `i` when the user needs attempt-level detail: `attempts.md`, `summary.md`, `round-state.json`, then optional deeper artifacts (`perf-analysis.md`, `profile/`, `ir/`) when present.
6. `NN_OperatorName.json` when shapes, dtypes, or case indices need to be tied back to perf lines.
7. `NN_OperatorName.py` and `NN_OperatorName_perf.txt` for the PyTorch reference implementation and its benchmark export.

## Perf text conventions

- Triton and optimized Triton benchmark exports use the same `*_perf.txt` pattern family as the PyTorch file: locate per-case blocks and any `raw-op-statistic-case-*` comment lines when you need structured timing JSON.
- For PyTorch timing specifically, read `avg_time_us` (and related fields) under each `raw-op-statistic-case-*` comment’s `"ops"` array in `NN_OperatorName_perf.txt`, as described in the layout reference.
- `opt-round-{i}/logs/compare-perf.txt` may be absent. When it exists, treat it as a convenient comparison transcript; when it does not, rely on `summary.md`, `attempts.md`, `round-state.json`, paired `*_perf.txt` exports, and `opt-note.md` instead of fabricating a consolidated speedup table.

## Boundaries

- Ignore `optimize-logs/` unless the user explicitly requests it.
- When `logs/compare-perf.txt` exists for a round, cite it instead of rebuilding the same comparison by hand from raw perf exports.
- For live optimization work, follow `triton-npu-optimize` and its artifact contract; use this skill for retrospective log reading and cross-run synthesis.

## Distilling lessons into `triton-npu-optimize-v2`

When the task is to fold bench history into durable optimize knowledge:

- Main objective: learn more precisely when and how each pattern should or should not be tried, and discover new patterns when existing cards do not fit.
- Follow [references/optimization-lessons-workflow.md](references/optimization-lessons-workflow.md): per kernel with a record, read PyTorch + initial Triton, walk rounds using `opt-note.md`, and map every round to patterns by semantic comparison against `skills/triton-npu-optimize-v2/references/pattern_index.md` (not keyword matching).
- For each round, append an **expanded narrative** to the matched pattern card(s) in `skills/triton-npu-optimize-v2/references/patterns/`.
- At the start of each touched pattern card, ensure a canonical inventory section exists:
  - `## NPUKernelBench field inventory`
  - `**Operator workspaces (deduped):**`
  - one bullet per operator workspace (deduped).
- **Mandatory five-field template (every narrated round):** do not substitute one-line summaries, merged multi-round blurbs, or italic disclaimers for real round entries. Each round you document on a pattern card must use this exact bullet list (all five labels present, in this order, even when a field is short):
  - **`Kernel / round / parent`**
  - **`Pre-change scenario`**
  - **`Change`**
  - **`Evidence`**
  - **`Interpretation`**
  If a pattern truly had **no** applicable round for that operator (for example no `@triton.autotune` work on the autotune card), write a short **non-round** note under a `###` heading—do **not** fake a five-field block. When you *do* narrate a round, skipping any of the five fields is not allowed.
- Each round entry must still be detailed enough that a later synthesis pass can reconstruct the scenario **without reopening raw logs** (`attempts.md`, `summary.md`, profile files, or perf files). Repetitive rounds may compress **wording** inside each field, but **not** by omitting a field or merging unrelated rounds into one pseudo-entry.
- Include code context, concrete kernel change, profile/perf evidence, and why the attempt worked or failed relative to the existing card guidance.
- **New pattern cards vs log citations:** `opt-round-*/attempts.md` files in this repo’s bench export link only a **fixed set** of legacy `patterns/*.md` paths (for example `autotune.md`, `program-multiple-rows.md`, …). Those filenames map onto existing `triton-npu-optimize-v2` cards. **Absence of a new filename in logs does not prove there is no new mechanism**—agents must still semantically map each round to `pattern_index.md` and **add a new card** when no existing `## Summary` / `## Use When` fits, even if the round’s `attempts.md` never cited that slug. Conversely, **if every round maps cleanly to existing cards, do not invent new pattern files** just to satisfy a quota; cite the inventory (use `grep`/`find` under `workspace/…` as in **Locating bench trees**) when explaining why no new slug was added.
- If no existing pattern matches a round's optimization mechanism, create a new pattern card in `skills/triton-npu-optimize-v2/references/patterns/`.
- After one full kernel-to-pattern mapping pass and before per-pattern synthesis, run an explicit **gap-filling audit**:
  - Verify every operator listed under `Operator workspaces (deduped)` appears in at least one narrative section on that card.
  - Prioritize cards with fewer inventory items first (small-card audit first).
  - Add missing narratives or explicit negative/anti-signal narratives using real round evidence; do not leave inventory-only operators.
- Then run a second strict audit to ensure every inventory operator has at least one explicit **five-field** round block containing:
  - `Kernel / round / parent`
  - `Pre-change scenario`
  - `Change`
  - `Evidence`
  - `Interpretation`
  Inventory mentions or cross-card notes do not satisfy this requirement.
- After all targeted rounds are mapped, run a per-pattern synthesis pass to update each touched card with clearer "use when / avoid when / expected signals" verdicts derived from all narratives.
- Regenerate `skills/triton-npu-optimize-v2/references/pattern_index.md` with `skills/triton-npu-optimize/scripts/build_pattern_index.py` after pattern edits.
- Track operator-level status in `workspace/NPUKernelBench_level_1_2_triton/PATTERN_AND_LOG_SYNC_PROGRESS.md` (kernels without `opt-note.md` / rounds stay `n/a`; kernels with history start as `todo` until the pass completes).

## Related skills

- `triton-npu-optimize` for executing new optimization rounds and live artifact discipline.
