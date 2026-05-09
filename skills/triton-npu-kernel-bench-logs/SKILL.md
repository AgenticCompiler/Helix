---
name: triton-npu-kernel-bench-logs
description: Interprets NPUKernelBench-style operator workspaces with baseline, opt-round-N, PyTorch and Triton sources, and perf text logs. Use when reviewing archived optimization runs under trees such as workspace/NPUKernelBench_level_1_2_triton, comparing rounds without rerunning benchmarks, extracting PyTorch timings from raw-op-statistic-case lines, distilling lessons into triton-npu-optimize-knowledge-v2 pattern cards, or tracking progress in PATTERN_AND_LOG_SYNC_PROGRESS.md.
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

## Distilling lessons into `triton-npu-optimize-knowledge-v2`

When the task is to fold bench history into durable optimize knowledge:

- Main objective: learn more precisely when and how each pattern should or should not be tried, and discover new patterns when existing cards do not fit.
- Follow [references/optimization-lessons-workflow.md](references/optimization-lessons-workflow.md) for the full procedural checklist.

### 1) Build the relevant kernel inventory

- Scope kernels with optimization records (`opt-note.md` plus `opt-round-*`) and map rounds semantically against `skills/triton-npu-optimize-knowledge-v2/references/pattern_index.md` (not keyword matching).
- During mapping (before synthesis), keep a temporary per-card checklist section:
  - `## NPUKernelBench field inventory`
  - `**Operator workspaces (deduped):**`
  - one deduped bullet per operator workspace attributed to that card.
- Treat this inventory as coverage bookkeeping only; it is not final prose.

### 2) Write round narratives on pattern cards

- For each mapped round, append an **expanded narrative** to the matched card in `skills/triton-npu-optimize-knowledge-v2/references/patterns/`.
- **Mandatory five-field template (every narrated round):**
  - **`Kernel / round / parent`**
  - **`Pre-change scenario`**
  - **`Change`**
  - **`Evidence`**
  - **`Interpretation`**
- If a pattern truly has no applicable round for an operator (for example no `@triton.autotune` work), write a short non-round note under a `###` heading instead of fabricating a five-field block.
- Each round entry must be detailed enough to reconstruct the scenario without reopening raw logs (`attempts.md`, `summary.md`, profile files, perf files). Repetitive rounds may shorten wording, but must keep all five fields.
- Include code context, concrete kernel change, profile/perf evidence, and why the attempt worked or failed relative to current card guidance.
- **New pattern cards vs log citations:** `opt-round-*/attempts.md` only cite a fixed legacy slug set. Absence of a new slug does not prove no new mechanism exists; create a new v2 card when no existing `## Summary` / `## Use When` semantically fits.

### 3) Run narrative coverage checks

- After one full kernel-to-pattern mapping pass and before synthesis, run a **gap-filling audit** for each touched card:
  - every inventory operator appears in at least one narrative section (or explicit no-applicable-round note),
  - prioritize small inventory cards first,
  - add missing success/failure/anti-signal narratives from real rounds.
- Then run a strict second audit:
  - every inventory operator has at least one explicit five-field round block,
  - inventory-only mentions or cross-card references do not satisfy this.

### 4) Synthesize each touched pattern card

- After rounds are fully mapped and audited, rewrite card guidance using all narratives:
  - refine practical `Use When` / `Avoid When` / expected signals verdicts,
  - incorporate failures, validated branches, and final wins (not just best rounds),
  - encode staged strategy when evidence supports it,
  - include explicit anti-signals for stop conditions.
- Synthesis writing requirements:
  - use kernel-agnostic wording (no operator-specific branch labels as universal terms),
  - keep the card self-contained and abstract (no dependence on round IDs or artifact paths),
  - explain technical terms briefly in plain language,
  - preserve existing code samples and add new concise samples when needed so each pattern remains directly actionable.
- **At final synthesis for each card, remove the entire temporary `NPUKernelBench field inventory` section** so committed content remains durable pattern guidance.

### 5) Rewrite `pattern_index.md` manually after full synthesis

- After synthesis is complete for all targeted pattern cards, rewrite `skills/triton-npu-optimize-knowledge-v2/references/pattern_index.md` as a human-authored quick-match index.
- Format requirements:
  - one **second-level markdown header** (`##`) per pattern slug,
  - inside each pattern section, write a short **4-5 line** summary of key points, with strongest emphasis on **when the pattern should be used**,
  - keep wording optimized for **locatability** so an LLM agent can quickly match current kernel symptoms to the right pattern,
  - length per pattern can vary, but compact scanability is mandatory.
- Authoring requirements:
  - read each source pattern card and write your own concise summary,
  - do not mechanically copy sentences from the pattern card into the index.
- **Do not use Python scripts (or other automatic generation) for `pattern_index.md`; maintain this file by direct manual editing.**

### 6) Finalize tracking artifacts

- Track operator-level progress in `workspace/NPUKernelBench_level_1_2_triton/PATTERN_AND_LOG_SYNC_PROGRESS.md` (`n/a` for kernels without records; `todo` -> `done` for completed passes).

## Related skills

- `triton-npu-optimize` for executing new optimization rounds and live artifact discipline.
