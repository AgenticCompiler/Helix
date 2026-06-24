---
name: ascend-npu-kernel-bench-logs
description: Interprets NPUKernelBench-style operator workspaces with baseline, opt-round-N, PyTorch and Triton sources, and perf text logs (including opt_<Operator>.py and <Operator>_perf.txt naming). Use when reviewing archived optimization runs under trees such as workspace/NPUKernelBench_level_1_2_triton or sibling exports, comparing rounds without rerunning benchmarks, extracting PyTorch timings from raw-op-statistic-case lines, distilling lessons into triton-npu-optimize-knowledge-v2 or triton-npu-optimize-knowledge-v3 pattern cards (keeping per-card NPUKernelBench field inventory while mapping), or tracking progress in PATTERN_AND_LOG_SYNC_PROGRESS.md at the bench export root (create that file there if missing).
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

## Triton snapshot and perf filenames (two conventions)

Some exports use the historical **`triton_<Operator>.py`** / **`triton_<Operator>_perf.txt`** (initial Triton) and **`opt_triton_<Operator>.py`** / **`opt_triton_<Operator>_perf.txt`** (latest optimized Triton at operator root and under `opt-round-*`). Others use **`opt_<Operator>.py`** for the optimized Triton snapshot at the root and under **`opt-round-*`** (no `triton_*` / `opt_triton_*` filenames). PyTorch reference perf stays **`NN_OperatorName_perf.txt`** (**`<Operator>_perf.txt`**); Triton-side perf in the short scheme is often **`baseline/perf.txt`** or files next to each **`opt_<Operator>.py`**. Treat these as equivalent roles when reading diffs and perf: pick whichever filenames exist for that operator, and compare **baseline** vs **round** vs **top-level latest** consistently.

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
3. `baseline/` versus the top-level **initial** Triton snapshot and its benchmark: either `triton_NN_OperatorName.py` and `triton_NN_OperatorName_perf.txt`, or only **`baseline/`** plus **`baseline/perf.txt`** when there is no `triton_*` pair (exports that use **`opt_<Operator>.py`** at the root instead of `triton_*`).
4. The top-level **latest optimized** Triton snapshot and its Triton benchmark text: either `opt_triton_NN_OperatorName.py` and `opt_triton_NN_OperatorName_perf.txt`, or **`opt_<Operator>.py`** with Triton perf taken from **`baseline/perf.txt`**, **`opt-round-*`** neighbors, or any `*_perf.txt` that accompanies that kernel file—**not** the PyTorch export **`NN_OperatorName_perf.txt`** / **`<Operator>_perf.txt`**, which stays the reference side in [references/kernel-bench-layout.md](references/kernel-bench-layout.md).
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

## Distilling lessons into `triton-npu-optimize-knowledge`

When the task is to fold bench history into durable optimize knowledge, target the staged `triton-npu-optimize-knowledge` skill (which corresponds to `triton-npu-optimize-knowledge-v2` or `triton-npu-optimize-knowledge-v3` in the source tree, depending on the batch):

- Main objective: learn more precisely when and how each pattern should or should not be tried, and discover new patterns when existing cards do not fit.
- Follow [references/optimization-lessons-workflow.md](references/optimization-lessons-workflow.md) for the full procedural checklist.

### 0) Progress file at the bench export root

The bench export root is the directory that **directly contains** the operator workspace folders (for example `workspace/NPUKernelBench_level_1_2_triton/` or `workspace/kernelagent_v3/`).

**Progress table integrity (mandatory):** Never use **bulk or scripted transitions** that set **`Kernel log review`** or **`Pattern card follow-up`** from `todo` to `done` for more than one operator row in a single automated step (shell loops, Python rewrites of the whole table, workspace-wide `sed`/regex replace, “if `opt-note.md` exists then mark done”, and similar). Those heuristics **lie about completion** and have already corrupted real progress files. **Only change `todo` → `done` by hand:** one deliberate edit to **that operator’s row** immediately after that column’s work for **that kernel** is finished per this skill (guided pass, narratives, or follow-up as the column defines). Scripts may still **discover** missing `opt-note.md` / `opt-round-*` for **`Has opt record`** or append **citation bookkeeping** elsewhere—they must **not** flip review columns in batch.

- **Before** building inventory or editing pattern cards, ensure **`PATTERN_AND_LOG_SYNC_PROGRESS.md`** exists **in that bench export root** (alongside the operator directories, not inside a single operator).
- If the file is **missing**, **create it there as the first step** of this workflow: use the same purpose and table shape as `workspace/NPUKernelBench_level_1_2_triton/PATTERN_AND_LOG_SYNC_PROGRESS.md` (title, column semantics, one table row per operator directory under that root). Adjust the intro sentence so it names the actual export path you are using.
- For the rest of the workflow, read and update that file at **`{bench_export_root}/PATTERN_AND_LOG_SYNC_PROGRESS.md`** instead of assuming a fixed `NPUKernelBench_level_1_2_triton` path.

### 1) Build the relevant kernel inventory

- Scope kernels with optimization records (`opt-note.md` plus `opt-round-*`) and map **every** round **semantically** against `../triton-npu-optimize-knowledge/references/pattern_index.md` (not keyword matching). The staging system maps both v2 and v3 source trees to the same `triton-npu-optimize-knowledge` logical skill name — always target that staged skill for pattern index reads.
- **`attempts.md` citations are optional evidence, not the mapping source of truth.** Rounds that cite no pattern file, cite only `pattern_index.md`, or cite a misleading legacy slug must still be matched by reading the round hypothesis, code diff, and perf/profile outcome against the **full** index and detailed cards. Absence of a citation does **not** mean “skip pattern bookkeeping.”
- **If no existing pattern’s `## Summary` / `## Use When` is an honest semantic fit** for a round or operator theme after reading the index and the strongest candidate cards, **add a new card** under `references/patterns/<new-slug>.md` in the same knowledge tree (follow `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`), then **register it in `references/pattern_index.md`** for that tree (manual index rules in **§5** below apply to staged knowledge skill indices—do not leave new cards undiscoverable).
- **Keep the per-card checklist on every touched pattern card for the whole mapping phase.** As soon as a kernel round maps to a card, ensure that card has (or gains) a temporary section with this exact shape:
  - `## NPUKernelBench field inventory`
  - `**Operator workspaces (deduped):**`
  - one deduped bullet per operator workspace attributed to that card (add a bullet the first time that operator maps here; do not duplicate).
- Maintain this section while you walk more kernels and rounds: it is the authoritative **“which bench operators invoked this pattern”** list until synthesis. Do not delete it early, omit it because citations were missing in `attempts.md`, or assume the progress table replaces it—the progress file tracks row status; the inventory tracks **card coverage**.
- Treat inventory prose as coverage bookkeeping only; it is not the final synthesized guidance.

### 2) Write round narratives on pattern cards

- **Operator order (mandatory):** When choosing which kernels to narrate in each batch, follow **`{bench_export_root}/PATTERN_AND_LOG_SYNC_PROGRESS.md` from the first operator row downward** through the table as printed (do **not** infer ordering from directory name prefixes). Take the **next** operator that still needs narrative coverage for the active knowledge tree—**do not** skip to later rows while earlier rows with **`Pattern card follow-up: todo`** are still incomplete for narratives, **unless the user explicitly names a different subset or ordering.**
- For each mapped round, append an **expanded narrative** to the **one primary** pattern card under the active staged knowledge skill (`../triton-npu-optimize-knowledge/references/patterns/`, which the staging system maps from whichever v2/v3 source tree is the target).
- **One primary pattern per ledger round (mandatory):** Pick a **single** best semantic home for each `opt-round-*` narrative. Do **not** paste the same round as a full five-field block on multiple pattern cards. You may add a **one-line cross-reference** in `Interpretation` (for example “see **`tiling`** for this round”) when a second pattern lens is informative.
- **Compress consecutive same-pattern rounds (optional but encouraged):** When **several consecutive** ledger rounds for the same operator map to the **same** primary pattern and tell one sweep story (for example a tuple ladder or repeated “bounded config retuning”), merge them into **one** five-field block titled `#### Rounds a–b (...)` (or similar). The five fields must still be present at **merged** granularity: **`Kernel / round / parent`** should name the operator and round span; **`Pre-change scenario` / `Change` / `Evidence` / `Interpretation`** should summarize the sweep and point to `opt-note.md` plus one or two representative `opt-round-*/attempts.md` paths—do not require readers to open every round file.
- **Mandatory five-field template (every narrated round or merged span):**
  - **`Kernel / round / parent`**
  - **`Pre-change scenario`**
  - **`Change`**
  - **`Evidence`**
  - **`Interpretation`**
- If a pattern truly has no applicable round for an operator (for example no `@triton.autotune` work), write a short non-round note under a `###` heading instead of fabricating a five-field block.
- Each **single-round** entry must be detailed enough to reconstruct the scenario without reopening raw logs (`attempts.md`, `summary.md`, profile files, perf files). **Compressed spans** may shorten wording, but must keep all five fields and name the round range plus where the detailed table lives (`opt-note.md`).
- Include code context, concrete kernel change, profile/perf evidence, and why the attempt worked or failed relative to current card guidance.
- **New pattern cards vs log citations:** `opt-round-*/attempts.md` only cite a fixed legacy slug set. Absence of a new slug does not prove no new mechanism exists; create a new card in the target knowledge tree when no existing `## Summary` / `## Use When` semantically fits, and update that tree’s **`pattern_index.md`** so agents can find the new slug.

### 3) Run narrative coverage checks

- After one full kernel-to-pattern mapping pass and before synthesis, run a **gap-filling audit** for each touched card:
  - every inventory operator appears in at least one narrative section (or explicit no-applicable-round note),
  - prioritize small inventory cards first,
  - add missing success/failure/anti-signal narratives from real rounds.
- Then run a strict second audit:
  - every inventory operator has at least one **explicit five-field** narrative block **per covered pattern**—a block may cover **one round** or a **merged span** of consecutive same-pattern rounds (see step **2)**),
  - inventory-only mentions or cross-card references do not satisfy this.

### 4) Synthesize each touched pattern card

- After rounds are fully mapped and audited, rewrite card guidance using all narratives:
  - refine practical `Use When` / `Avoid When` / expected signals verdicts,
  - incorporate failures, validated branches, and final wins (not just best rounds),
  - encode staged strategy when evidence supports it,
  - include explicit anti-signals for stop conditions.
- **Synthesis output shape (mandatory):**
  - Treat synthesis as a **full card rewrite pass**, not as an appended recap section.
  - The post-synthesis file must read like a naturally authored pattern reference, where synthesized lessons are integrated into the main sections.
  - Do **not** keep per-round ledger prose, round IDs, or artifact-path narration in the final card body.
  - Do **not** satisfy synthesis by adding a trailing section such as `## Synthesis verdicts` while leaving the pre-synthesis body intact.
  - Prefer integrating lessons into common post-synthesis section families seen in v2 cards (for example: `## Summary`, `## Use When`, `## Avoid When`, `## Signals`, `## Repairs`/`## Common Repairs`, `## Failure Modes And Anti-signals`, `## Risks`, `## Related Patterns`, `## What To Verify After Applying`). Exact headings may vary by pattern, but the card must be **holistically rewritten**.
- **Information preservation (mandatory):**
  - Do **not** omit substantive guidance from the pre-synthesis card (for example technique catalogs, examples, detection heuristics, implementation notes) just because the card is being rewritten.
  - Preserve prior content and reorganize/refine it into the synthesized structure; merge or compress only when meaning is retained.
  - Remove prior content only when round evidence clearly shows it is misleading, invalid, or contradicted on the target stack.
  - Rationale: the synthesized card becomes the **only** future optimization guidance artifact; dropping valid prior knowledge can reduce future optimization effectiveness and cause regressions in recommendation quality.
- Synthesis writing requirements:
  - use kernel-agnostic wording (no operator-specific branch labels as universal terms),
  - keep the card self-contained and abstract (no dependence on round IDs or artifact paths),
  - explain technical terms briefly in plain language,
  - preserve existing code samples and add new concise samples when needed so each pattern remains directly actionable.
- **Only at final synthesis for that card** (after narrative coverage audits pass), remove temporary mapping scaffolding from that card:
  - remove the entire `## NPUKernelBench field inventory` section,
  - remove the entire `## Bench round narratives (...)` section.
  Until final synthesis, keep and extend those sections as mapping scaffolding.

#### Post-synthesis validation (required before marking a card done)

- Confirm the card no longer contains:
  - `## NPUKernelBench field inventory`
  - `## Bench round narratives (...)`
  - round-specific headers or round-ledger bullets used only for mapping.
- Confirm synthesized guidance is reflected in the main card sections (not in an appended recap block).
- Confirm major pre-synthesis knowledge blocks are preserved (or explicitly superseded by evidence), especially examples/technique catalogs that remain valid.
- **Mandatory side-by-side diff check before `done`:**
  - Compare the synthesized card **directly** against its pre-synthesis source card (same pattern slug) side-by-side.
  - Verify substantive guidance was not dropped during rewrite (examples, technique cases, detection heuristics, implementation notes, caveats).
  - If any substantive content is missing and not clearly invalidated by round evidence, restore it before marking the card done.
- If unsure about style/shape, compare representative cards:
  - pre-synthesis style: the original `triton-npu-optimize-knowledge` skill's pattern cards (staged flat alongside this skill)
  - post-synthesis style: the pattern cards you are actively editing in this workflow (same staged skill, `../triton-npu-optimize-knowledge/references/patterns/`)

### 5) Rewrite `pattern_index.md` manually after full synthesis

- After synthesis is complete for all targeted pattern cards, rewrite `pattern_index.md` under the staged knowledge skill (`../triton-npu-optimize-knowledge/references/pattern_index.md`) as a human-authored quick-match index.
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

- Track operator-level progress in **`{bench_export_root}/PATTERN_AND_LOG_SYNC_PROGRESS.md`** (see **0) Progress file** above; create the file at that root if it was missing). Use `n/a` for kernels without records. For **`Kernel log review`** and **`Pattern card follow-up`**, advance **`todo` → `done` only with a manual, per-row edit** after that kernel’s pass is complete—never bulk-mark from scripts (see **0) Progress table integrity**).

## Related skills

- `triton-npu-optimize` for executing new optimization rounds and live artifact discipline.
