# From bench logs to `triton-npu-optimize-knowledge` pattern cards

Use this workflow when distilling `workspace/NPUKernelBench_level_1_2_triton` (or a sibling export such as `workspace/kernelagent_v3`) into durable pattern knowledge. **Target only** the designated staged knowledge skill (`triton-npu-optimize-knowledge`, which corresponds to `triton-npu-optimize-knowledge-v2` or `triton-npu-optimize-knowledge-v3` in the source tree) for pattern edits; leave the triton-npu-optimize workflow skill unchanged unless the project explicitly promotes another path.

## Finding files under `workspace/` (gitignored)

The repo root **`.gitignore` ignores `workspace/`**. Default **git-aware** searches (ripgrep from the repo without flags, many IDE search panels, and some glob/listing tools) **often skip `workspace/`**, which can look like “no `opt-round-*` / no `attempts.md`” even when the bench export exists.

**Always** use discovery that includes gitignored paths before concluding a kernel has no record:

- **Read** known paths by **full absolute path** (not skipped due to `.gitignore`).
- Shell: **`find workspace/NPUKernelBench_level_1_2_triton …`** (POSIX `find` does not read `.gitignore`).
- Shell: **`rg --no-ignore '…' workspace/NPUKernelBench_level_1_2_triton`** when you need content search under the bench tree.

Do not treat an empty git-aware repo search as proof the tree is missing. See `../SKILL.md` (section **Locating bench trees**) for the full rule.

## Core objective

Learn more precisely when and how each optimization pattern should or should not be used, based on round-level evidence. Discover and codify new patterns when existing cards are not a semantic fit.

## Progress tracking

The progress file lives at **`{bench_export_root}/PATTERN_AND_LOG_SYNC_PROGRESS.md`**, where **bench export root** is the directory whose immediate children are the operator workspace folders (for example `workspace/NPUKernelBench_level_1_2_triton/` or `workspace/kernelagent_v3/`).

**Forbidden:** bulk or scripted **`todo` → `done`** updates for **`Kernel log review`** or **`Pattern card follow-up`** (e.g. a Python loop or `sed` over the whole table, or “if `opt-note.md` exists then set done”). That pattern **writes false completion** and must not be used. **Required:** change **`todo` → `done` only by hand**, one operator row at a time, right after that kernel’s column work is actually finished. Repairing mistaken `done` values may use automation **only** with explicit human verification of each affected row; do not reintroduce bulk `todo` → `done`.

- **If that file does not exist, create it in the bench export root as the first step** before kernel inventory or pattern work. Mirror the structure of `workspace/NPUKernelBench_level_1_2_triton/PATTERN_AND_LOG_SYNC_PROGRESS.md` (title, column semantics table, operator index table); set the intro line to describe the actual path you are using.
- Optionally maintain narrative digests such as `{bench_export_root}/PILOT_KERNEL_DIGESTS.md` when batching multiple kernels.
- One table row per operator directory under that root.
- Mark **Has opt record** `no` when `opt-note.md` is missing or there are zero `opt-round-*` directories.
- Keep **Kernel log review** and **Pattern card follow-up** as `todo` until that kernel’s pass is finished.

## Per-kernel pass (only when an optimization record exists)

For each operator workspace `NN_OperatorName/` that has `opt-note.md` and `opt-round-*`:

1. **Context**
   - Read `NN_OperatorName.py` (PyTorch reference) and `NN_OperatorName.json` (cases).
   - Read the **initial** Triton snapshot: `triton_NN_OperatorName.py` if present, else the kernel under `baseline/` alongside **`opt_NN_OperatorName.py`** when the export uses the short naming scheme.
   - Skim `NN_OperatorName_perf.txt` for PyTorch `raw-op-statistic-case-*` timings when you need reference timings.

2. **Guided round walk**
   - Open `opt-note.md` and walk rounds in order (`## Round 1`, `## Round 2`, …).
   - For each round, open `opt-round-N/attempts.md` for hypotheses, cited pattern cards, analysis level, and failures; open `opt-round-N/summary.md` for the compact outcome; read `round-state.json` when paths or statuses need confirmation.
   - For **each** round, read `references/pattern_index.md` in the **same** knowledge tree you are editing (the `triton-npu-optimize-knowledge` skill, which maps to the appropriate v2/v3 source tree) and choose the best semantic match as **one primary pattern** for the five-field narrative. Optional secondary lenses belong as **short cross-references** inside `Interpretation`, not as duplicate full round blocks on other cards. **Even when `attempts.md` cites no `references/patterns/*.md` path or only cites the index.** Log citations are hints, not proof of mapping completeness.
   - Do not map patterns by keyword search alone. Use kernel structure, code change intent, and evidence from perf/profile logs.
   - If no existing pattern’s `## Summary` / `## Use When` is a defensible fit after reading the index and the top candidate cards, treat the round as a **new pattern candidate**: add `references/patterns/<slug>.md` in the staged knowledge skill (authoring contract in `docs/notes/2026-04-29-optimize-pattern-card-authoring.md`) and **update `references/pattern_index.md` in the staged knowledge skill** so the slug is discoverable on the next pass.

3. **Code diff discipline**
   - Compare Triton **before vs after** the round: parent candidate (usually prior round **`opt_triton_*.py`** or **`opt_*.py`**, else `baseline/`) vs this round’s **`opt_triton_*.py`** or **`opt_*.py`** (whichever the export uses).
   - Tie observed code motion back to the cited pattern (what matched the card, what did not).

4. **Kernel-local summary (before touching pattern cards)**

   Capture briefly:

   - Which patterns were **tried** (semantic mapping, not only cited names).
   - What **worked** vs **did not** (correctness, perf, promotion vs not promoted).
   - Any **successful pattern application** with concrete code shape (one or two sentences plus file/round pointers).

5. **Per-round narrative updates (knowledge-v2 or knowledge-v3 tree only)**

   **Batching order:** Walk operators in **`{bench_export_root}/PATTERN_AND_LOG_SYNC_PROGRESS.md` table order** (top to bottom as printed—do **not** assume any naming scheme for operator directories). When writing narratives in batches, always continue from the **first row that still lacks** the required coverage for this step; do not skip to later kernels unless the user instructs otherwise.

   For each round, append a narrative to the **primary** mapped pattern card (or to a new card if no match exists). Include:

   - Kernel and round id (`NN_OperatorName`, `opt-round-N`), plus parent round.
   - Code context before the change (hot path shape and bottleneck symptoms).
   - Exact change made (what was rewritten, tiled, fused, hoisted, hinted, etc.).
   - Observed evidence (profile/perf deltas, correctness outcomes, regressions).
   - Why it worked or failed compared with this card's current guidance.
   - Enough concrete detail to reconstruct the round later without reopening raw log files.
   - Novelty tag:
     - `novel`: adds new boundary/insight; keep detailed.
     - `repetitive`: same lesson as prior narratives; keep short and cross-reference existing narrative bullets.

   **One primary pattern per ledger round (mandatory):** each `opt-round-*` must have **exactly one** pattern card that carries its full five-field narrative. Do not duplicate the same round across multiple cards; use a **single-line cross-reference** in `Interpretation` when a second lens helps.

   **Compress consecutive same-pattern rounds (encouraged):** when several **consecutive** ledger rounds for the same operator belong on the **same** primary card and repeat one sweep (for example a tile tuple ladder), merge them into **one** five-field block titled `#### Rounds a–b (...)` (or similar). The merged block must still expose all five fields; cite `opt-note.md` for the per-round table and name one or two representative `opt-round-*` directories for code detail.

   **Five-field template (mandatory for every narrated round or merged span):** each block on a pattern card must include all of the following as explicit Markdown bullets with these exact bold labels (same order). Do not replace fields with a single undifferentiated prose paragraph.

   - **`Kernel / round / parent`**
   - **`Pre-change scenario`**: shape/workload context and bottleneck symptoms
   - **`Change`**: exact mechanism and key parameter values or branch guards
   - **`Evidence`**: correctness status plus main perf/profile observations (including promoted vs validated-branch vs not promoted)
   - **`Interpretation`**: why it likely worked or failed for this scenario, relative to current pattern guidance

   If the pattern has no applicable round for that operator, state that under a short heading; do not fabricate a five-field entry.

   The synthesis stage should be able to run from pattern-card narratives alone. Do not rely on future readers reopening `attempts.md` to recover missing context.

   **New cards:** triage text in `attempts.md` only references a small set of legacy pattern paths. Still perform semantic mapping against the full `pattern_index.md`. Create a new card in the target tree when the mechanism is not covered, and **extend that tree’s `pattern_index.md`**; omit new cards only when every round already maps cleanly to existing cards after an honest semantic pass (missing citations are not sufficient reason to skip mapping).

   **Inventory setup requirement (temporary until final synthesis for that card):**
   - **Keep** a deduped checklist on each touched pattern card for the entire mapping and narrative period:
     - `## NPUKernelBench field inventory`
     - `**Operator workspaces (deduped):**`
     - one bullet per operator workspace (add when that operator first maps to the card; never duplicate).
   - Extend the list as you process more kernels; it records **which operators invoked this pattern** and drives coverage checks—do not rely on the bench `PATTERN_AND_LOG_SYNC_PROGRESS.md` table alone for that.
   - At the **final synthesis step only** for that card (after audits), remove the inventory section so the committed card remains durable pattern guidance.

6. **Pattern card updates and new pattern creation**

   - If the round fits an existing card under the target knowledge tree’s `references/patterns/`, update that card with the narrative and any refined guidance.
   - If the round is a "new pattern candidate", create `references/patterns/<slug>.md` in that same tree following `docs/notes/2026-04-29-optimize-pattern-card-authoring.md` (`# Title`, required `## Summary` and `## Use When`, optional sections as needed), then attach the round narrative there and add the operator to the new card’s **NPUKernelBench field inventory**.
   - After all targeted pattern synthesis is complete, rewrite that tree’s `references/pattern_index.md` manually as a compact quick-match index (one `##` section per pattern, concise 4-5 line actionable summary, emphasis on when to use).
   - Do not auto-generate `pattern_index.md` via Python or other scripts.

7. **Gap-filling audit pass (required before synthesis)**

   After completing one full kernel pass and before synthesis:

   - For every touched pattern card, check that each operator listed in `Operator workspaces (deduped)` is represented on the card with real round evidence.
   - Prioritize cards with fewer inventory items first; they are easiest to fully close.
   - Add missing narratives grounded in real rounds (success, failure, or anti-signal); do not leave inventory-only operators undocumented.

   Then run a strict second check:

   - Every inventory operator must have at least one explicit five-field entry (single round **or** merged consecutive same-pattern span) with:
     - `Kernel / round / parent`
     - `Pre-change scenario`
     - `Change`
     - `Evidence`
     - `Interpretation`
   - Inventory bullets, cross-card references, or generic notes do not satisfy this requirement.

8. **Per-pattern synthesis (default, after round narratives and gap-filling)**

   Unless the task explicitly skips synthesis, for each pattern touched in the batch add a synthesis section that combines narratives across kernels:

   - What preconditions made this pattern succeed.
   - Common failure modes and anti-signals.
   - Whether the pattern should be tried early, late, or only after profiling.
   - Practical "try / avoid" verdict that refines `## Use When` and `## Avoid When`.
   - A recommended application order when evidence shows staged success (for example structural branch cleanup first, bounded threshold/tile tuning second).

   Synthesis quality rules:

   - **Kernel-agnostic phrasing:** avoid naming one operator’s branch labels as if they are universal; rewrite into general semantic regimes.
   - **Broad experience coverage:** synthesize from successes, failures, and near-miss branches across available narratives on the card.
   - **Abstraction:** keep guidance independent from specific round identifiers and concrete file paths in the final pattern prose.
   - **Explanatory clarity:** avoid unexplained jargon; define specialized terms briefly when first introduced.
   - **Self-contained output:** a reader should understand when/how to apply the pattern without reopening raw logs.

  When the user requests **round narratives only**, stop after step 7 and leave synthesis for a later pass.

9. **Mark progress**

   - Set that kernel’s **Kernel log review** and **Pattern card follow-up** columns to `done` in **`{bench_export_root}/PATTERN_AND_LOG_SYNC_PROGRESS.md`** when the narrative summary is written and cards are updated (create that file at the bench root first if it was missing; see **Progress tracking** above). **Edit only that operator’s row** in a single intentional change—do not script or batch-flip `todo` → `done` across the table (see **Forbidden** under **Progress tracking**).

## Automation vs manual narrative

- Automated scans can append **citation inventories** (which operators linked a card in `attempts.md`). Treat that as bookkeeping, not a substitute for semantic mapping and narrative analysis.
- **Do not** use automation to set **`PATTERN_AND_LOG_SYNC_PROGRESS.md`** review columns from `todo` to `done`; that is always a **manual, per-kernel** step after the real pass (see **Progress tracking**).
- **Scans must see gitignored bench trees:** any script or shell loop that walks a bench export under `workspace/` (for example `workspace/NPUKernelBench_level_1_2_triton` or `workspace/kernelagent_v3`) must use **`find`**, **`rg --no-ignore`** (path restricted to that export root), or equivalent—not plain **`rg`** / **git-aware** search alone—or it may find **zero** `attempts.md` files while the tree exists on disk.
- Do not claim a pattern “worked” globally from a citation alone; ground claims in round outcomes, code diffs, and profile/perf evidence.

## Related files

- [kernel-bench-layout.md](kernel-bench-layout.md) for filesystem semantics.
- `docs/notes/2026-04-29-optimize-pattern-card-authoring.md` for pattern card structure and index regeneration.
