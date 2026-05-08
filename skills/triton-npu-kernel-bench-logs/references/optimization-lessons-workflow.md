# From bench logs to `triton-npu-optimize-v2` pattern cards

Use this workflow when distilling `workspace/NPUKernelBench_level_1_2_triton` (or a sibling export) into durable pattern knowledge. **Target only** `skills/triton-npu-optimize-v2/` for pattern edits; leave `skills/triton-npu-optimize/` unchanged unless the project explicitly promotes v2.

## Finding files under `workspace/` (gitignored)

The repo root **`.gitignore` ignores `workspace/`**. Default **git-aware** searches (ripgrep from the repo without flags, many IDE search panels, and some glob/listing tools) **often skip `workspace/`**, which can look like “no `opt-round-*` / no `attempts.md`” even when the bench export exists.

**Always** use discovery that includes gitignored paths before concluding a kernel has no record:

- **Read** known paths by **full absolute path** (not skipped due to `.gitignore`).
- Shell: **`find workspace/NPUKernelBench_level_1_2_triton …`** (POSIX `find` does not read `.gitignore`).
- Shell: **`rg --no-ignore '…' workspace/NPUKernelBench_level_1_2_triton`** when you need content search under the bench tree.

Do not treat an empty git-aware repo search as proof the tree is missing. See `skills/triton-npu-kernel-bench-logs/SKILL.md` (section **Locating bench trees**) for the full rule.

## Core objective

Learn more precisely when and how each optimization pattern should or should not be used, based on round-level evidence. Discover and codify new patterns when existing cards are not a semantic fit.

## Progress tracking

Maintain `workspace/NPUKernelBench_level_1_2_triton/PATTERN_AND_LOG_SYNC_PROGRESS.md` and optional narrative digests such as `workspace/NPUKernelBench_level_1_2_triton/PILOT_KERNEL_DIGESTS.md` when batching multiple kernels:

- One row per numbered operator directory.
- Mark **Has opt record** `no` when `opt-note.md` is missing or there are zero `opt-round-*` directories.
- Keep **Kernel log review** and **Pattern card follow-up** as `todo` until that kernel’s pass is finished.

## Per-kernel pass (only when an optimization record exists)

For each operator workspace `NN_OperatorName/` that has `opt-note.md` and `opt-round-*`:

1. **Context**
   - Read `NN_OperatorName.py` (PyTorch reference) and `NN_OperatorName.json` (cases).
   - Read `triton_NN_OperatorName.py` (initial Triton) and `baseline/` mirror for the same snapshot.
   - Skim `NN_OperatorName_perf.txt` for PyTorch `raw-op-statistic-case-*` timings when you need reference timings.

2. **Guided round walk**
   - Open `opt-note.md` and walk rounds in order (`## Round 1`, `## Round 2`, …).
   - For each round, open `opt-round-N/attempts.md` for hypotheses, cited pattern cards, analysis level, and failures; open `opt-round-N/summary.md` for the compact outcome; read `round-state.json` when paths or statuses need confirmation.
   - For each round, read `skills/triton-npu-optimize-v2/references/pattern_index.md` and choose the best semantic match (one primary pattern and optional secondary pattern).
   - Do not map patterns by keyword search alone. Use kernel structure, code change intent, and evidence from perf/profile logs.
   - If no existing pattern in the index matches the round mechanism, mark the round as "new pattern candidate".

3. **Code diff discipline**
   - Compare Triton **before vs after** the round: parent candidate (usually prior round `opt_triton_*.py`, else `baseline/`) vs this round’s `opt_triton_*.py`.
   - Tie observed code motion back to the cited pattern (what matched the card, what did not).

4. **Kernel-local summary (before touching pattern cards)**

   Capture briefly:

   - Which patterns were **tried** (semantic mapping, not only cited names).
   - What **worked** vs **did not** (correctness, perf, promotion vs not promoted).
   - Any **successful pattern application** with concrete code shape (one or two sentences plus file/round pointers).

5. **Per-round narrative updates (`triton-npu-optimize-v2` only)**

   For each round, append a narrative to the mapped pattern card (or to a new card if no match exists). Include:

   - Kernel and round id (`NN_OperatorName`, `opt-round-N`), plus parent round.
   - Code context before the change (hot path shape and bottleneck symptoms).
   - Exact change made (what was rewritten, tiled, fused, hoisted, hinted, etc.).
   - Observed evidence (profile/perf deltas, correctness outcomes, regressions).
   - Why it worked or failed compared with this card's current guidance.
   - Enough concrete detail to reconstruct the round later without reopening raw log files.
   - Novelty tag:
     - `novel`: adds new boundary/insight; keep detailed.
     - `repetitive`: same lesson as prior narratives; keep short and cross-reference existing narrative bullets.

   **Five-field template (mandatory for every narrated round):** each round block on a pattern card must include all of the following as explicit Markdown bullets with these exact bold labels (same order). Do not merge multiple rounds into one block. Do not replace fields with a single prose paragraph.

   - **`Kernel / round / parent`**
   - **`Pre-change scenario`**: shape/workload context and bottleneck symptoms
   - **`Change`**: exact mechanism and key parameter values or branch guards
   - **`Evidence`**: correctness status plus main perf/profile observations (including promoted vs validated-branch vs not promoted)
   - **`Interpretation`**: why it likely worked or failed for this scenario, relative to current pattern guidance

   If the pattern has no applicable round for that operator, state that under a short heading; do not fabricate a five-field entry.

   The synthesis stage should be able to run from pattern-card narratives alone. Do not rely on future readers reopening `attempts.md` to recover missing context.

   **New cards:** triage text in `attempts.md` only references a small set of legacy pattern paths. Still perform semantic mapping against the full `pattern_index.md`. Create a new v2 card when the mechanism is not covered; omit new cards when all rounds map to existing summaries.

   **Inventory setup requirement (start of card work):**
   - Ensure each touched card has `## NPUKernelBench field inventory`.
   - Ensure it contains `**Operator workspaces (deduped):**` with deduped operator bullets.
   - Treat this inventory as the authoritative checklist for later gap-filling.

6. **Pattern card updates and new pattern creation**

   - If the round fits an existing card under `skills/triton-npu-optimize-v2/references/patterns/`, update that card with the narrative and any refined guidance.
   - If the round is a "new pattern candidate", create `skills/triton-npu-optimize-v2/references/patterns/<slug>.md` following `docs/notes/2026-04-29-optimize-pattern-card-authoring.md` (`# Title`, required `## Summary` and `## Use When`, optional sections as needed), then attach the round narrative there.
   - After **any** pattern card edit, regenerate the index:

     ```bash
     python3 skills/triton-npu-optimize/scripts/build_pattern_index.py \
       --patterns-dir skills/triton-npu-optimize-v2/references/patterns \
       --output skills/triton-npu-optimize-v2/references/pattern_index.md
     ```

7. **Gap-filling audit pass (required before synthesis)**

   After completing one full kernel pass and before synthesis:

   - For every touched pattern card, check that each operator listed in `Operator workspaces (deduped)` is represented in the card narratives.
   - Prioritize cards with fewer inventory items first; they are easiest to fully close.
   - Add missing narratives grounded in real rounds (success, failure, or anti-signal); do not leave inventory-only operators.

   Then run a strict second check:

   - Every inventory operator must have at least one explicit five-field entry with:
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

  When the user requests **round narratives only**, stop after step 7 and leave synthesis for a later pass.

9. **Mark progress**

   - Set that kernel’s **Kernel log review** and **Pattern card follow-up** columns to `done` in `PATTERN_AND_LOG_SYNC_PROGRESS.md` when the narrative summary is written and cards are updated.

## Automation vs manual narrative

- Automated scans can append **citation inventories** (which operators linked a card in `attempts.md`). Treat that as bookkeeping, not a substitute for semantic mapping and narrative analysis.
- **Scans must see gitignored bench trees:** any script or shell loop that walks `workspace/NPUKernelBench_level_1_2_triton` must use **`find`**, **`rg --no-ignore`** (path restricted to `workspace/…`), or equivalent—not plain **`rg`** / **git-aware** search alone—or it may find **zero** `attempts.md` files while the tree exists on disk.
- Do not claim a pattern “worked” globally from a citation alone; ground claims in round outcomes, code diffs, and profile/perf evidence.

## Related files

- [kernel-bench-layout.md](kernel-bench-layout.md) for filesystem semantics.
- `docs/notes/2026-04-29-optimize-pattern-card-authoring.md` for pattern card structure and index regeneration.
