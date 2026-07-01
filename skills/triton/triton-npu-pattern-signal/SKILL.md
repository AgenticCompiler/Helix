---
name: triton-npu-pattern-signal
description: Use when you need to identify strong signal features for optimization patterns after operator optimization is complete. Correlates profiling data (report.txt) with patterns to determine which feature value ranges indicate a pattern should be applied.
---

# Pattern Signal Analysis

## Purpose

After operator optimization is complete, analyze **parent-round profiling data** to identify which features, when observed in a kernel's profile, are strong signals telling you **"you should apply specific Pattern X"**.

**Critical distinction**: The analysis is about the PARENT round's report.txt — not the child's, not the delta. A "signal" is something you observe in your current kernel's profile that predicts what pattern to apply next.

## Core Concept

- Analyze the **PARENT** round's report.txt (the state BEFORE pattern was applied) — these values are the SIGNALS
- The child round's report.txt serves only as verification that the pattern addressed the bottleneck
- For each feature in the parent: ask **"Does this value indicate a bottleneck that pattern P specifically fixes?"**
- **Do NOT** analyze how much a feature changed — that's an EFFECT, not a signal
- Use **logical reasoning**: connect feature values → hardware bottleneck → pattern's mechanism → why it works

## Inputs

| File | Location | Purpose |
|------|----------|---------|
| `opt-note.md` | Workspace root | Identify effective optimization rounds |
| `opt-round-N/summary.md` | Workspace opt-round-N | Pattern used (Selected Pattern Direction), hypothesis, code changes |
| `opt-round-N/attempts.md` | Workspace opt-round-N | Alternative: pattern used (Selected pattern field) |
| `{parent}/extracted_bin_data/report.txt` | Parent round (baseline or opt-round-(N-1)) | **THE SOURCE OF SIGNALS** — features before pattern was applied |
| `opt-round-N/extracted_bin_data/report.txt` | Workspace opt-round-N | Verification: did the pattern fix the bottleneck? |
| `opt-round-N/opt_triton_xxx.py` (or .py file) | Workspace opt-round-N | Code change reference for logical reasoning |

**report.txt locations**: Check both `extracted_bin_data/report.txt` and `OPPROF_*/simulator/extracted_bin_data/report.txt`. The `extracted_bin_data/` directory may be directly under the operator dir, under `baseline/`, or under `opt-round-N/`.

---

## report.txt Format

See [`references/report-txt-format.md`](references/report-txt-format.md) for complete field format of every section, feature name mapping table, and special notes (e.g., CACHEMISS only in Per-Core, VECTOR Unit label is `Top instr types:` not `Top-conflict instrs`).

---

## Feature Inventory (from report.txt)

### Tier 1: Global Features

#### A. Pipe Distribution (6 features in global, +CACHEMISS only in per-core)

| Feature | report.txt Line | Description |
|---------|----------------|-------------|
| %(SCALAR) | `SCALAR instr=... cycles%` | Scalar pipe occupancy (address calc, div/mod, control flow) |
| %(VECTOR) | `VECTOR instr=... cycles%` | Vector pipe occupancy (SIMD compute) |
| %(MTE2) | `MTE2 instr=... cycles%` | MTE2 DMA load pipe occupancy |
| %(MTE3) | `MTE3 instr=... cycles%` | MTE3 DMA store pipe occupancy |
| %(ALL) | `ALL instr=... cycles%` | All-pipe parallel execution barrier occupancy |
| %(FLOWCTRL) | `FLOWCTRL instr=... cycles%` | Flow control pipe occupancy |
| *(CACHEMISS)* | *Per-core only: `%(CACHEMISS): dur=N%`* | Cache miss duration % — only in per-core, not in global [Pipe Distribution] |

#### B. Key Ratios (from [Key Ratios] section)

| Feature | report.txt Line | Description |
|---------|----------------|-------------|
| SCALAR:VECTOR_instr | Ratio | Scalar-to-vector instruction ratio — high = under-vectorized |
| SCALAR:VECTOR_cycles | Ratio | Scalar-to-vector cycle ratio — high = scalar-bottlenecked |
| SCALAR_instr_pct | Percentage | % of instructions that are SCALAR |
| SCALAR_cycles_pct | Percentage | % of cycles in SCALAR pipe |
| VECTOR_instr_pct | Percentage | % of instructions that are VECTOR |
| VECTOR_cycles_pct | Percentage | % of cycles in VECTOR pipe |
| MTE2_instr_pct | Percentage | % of instructions that are MTE2 (DMA load) |
| MTE2_cycles_pct | Percentage | % of cycles in MTE2 pipe |

#### C. VECTOR Unit Details (from [VECTOR Unit] section)

| Feature | Description |
|---------|-------------|
| UB Read Conflict | Unified Buffer read bank conflicts — high = bank conflicts stalling vector pipe |
| UB Write Conflict | Unified Buffer write bank conflicts |
| UB Conflict Total | Total UB conflicts |
| Utilization avg/min/max | Vector unit utilization (samples) |
| Top instr types | Dominant VECTOR instruction types (always shown regardless of conflict count) |

#### D. TRACE Events (from [TRACE Events] section)

| Feature | Description |
|---------|-------------|
| Total events | Total trace events — proxy for kernel complexity |
| DIV count | Division events — per-element coordinate decode signal |
| REM count | Remainder/modulo events — per-element coordinate decode signal |
| Arithmetic events % | (SIGNEXT+ADD+MUL+DIV+SUB+MADD) / total — compute intensity |
| Arithmetic breakdown | Per-type counts — DIV high = coordinate decode; MUL/ADD high = compute |

#### E. Synchronization (from [WAIT_FLAG / BAR Sync] and [Pipeline Flows])

| Feature | Description |
|---------|-------------|
| WAIT_FLAG total | Total wait-for-flag synchronizations |
| BAR total | Total barrier synchronizations |
| Pipeline Flows counts | SCALARToVECTOR, VECTORToMTE3, MTE2ToMTE3, etc. — inter-pipe communication |

#### F. Pipe Overlap Ratio (18 features — in [Pipe Overlap Ratio] section)

| Feature Name | Description |
|--------------|-------------|
| %(SCALAR&MTE2/SCALAR) | How much SCALAR time overlaps with MTE2 (relative to SCALAR) |
| %(SCALAR&MTE2/MTE2) | How much MTE2 time overlaps with SCALAR (relative to MTE2) |
| %(SCALAR&VECTOR/SCALAR) | How much SCALAR time overlaps with VECTOR (relative to SCALAR) |
| %(SCALAR&VECTOR/VECTOR) | How much VECTOR time overlaps with SCALAR (relative to VECTOR) |
| %(SCALAR&MTE3/SCALAR) | SCALAR overlap with MTE3 (relative to SCALAR) |
| %(SCALAR&MTE3/MTE3) | MTE3 overlap with SCALAR (relative to MTE3) |
| %(MTE2&VECTOR/MTE2) | MTE2 overlap with VECTOR (relative to MTE2) |
| %(MTE2&VECTOR/VECTOR) | VECTOR overlap with MTE2 (relative to VECTOR) |
| %(MTE2&MTE3/MTE2) | MTE2 overlap with MTE3 — high = load-store pipelined |
| %(MTE2&MTE3/MTE3) | MTE3 overlap with MTE2 |
| %(VECTOR&MTE3/VECTOR) | VECTOR overlap with MTE3 |
| %(VECTOR&MTE3/MTE3) | MTE3 overlap with VECTOR |
| %((VECTOR+CUBE)&SCALAR/(VECTOR+CUBE)) | (V+C) overlap with SCALAR (relative to V+C) |
| %((VECTOR+CUBE)&SCALAR/SCALAR) | SCALAR overlap with (V+C) (relative to SCALAR) |
| %((VECTOR+CUBE)&MTE2/(VECTOR+CUBE)) | (V+C) overlap with MTE2 |
| %((VECTOR+CUBE)&MTE2/MTE2) | MTE2 overlap with (V+C) |
| %((VECTOR+CUBE)&MTE3/(VECTOR+CUBE)) | (V+C) overlap with MTE3 |
| %((VECTOR+CUBE)&MTE3/MTE3) | MTE3 overlap with (V+C) |

### Tier 2: Per-Core Features

#### G. Pipe Distribution Over Each Core

Each core has 7 features (SCALAR, VECTOR, MTE2, MTE3, ALL, FLOWCTRL, CACHEMISS). **CACHEMISS only appears here, not in global.** Core naming: `coreN.veccoreM` (N=0..31, M=0..1). **Summarize as patterns**, not individual values:
- "Uniform across all cores: feature ~X%"
- "Bi-modal: cores 0-3 at X%, cores 4-15 at Y%"
- "Single outlier: core5 at 2x others"

#### H. Pipe Overlap Ratio Over Each Core

Same 18 overlap features, per-core. **Summarize as patterns**.

---

## Workflow

### Step 1: Identify Effective Optimization Rounds

Read `opt-note.md` and filter for effective optimization rounds.

**Criteria**:
- Rounds with `Best status: best` or `Best status: current best`
- Or rounds with `Best status: superseded` but significant performance improvement (> 5%)

**Output**: List of effective rounds e.g. `[1, 2, 3]`

### Step 2: Associate Patterns with Rounds

For each effective round N, read `opt-round-N/summary.md` and extract:
- `Selected Pattern Direction` field → pattern name
- `Optimization Hypothesis` → what the pattern does
- `Code Changes` → what code was modified
- `Benchmark Comparison` → speedup achieved

If `summary.md` is missing, read `opt-round-N/attempts.md` for `Selected pattern:` field.

**Output**: Round→Pattern mapping with hypothesis summary.

```
Round 1 → padded_row_col_copy (2D tiling to replace 1D div/mod indexing)
Round 2 → scalar-latency-traps (constexpr shapes, branch elimination)
Round 3 → exact-tile-no-boundary-fast-path (unconditional load/store path)
```

### Step 3: For Each Pattern, Analyze Parent Features as PREDICTIVE SIGNALS

**THIS IS THE CORE STEP. Analyze the PARENT round's report.txt, not the child's changes.**

For each effective round N with pattern P:
- **Parent** = round N-1 (or baseline if N=1)
- The parent's report.txt contains the SIGNALS that predict pattern P should be used
- The child's report.txt (round N) only VERIFIES that P addressed the bottleneck

#### 3.1 Read Parent and Child report.txt

Read both files and extract all feature values.

#### 3.2 Read the Pattern's Hypothesis

From `opt-round-N/summary.md`, understand:
- What code change was made
- What bottleneck it targets
- What the expected effect is

#### 3.3 Analyze Each Feature in the PARENT as a Signal

For every feature in the parent's report.txt, ask this structured question:

> **"Looking at the PARENT's value of feature F, and understanding what pattern P does (from summary.md), does F's value indicate the bottleneck that P fixes?"**

The analysis must follow this logical chain:
1. **Feature value** → what hardware behavior does it indicate?
2. **Hardware behavior** → what code pattern caused it?
3. **Code pattern** → does pattern P specifically target this code pattern?
4. If yes → this feature is a **SIGNAL** for pattern P

**Example (correct analysis)**:

Pattern: `padded_row_col_copy` (Summary: "Replace flat 1D indexing with vector div/mod by 2D row-column tiling where div/mod becomes scalar from tl.program_id")

| Feature | Parent Value | Parent Value Indicates | Is Signal for padded_row_col_copy? | Logical Reasoning |
|---------|-------------|----------------------|----------------------------------|-------------------|
| SCALAR cycles% | 97.9% | Almost all time in SCALAR pipe — address calculation dominates | **YES** | padded_row_col_copy eliminates per-element div/mod address computation (the primary SCALAR consumer), replacing it with scalar program_id-based tiling |
| DIV events | 2049 | Thousands of DIV instructions — per-element coordinate decode | **YES** | Each DIV is one coordinate component. padded_row_col_copy replaces vector DIV with scalar program_id computation, reducing DIV count |
| MTE2 cycles% | 0.0% | DMA load pipe completely idle — no coalesced memory access | **YES** | 1D flat indexing prevents block-pointer/DMA usage. 2D tiling enables tl.make_block_ptr for coalesced loads via MTE2 |
| VECTOR cycles% | 1.6% | Vector pipe nearly idle despite 21% vector instructions — stalled by scalar | **YES** | VECTOR instructions are stalled waiting for SCALAR-produced addresses. Reducing SCALAR work unblocks VECTOR |
| SCALAR&VECTOR/SCALAR | 0.15% | Almost zero overlap — SCALAR and VECTOR are serialized | **YES** | VECTOR can't start until SCALAR finishes address computation. 2D tiling lets them overlap |
| %(SCALAR&MTE2/SCALAR) | 0.00% | No DMA happening during SCALAR | NO | This is a consequence of MTE2=0%, not an independent signal |
| %(VECTOR&MTE3/VECTOR) | 0.00% | No vector-store overlap | NO | Not directly related to div/mod elimination |
| Arithmetic events% | 68.0% | High compute-to-total ratio | NO | This is a consequence, not a specific signal for padded_row_col_copy |

**Example (INCORRECT — what NOT to do)**:

| Feature | Parent | Child | Delta | Signal? |
|---------|--------|-------|-------|---------|
| SCALAR cycles% | 97.9% | 56.6% | -41.3pp | YES |

This is wrong because it just describes the change. The PARENT value (97.9%) is the signal, not the delta.

#### 3.4 Analyze ALL Features — No Exceptions

**Every feature from the report.txt must be analyzed.** DO NOT skip features or filter by arbitrary thresholds. The output table must include every global feature (Pipe Distribution, Key Ratios, VECTOR Unit, TRACE Events, Pipeline Flows, Pipe Overlap Ratio = ~40+ features).

Features with no logical connection to the pattern should be marked `NO` in the Signal column with a brief explanation of why not.

#### 3.5 Two-Level Analysis

**Level 1**: Global features extracted as specific values with logical reasoning per feature.

**Level 2**: Per-core features described as PATTERNS (not individual values per core). Describe:
- Is the pattern uniform across cores? If not, which cores differ?
- Does the pattern itself indicate a bottleneck that P fixes?
- "All cores 0% MTE2" is a signal; "cores 0-3 at 5%, cores 4-7 at 6%" is NOT informative

#### 3.6 Write Consolidated Signal Summary for Each Pattern

After completing the per-feature analysis (global + per-core), synthesize ALL "YES" signals into a **consolidated signal summary** for the pattern.

**Purpose**: This summary becomes the pattern's "signal signature" for skill self-evolution. Future optimizers use it for pattern matching decisions without re-analyzing raw data.

**How to synthesize** — group related YES signals into behavioral groups:

1. **Identify signal clusters** — look at the YES features and identify clusters of related behaviors:
   - Compute-related: `SCALAR cycles%`, `DIV events`, `REM events`, `SIGNEXT events`, `Arithmetic events%`, `SCALAR:VECTOR_cycles`, `SCALAR_instr_pct`, `VECTOR_instr_pct`, `SCALAR Instr Types`
   - DMA-related: `MTE2 cycles%`, `MTE3 cycles%`, `SCALAR&MTE2/SCALAR`, `SCALAR&MTE3/SCALAR`, `MTE2&MTE3/MTE2`, `MTE2&MTE3/MTE3`, `Pipeline Flows`, `ProcessBytes`
   - Sync-related: `WAIT_FLAG total`, `BAR total`, `ALL cycles%`
   - Pipe-overlap-related: All 18 overlap ratio features
   - Parallelism-related: Per-core patterns, core count changes, load distribution

2. **Merge within each cluster** — combine features that describe the same root cause:
   - "SCALAR cycles high + DIV count high + REM count high + SIGNEXT high + Arithmetic% high" → one group about "coordinate decoding dominates execution"
   - "MTE2 near-zero + MTE3 near-zero + SCALAR&MTE2 zero + SCALAR&VECTOR zero" → one group about "DMA idle and pipes serialized"

3. **Write each group as a behavioral description** (natural language, NOT numeric):
   - What the profile behavior looks like (merged symptom)
   - What code pattern causes it (root cause)
   - Why this pattern fixes it (mechanism)
   - Which original features belong to this group (covered features)

4. **Output format** — append to each pattern's section in `pattern_signal.md` as `### Signal Summary — Consolidated Strong Signals for [Pattern Name]` (see template in Step 4).

**Quality checks**:
- Every YES feature from both global and per-core analysis is covered by at least one group
- No two groups describe overlapping behaviors (check for redundancy)
- Each group reads as a standalone pattern-matching rule
- Descriptions use behavioral language ("dominates", "nearly idle", "extremely numerous") not numeric thresholds

### Step 4: Output Pattern Signal Table

Output to `pattern_signal.md` in the operator's root directory.

**CRITICAL RULE**: The file is organized by PATTERN, not by round. Each pattern gets its own section showing signals from ALL rounds where it was used. If a pattern was used in multiple rounds (across operators), aggregate the signal information.

Full output format template at [`references/pattern-signal-output-format.md`](references/pattern-signal-output-format.md), including file structure, Signal Summary format, Pattern Signal Summary table, Bottleneck Evolution Diagram, and full padded_row_col_copy example.

## Important Notes

### The Signal vs Effect Distinction

| Concept | Definition | Example |
|---------|-----------|---------|
| **Signal** | A feature value in the PARENT that predicts you should use pattern P | Parent has SCALAR cycles = 97.9% → "I should use padded_row_col_copy because it eliminates div/mod overhead" |
| **Effect** | How a feature value changed after applying P | "SCALAR cycles dropped by 41.3pp" — this is the result, not the prediction |
| **Verification** | The child value confirms the signal was correct | Child SCALAR cycles = 56.6% confirms the bottleneck was addressed |

**Always analyze signals from the parent's perspective.** The child is only for verification.

### Logical Reasoning Chain

Every "YES" in the Signal column MUST include this full reasoning chain:
1. **Feature value → what it means**: "SCALAR cycles = 97.9% means almost all time is spent in scalar instructions"
2. **What it means → code bottleneck**: "This indicates the kernel is doing per-element coordinate computation (div/mod) in a 1D loop"
3. **Bottleneck → how pattern fixes it**: "padded_row_col_copy specifically replaces per-element vector div/mod with scalar program_id-based 2D tiling"
4. **Pattern mechanism → expected outcome**: "This moves coordinate computation from SCALAR's hot loop to one-time scalar setup"

### Cross-Pattern Aggregation

When the same pattern appears in multiple operators (e.g., padded_row_col_copy in Cat, Split, Pad, RepeatInterleave):

**Organization rule**: Aggregate by PATTERN, not by signal feature. For each pattern, summarize its signal profile from all operators where it was effective.

For each pattern used across operators, produce:

#### [Pattern Name] — Cross-Operator Signal Profile

| Feature | Op1 Parent Value | Op2 Parent Value | Op3 Parent Value | Consistent Range | Signal Strength |
|---------|-----------------|-----------------|-----------------|-----------------|-----------------|
| SCALAR cycles% | 97.9% | ... | ... | > 90% | STRONG |
| ... | ... | ... | ... | ... | ... |

**Conclusion**: If SCALAR cycles > 90% in ALL cases where padded_row_col_copy was effective → that's a strong universal signal. If some features vary widely → note them as context-dependent (operator-specific) signals.

**When the pattern spans only a single operator**: Still produce the per-pattern signal summary (from the `## Pattern Signal Summary` section above), but omit the cross-operator table — each pattern section in `Pattern Signal Summary` serves as the authoritative signal profile for future use.

### Per-core Analysis Rules

1. **CACHEMISS** only exists in per-core, NOT in global Pipe Distribution
2. Core naming is `coreN.veccoreM` — N is physical core (0-31), M is vector core (0-1)
3. Single-core runs: report shows only 1 core. Multi-core: shows all active cores
4. Focus on whether the per-core PATTERN (uniform vs uneven) is meaningful, not individual values
5. Multi-core activation (from 1→N cores) is a significant pattern change

### Feature Sections NOT to Skip

Beyond the 18 Pipe Overlap Ratios and 6 Pipe Distribution features, these sections in report.txt also contain signal-relevant data:

- **[Key Ratios]**: Quick SCALAR:VECTOR ratios
- **[VECTOR Unit]**: UB conflicts → vector pipeline stalls
- **[MTE2 Data Transport]**: Data movers count → DMA utilization
- **[WAIT_FLAG / BAR Sync]**: Synchronization overhead
- **[Pipeline Flows]**: Inter-pipe communication patterns
- **[SCALAR Instr Types]**: Dominant scalar instructions
- **[TRACE Events]**: DIV/REM counts, arithmetic breakdown
- **[Simulator Runtime]**: Runtime statistics

### What "Not a Signal" Means

When marking a feature as `NO` in the Signal column, briefly explain why:
- "Feature X is NOT a signal because it reflects a consequence, not a cause (e.g., it's always near-zero for this operator type)"
- "Feature X does not relate to what pattern P changes (e.g., it tracks MTE3 store behavior while P only affects compute)"
- "Feature X value is within normal range for healthy kernels"

## Relationship with Other Skills

- `triton-npu-optimize`: This skill runs after it completes, analyzing optimization results
- The output `pattern_signal.md` serves as input for the Pattern knowledge base
