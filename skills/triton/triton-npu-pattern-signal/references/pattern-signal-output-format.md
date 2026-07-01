# pattern_signal.md Output Format Template

Output to `pattern_signal.md` in the operator's root directory.

**Critical rule**: Organized by **PATTERN**, not by round. When the same pattern appears across multiple rounds, aggregate signal information.

---

## File Structure

```markdown
# Pattern Signal Analysis — [Operator Name]

## Operator Summary
- Total effective rounds: N
- Best round: R[N] (+X% speedup)
- Pattern sequence: R1→P1, R2→P2, ...

## [Pattern Name 1] — Round N

### What It Does
[Extracted from summary.md's Optimization Hypothesis]

### Code Change
[Extracted from summary.md's Code Changes]

### Performance
Baseline: X us → Round N: Y us (speedup: Zx)

### PARENT Features as Predictive Signals

Analyzing the PARENT round's report.txt — these are the features that, when observed,
indicate [Pattern Name] should be applied.

| Feature | Parent Value | What Parent Value Indicates | Signal? | Logical Reasoning |
|---------|-------------|---------------------------|---------|-------------------|
| SCALAR cycles% | 97.9% | ... | YES | [specific reasoning] |
| ... | ... | ... | ... | ... |

### PER-CORE Patterns as Signals

| Feature Pattern | Parent Pattern | Signal? | Reasoning |
|----------------|---------------|---------|-----------|
| SCALAR cycles% per-core | Single-core only | YES | ... |
| MTE2 cycles% per-core | All cores 0% | YES | ... |

### Verification (Child round results)

Brief check that the pattern worked:
- SCALAR cycles%: 97.9% → 56.6% (signal confirmed: bottleneck relieved)
- ...

### Signal Summary — Consolidated Strong Signals for **[Pattern Name]**

**CRITICAL**: This is the pattern's "signal signature" for skill self-evolution.
It consolidates all YES signals into reusable behavioral descriptions.

See SKILL.md Step 3.6 for writing rules. Template for each signal group:

- **Group: [Short descriptive name]**
  - **Profile behavior**: [Language description merging related features]
  - **Root cause**: [Code pattern that causes this profile behavior]
  - **Apply [Pattern Name] because**: [How the pattern resolves this root cause]
  - **Covered features**: `[comma-separated feature names]`

**Prerequisites**: [Any patterns that must be applied first]
**Common variants**: [Notable variants of this pattern]

## [Pattern Name 2] — Round N
... (same structure)

## Pattern Signal Summary

**CRITICAL**: Organized by PATTERN. Each entry merges signals from ALL rounds/operators
where it was effective. Use behavioral language, not numeric ranges.

### [Pattern Name]

| Property | Value |
|----------|-------|
| Speedup range | e.g., 2.0x–3.5x (across operators) |
| Bottleneck targeted | [e.g., "excessive per-element coordinate decoding via div/mod"] |
| Mechanism | [what the pattern does at code level] |
| Code change | [1-line summary] |

#### Signal Signature (language descriptions)

**Primary behavioral signals** (MUST observe these):
- **[Group Name]**: [Behavioral description]. Covered features: `[...]`.

**Supporting behavioral signals** (reinforce diagnosis but insufficient alone):
- **[Group Name]**: [Behavioral description]. Covered features: `[...]`.

**Per-core behavioral signals**:
- **[Group Name]**: [Behavioral description]. Covered features: `[...]`.

**When NOT to use this pattern** (counter-indications):
- [Behavioral description], because [reason].

**Post-pattern bottleneck shift**: [What becomes the new bottleneck after applying this pattern]

**Prerequisites**: [Conditions that must hold]

---

### Bottleneck Evolution Diagram

```
Baseline: [pipe] dominates — [behavioral description]
   ↓ R1: [pattern name] → [what changed]
Round 1: [new pipe] dominates — [new bottleneck description]
   ↓ R2: [pattern name] → [what changed]
     ...
```

---

## Full Example: padded_row_col_copy

### Signal Summary — Consolidated Strong Signals for **padded_row_col_copy**

The following behavioral signal groups, when observed in a kernel profile, indicate that
**padded_row_col_copy** (2D row-column tiling) should be applied:

- **Group: Per-element coordinate decoding dominates compute**
  - **Profile behavior**: SCALAR pipe consumes the vast majority of execution cycles, with
    DIV/REM/SIGNEXT trace events dominating the instruction stream and arithmetic events
    comprising a high percentage of all trace events. The SCALAR:VECTOR cycle ratio is
    extreme, and SCALAR instructions far outnumber VECTOR instructions.
  - **Root cause**: The kernel uses 1D flat indexing (`pid * BLOCK + arange`) over the
    total element count, requiring vector div/mod operations to reconstruct
    multi-dimensional coordinates for each element on the hot path.
  - **Apply padded_row_col_copy because**: It decomposes the problem into a 2D grid
    (rows x col_tiles), computing row/col from scalar `program_id(0)` and `program_id(1)`
    outside the per-element hot loop, eliminating vector div/mod entirely.
  - **Covered features**: `SCALAR cycles%`, `DIV events`, `REM events`, `SIGNEXT events`,
    `Arithmetic events%`, `SCALAR:VECTOR_cycles`, `SCALAR_instr_pct`, `VECTOR_instr_pct`,
    `SCALAR Instr Types: MOV_XD_IMM`

- **Group: DMA pipes idle, pipes serialized**
  - **Profile behavior**: MTE2 and MTE3 cycle percentages are near-zero, SCALAR-to-MTE2
    overlap is zero, and SCALAR-to-VECTOR overlap is near-zero. VECTOR pipe is nearly
    idle despite having a non-trivial instruction count, and VECTOR utilization is very low.
  - **Root cause**: 1D flat indexing prevents generating block pointers for coalesced
    DMA loads/stores. VECTOR execution is stalled waiting for SCALAR to produce addresses.
  - **Apply padded_row_col_copy because**: 2D tiling with contiguous tiles enables
    coalesced memory access via block pointers, activating MTE2/MTE3 DMA and breaking
    the SCALAR to VECTOR serial dependency.
  - **Covered features**: `MTE2 cycles%`, `MTE3 cycles%`, `VECTOR cycles%`,
    `SCALAR&MTE2/SCALAR`, `SCALAR&VECTOR/SCALAR`, `VECTOR utilization avg`,
    `VECTOR_instr_pct`

- **Group: Single-core execution, no multi-core parallelism**
  - **Profile behavior**: Only one vector core appears in the per-core detail section.
  - **Root cause**: 1D flat indexing uses a single grid axis (`program_id(0)` only).
  - **Apply padded_row_col_copy because**: 2D grid decomposition
    (`grid = (num_rows, num_col_tiles)`) distributes work across multiple cores.
  - **Covered features**: `Per-core SCALAR cycles%` (single-core pattern),
    `Per-core distribution` (single-entry pattern)

**Prerequisites**: None — typically the FIRST pattern for 1D div/mod-heavy indexing.
**Common variants**: Dimension mapping (row-major vs column-major) depends on memory layout.
