---
name: triton-npu-pattern-signal
description: Use when you need to identify strong signal features for optimization patterns after operator optimization is complete. Correlates profiling data (report.txt) with patterns to determine which feature value ranges indicate a pattern should be applied.
---

# Pattern Signal Analysis

## Purpose

After operator optimization is complete, correlate profiling data features with optimization patterns to identify which data features are strong signals for "should use this Pattern".

## Core Concept

- Compare feature value changes between effective optimization rounds and their parent rounds to identify which features are affected by each Pattern
- **Do NOT rely on change magnitude** to determine if something is a signal
- Use **logical reasoning**: when you see a feature value in a certain range, does it mean you should use this Pattern?

## Inputs

| File | Location | Purpose |
|------|----------|---------|
| `opt-note.md` | Workspace root | Identify effective optimization rounds |
| `opt-round-N/attempts.md` | Workspace opt-round-N | Identify which Pattern was used (Selected pattern field) |
| `opt-round-N/summary.md` | Workspace opt-round-N | Confirm which Pattern was used (Selected Pattern Direction field) |
| `opt-round-N/OPPROF_xxx/simulator/extracted_bin_data/report.txt` | Workspace opt-round-N | Feature data source |
| `opt-round-N/opt_triton_xxx.py` (or corresponding py file) | Workspace opt-round-N | Code change reference |

## report.txt Feature Inventory

### Tier 1: Global Features (25 total) — Direct Value Comparison

#### A. Pipe Distribution (7 features)

| Feature Name | Description |
|--------------|-------------|
| %(SCALAR) | Scalar pipe occupancy |
| %(VECTOR) | Vector pipe occupancy |
| %(MTE2) | MTE2 pipe occupancy |
| %(MTE3) | MTE3 pipe occupancy |
| %(ALL) | ALL pipe occupancy |
| %(FLOWCTRL) | Flow control occupancy |
| %(CACHEMISS) | Cache miss occupancy |

#### B. Pipe Overlap Ratio (18 features)

| Feature Name | Description |
|--------------|-------------|
| %(SCALAR&MTE2/SCALAR) | SCALAR & MTE2 overlap ratio (relative to SCALAR) |
| %(SCALAR&MTE2/MTE2) | SCALAR & MTE2 overlap ratio (relative to MTE2) |
| %(SCALAR&VECTOR/SCALAR) | SCALAR & VECTOR overlap ratio (relative to SCALAR) |
| %(SCALAR&VECTOR/VECTOR) | SCALAR & VECTOR overlap ratio (relative to VECTOR) |
| %(SCALAR&MTE3/SCALAR) | SCALAR & MTE3 overlap ratio (relative to SCALAR) |
| %(SCALAR&MTE3/MTE3) | SCALAR & MTE3 overlap ratio (relative to MTE3) |
| %(MTE2&VECTOR/MTE2) | MTE2 & VECTOR overlap ratio (relative to MTE2) |
| %(MTE2&VECTOR/VECTOR) | MTE2 & VECTOR overlap ratio (relative to VECTOR) |
| %(MTE2&MTE3/MTE2) | MTE2 & MTE3 overlap ratio (relative to MTE2) |
| %(MTE2&MTE3/MTE3) | MTE2 & MTE3 overlap ratio (relative to MTE3) |
| %(VECTOR&MTE3/VECTOR) | VECTOR & MTE3 overlap ratio (relative to VECTOR) |
| %(VECTOR&MTE3/MTE3) | VECTOR & MTE3 overlap ratio (relative to MTE3) |
| %((VECTOR+CUBE)&SCALAR/(VECTOR+CUBE)) | (VECTOR+CUBE) & SCALAR overlap ratio |
| %((VECTOR+CUBE)&SCALAR/SCALAR) | (VECTOR+CUBE) & SCALAR overlap ratio (relative to SCALAR) |
| %((VECTOR+CUBE)&MTE2/(VECTOR+CUBE)) | (VECTOR+CUBE) & MTE2 overlap ratio |
| %((VECTOR+CUBE)&MTE2/MTE2) | (VECTOR+CUBE) & MTE2 overlap ratio (relative to MTE2) |
| %((VECTOR+CUBE)&MTE3/(VECTOR+CUBE)) | (VECTOR+CUBE) & MTE3 overlap ratio |
| %((VECTOR+CUBE)&MTE3/MTE3) | (VECTOR+CUBE) & MTE3 overlap ratio (relative to MTE3) |

### Tier 2: Per-Core Features (800+ total) — Pattern-Based Comparison

#### C. Pipe Distribution Over Each Core (32 cores × 7 features = 224)

Each core has the same 7 features as Pipe Distribution. **Use pattern-based analysis**: describe the distribution pattern across cores, not individual values.

#### D. Special Event Distribution Over Each Core

Typically empty — skip.

#### E. Pipe Overlap Ratio Over Each Core (32 cores × 18 features = 576)

Same 18 features from Pipe Overlap Ratio, distributed across 32 cores. **Use pattern-based analysis**: describe the distribution pattern across cores, not individual values.

**Pattern Description Examples**:
- "Uniform: all cores within 0.1% of mean"
- "Single outlier: core5 is 2x higher than others"
- "Bi-modal: cores 0-7 cluster around 2.1%, cores 8-15 around 1.8%"

## Workflow

### Step 1: Identify Effective Optimization Rounds

Read `opt-note.md` and filter for effective optimization rounds.

**Criteria**:
- Rounds with `Best status: best` or `Best status: current best`
- Or rounds with `Best status: superseded` but significant performance improvement (> 5%)

**Output**: List of effective rounds, e.g., `[1, 2, 3, 4, 5, 6]`

### Step 2: Associate Patterns

For each effective round N, read `opt-round-N/attempts.md` and extract the `Selected pattern:` field. If not found, read `opt-round-N/summary.md` and extract the `Selected Pattern Direction` field. Build "round -> Pattern" mapping.

**Output**:
```
Round 2 -> block-pointer-dimensionality
Round 3 -> autotune (manual)
Round 4 -> parallel
Round 5 -> reorder-load
Round 6 -> autotune (Ascend guidance)
```

### Step 3: Analyze All Features with Logical Reasoning

**IMPORTANT: Do NOT use arbitrary thresholds to filter features. Analyze ALL features.**

For each effective round N (parent is N-1):

#### 3.1 Extract ALL Features

Read both report.txt files and extract:

| Category | Count | What to Extract |
|----------|-------|-----------------|
| [Pipe Distribution] (global) | 7 | All 7 specific values |
| [Pipe Overlap Ratio] (global) | 18 | All 18 specific values |
| [Pipe Distribution Over Each Core] | 32×7=224 | Pattern description per feature |
| [Pipe Overlap Ratio Over Each Core] | 32×18=576 | Pattern description per feature |

#### 3.2 Describe Patterns (Per-Core)

For per-core features, summarize into patterns:
- "Uniform: all cores within X% of mean"
- "Single outlier: core-N is 2x higher"
- "Bi-modal: cores 0-7 cluster at X%, cores 8-15 at Y%"

**Do NOT list 800+ individual values. Describe patterns.**

#### 3.3 Direct Analysis with Logical Reasoning

**For EVERY feature (global and per-core), ask:**

> **"When this feature has a HIGH/LOW value, does it indicate a bottleneck that a specific pattern can fix?"**

**Process ALL 25 global features this way:**

| Feature | Parent Value | Child Value | High Value Indicates | Pattern Signal? |
|---------|-------------|-------------|---------------------|-----------------|
| %(SCALAR) | 44.63% | 1.50% | High SCALAR% = scalar div/mod bottleneck | YES → block-pointer |
| %(MTE3) | 66.32% | 0.81% | High MTE3% = memory bottleneck | YES → block-pointer |
| ... | ... | ... | ... | ... |

**Key principle: Focus on what HIGH/LOW values MEAN, not on how much they changed.**

#### 3.4 Two-Level Output

**Level 1: Global Features** — List all 25 with logical analysis
**Level 2: Per-Core Patterns** — Describe patterns with logical analysis

**Do NOT filter out features. Every feature gets analyzed for its logical meaning.**

Example output:

```
| Feature | Parent | Child | High Value Means | Signal? | Pattern |
|---------|--------|-------|------------------|---------|---------|
| %(SCALAR) | 44.63% | 1.50% | Scalar bottleneck | YES | block-pointer |
| %(MTE3) | 66.32% | 0.81% | Memory bottleneck | YES | block-pointer |
| %(SCALAR&MTE2/SCALAR) | 1.63% | 17.36% | Not a direct signal | NO | — |
| ... | ... | ... | ... | ... | ... |

Per-core patterns:
- SCALAR% per-core: Uniform → Uniform (no pattern change)
- MTE2&MTE3/MTE2 per-core: Uniform high (99%) → Uniform low (0%) → This pattern change is significant
```
| MTE2% per-core | Uniform (mean=1.9%) | Uniform (mean=0.8%) | Mean decreased, pattern similar |

**Note**: Unlike Tier 1, Tier 2 per-core analysis focuses on whether the distribution pattern itself changed, not the absolute values.

### Step 4: Output Pattern Signal Table

Output to `pattern_signal.md` in the workspace root directory.

```markdown
# Pattern Signal Table

## [Pattern Name]

### Successful Cases
- Round N: [before] us -> [after] us ([speedup])

### Feature Analysis (ALL features, logical reasoning)

| Feature | Parent Value | Child Value | High/Low Value Indicates | Signal? | Pattern |
|---------|--------------|-------------|------------------------|---------|---------|
| %(SCALAR) | 44.63% | 1.50% | High SCALAR% = scalar div/mod bottleneck | YES | block-pointer-dimensionality |
| %(MTE3) | 66.32% | 0.81% | High MTE3% = memory bottleneck | YES | block-pointer-dimensionality |
| %(SCALAR&MTE2/SCALAR) | 1.63% | 17.36% | Change not indicative of bottleneck | NO | — |
| ... | ... | ... | ... | ... | ... |

### Per-Core Pattern Analysis

| Feature Pattern | Observation | Signal? | Pattern |
|----------------|------------|---------|---------|
| MTE2&MTE3/MTE2 per-core | R1: uniform >99% across all cores; R2: uniform 0% | YES | High uniform overlap = sequential memory = block-pointer |

### Strong Signals

| Feature | Signal Range | Why It's a Signal |
|---------|-------------|-------------------|
| %(SCALAR) | Notably high | High SCALAR% = scalar bottleneck (div/mod) exists |
| %(MTE3) | Notably high | High MTE3% = memory pipeline overloaded |
| %(SCALAR&MTE3/SCALAR) | Notably high | High overlap = scalar blocking memory ops |
| %(MTE2&MTE3/MTE2) | Very high (>90%) | Very high overlap = sequential memory access |

> **Note:** Signal ranges are descriptive, not precise thresholds. They indicate "notably high" or "very high" based on the observed case. More cases needed to establish reliable thresholds.
```

---

## Important Notes

1. **Analyze ALL features — no filtering by thresholds**:
   - All 25 global features must be analyzed
   - Per-core features described as patterns, not individual values
   - Do NOT use arbitrary thresholds to filter features

2. **Logical reasoning is the only tool**:
   - Ask: "Does a HIGH/LOW value of this feature indicate a bottleneck?"
   - Do NOT ask: "Did this feature change by more than X%?"
   - The question is about what values MEAN, not how much they changed

3. **Signal = "High/Low value tells me to use this pattern"**:
   - High SCALAR% → scalar bottleneck → use block-pointer
   - This is a signal because before optimization, seeing high SCALAR% would tell you to try block-pointer
   - NOT a signal: "SCALAR% dropped by 50%" — this is an effect, not a predictive signal

4. **Per-core patterns are valid signals**:
   - "All cores have >90% MTE2&MTE3 overlap" is a signal
   - This uniform high overlap across cores indicates sequential memory access pattern

5. **Avoid overfitting**: Signals must have logical justification linking the feature to the pattern's effect

## Relationship with Other Skills

- `triton-npu-optimize`: This skill runs after it completes, analyzing optimization results
- `../triton-npu-optimize-knowledge/SKILL.md`: The output `pattern_signal.md` can serve as input for the Pattern knowledge base
- `../triton-npu-analyze-round-performance/SKILL.md`: Provides detailed round-level diagnostics; this skill focuses on signal discovery