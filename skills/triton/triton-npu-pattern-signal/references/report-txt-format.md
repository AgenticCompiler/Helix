# report.txt Format Reference

Verified against real profiling output (Abs, Sum, RepeatInterleave operators, 2026-06-01).

---

## Header

```
========================================================================
Kernel : simulator
Data   : <absolute path to extracted_bin_data>
========================================================================
```

---

## Overall Section

```
========================================================================
Overall
========================================================================
```

### [Pipe Distribution]

```
[Pipe Distribution]  instr count / instr% / cycles / cycles% / dur%
  Total instr: N  |  Total cycles: N
  ALL           instr=N (N%)  cycles=N (N%)  dur=N%
  FLOWCTRL      instr=N (N%)  cycles=N (N%)  dur=N%
  MTE2          instr=N (N%)  cycles=N (N%)  dur=N%
  MTE3          instr=N (N%)  cycles=N (N%)  dur=N%
  SCALAR        instr=N (N%)  cycles=N (N%)  dur=N%
  VECTOR        instr=N (N%)  cycles=N (N%)  dur=N%
```

Pipe names: `ALL`, `FLOWCTRL`, `MTE2`, `MTE3`, `SCALAR`, `VECTOR`.

### [Key Ratios]

```
[Key Ratios]
  SCALAR:VECTOR_instr = N:N = N.N:1
  SCALAR:VECTOR_cycles = N:N = N.N:1
  SCALAR_instr_pct = N.N
  SCALAR_cycles_pct = N.N
  VECTOR_instr_pct = N.N
  VECTOR_cycles_pct = N.N
  MTE2_instr_pct = N.N
  MTE2_cycles_pct = N.N
```

### [VECTOR Unit]

```
[VECTOR Unit]
  Instr count: N
  UB Read Conflict:  N
  UB Write Conflict: N
  UB Conflict Total: N
  Utilization avg/min/max = N.N% / N.N% / N.N%  (samples=N)
  Top instr types:
        N: INSTR_TYPE ...
        N: INSTR_TYPE ...
```

Note: The label is `Top instr types:`, NOT `Top-conflict instrs`. When conflicts are zero, this section still lists the instruction breakdown.

### [MTE2 Data Transport]

```
[MTE2 Data Transport]
  Instr count: N  |  Data movers: N  |  Flow control: N
  ProcessBytes / data mover:  avg=N  min=N  max=N
  ProcessBytes / all MTE2:    avg=N  max=N
  Top instr types:
        N: MOV_SPR_XN SPR:MOV_PAD_VAL,...
        N: MOV_SRC_TO_DST_ALIGN ...
```

### [WAIT_FLAG / BAR Sync]

```
[WAIT_FLAG / BAR Sync]  Totals across all pipes
  WAIT_FLAG total: N  |  BAR total: N
  WAIT_FLAG by pipe: {'PIPE1': N, 'PIPE2': N, ...}
  BAR by pipe: {'PIPE1': N, ...}
```

### [Pipeline Flows]

```
[Pipeline Flows]  category / count / avg_delta(ns) / min / max
  PIPE_A_TO_PIPE_B           count=N  avg=N.Nns  min=N.Nns  max=N.Nns
```

Possible flow categories observed: `MTE2ToVECTOR`, `SCALARToMTE3`, `SCALARToVECTOR`, `VECTORToSCALAR`, `MTE2ToMTE3`, `VECTORToMTE3`.

### [SCALAR Instr Types]

```
[SCALAR Instr Types]
  Instr count: N
        N: INSTR_TYPE ...
```

Each line is `count: INSTR_TYPE` with count right-aligned. Types include MOVK, MOV_XD_IMM, MOV_XD_SPR, ADD, SHL, AND, INSERT_XD, SIGNEXT, LDP_XI_XJ_XN, DIV, MUL, etc.

### [TRACE Events]

```
[TRACE Events]
  Total events: N  |  Complete events(ph=X): N
  Top-20 event names:
           N: EVENT_NAME
           ...
  Arithmetic events (SIGNEXT+ADD+MUL+DIV+SUB+MADD): N / N = N.N%
  Arithmetic breakdown: {'SIGNEXT': N, 'ADD': N, 'MUL': N, 'DIV': N, 'SUB': N, 'MADD': N}
```

Note: `REM` (remainder/modulo) events appear in the Top-20 list but are NOT included in the Arithmetic events formula or breakdown.

### [Special Event Distribution]

```
[Special Event Distribution]
(usually empty — no content lines)
```

### [Simulator Runtime]

```
[Simulator Runtime]
  AVG(runtime): N.N
  MAX(runtime): N.N
  MIN(runtime): N.N
  MAX_DIFF(runtime): N.N
  %(MAX_DIFF/MAX): N.N%
```

### [Pipe Overlap Ratio]

```
[Pipe Overlap Ratio]
  %(SCALAR&MTE2/SCALAR): N.N%
  %(SCALAR&MTE2/MTE2): N.N%
  %(SCALAR&VECTOR/SCALAR): N.N%
  %(SCALAR&VECTOR/VECTOR): N.N%
  %(SCALAR&MTE3/SCALAR): N.N%
  %(SCALAR&MTE3/MTE3): N.N%
  %(MTE2&VECTOR/MTE2): N.N%
  %(MTE2&VECTOR/VECTOR): N.N%
  %(MTE2&MTE3/MTE2): N.N%
  %(MTE2&MTE3/MTE3): N.N%
  %(VECTOR&MTE3/VECTOR): N.N%
  %(VECTOR&MTE3/MTE3): N.N%
  %((VECTOR+CUBE)&SCALAR/(VECTOR+CUBE)): N.N%
  %((VECTOR+CUBE)&SCALAR/SCALAR): N.N%
  %((VECTOR+CUBE)&MTE2/(VECTOR+CUBE)): N.N%
  %((VECTOR+CUBE)&MTE2/MTE2): N.N%
  %((VECTOR+CUBE)&MTE3/(VECTOR+CUBE)): N.N%
  %((VECTOR+CUBE)&MTE3/MTE3): N.N%
```

Exactly 18 features, always in this order.

---

## Per-Core Detail Section

```
========================================================================
Per-Core Detail
========================================================================
```

### [Pipe Distribution Over Each Core]

Each core has its own subsection. 7 features per core: `ALL`, `CACHEMISS`, `FLOWCTRL`, `MTE2`, `MTE3`, `SCALAR`, `VECTOR`.

```
[Pipe Distribution Over Each Core]
  [Pipe Distribution Over coreN.veccoreM]
    %(ALL): instr=N (N%)  cycles=N (N%)  dur=N%
    %(CACHEMISS): dur=N%
    %(FLOWCTRL): instr=N (N%)  cycles=N (N%)  dur=N%
    %(MTE2): instr=N (N%)  cycles=N (N%)  dur=N%
    %(MTE3): instr=N (N%)  cycles=N (N%)  dur=N%
    %(SCALAR): instr=N (N%)  cycles=N (N%)  dur=N%
    %(VECTOR): instr=N (N%)  cycles=N (N%)  dur=N%
```

Key points:
- `CACHEMISS` only appears in per-core (NOT in the global `[Pipe Distribution]`)
- `CACHEMISS` uses only `dur=N%` format (no instr/cycles columns)
- Core naming: `coreN.veccoreM` (N = physical core 0..31, M = vector core 0..1)
- Single-core runs show only one `core0.veccore0` entry; multi-core shows all active cores

### [Special Event Distribution Over Each Core]

```
[Special Event Distribution Over Each Core]
(usually empty)
```

### [Simulator Runtime Over Each Core]

```
[Simulator Runtime Over Each Core]
  Runtime(coreN.veccoreM): N.N
```

### [Pipe Overlap Ratio Over Each Core]

Each of the 18 overlap features gets its own `[Pipe Overlap Ratio Of ...]` subsection:

```
[Pipe Overlap Ratio Over Each Core]
  [Pipe Overlap Ratio Of (VECTOR+CUBE)&MTE2/(VECTOR+CUBE)]
    %(coreN.veccoreM): N.N%
    %(coreN.veccoreM): N.N%    (if multi-core)
  [Pipe Overlap Ratio Of (VECTOR+CUBE)&MTE2/MTE2]
    %(coreN.veccoreM): N.N%
  ...
  [Pipe Overlap Ratio Of VECTOR&MTE3/VECTOR]
    %(coreN.veccoreM): N.N%
```

Same 18 overlap features as global, but each as a separate named subsection with per-core values. Order matches the global section.

---

## Feature Name Mapping for Signal Analysis

When extracting feature values for signal analysis, map report.txt content to feature names as follows:

| report.txt Location | Feature Name | Type |
|---------------------|-------------|------|
| `[Pipe Distribution]` SCALAR cycles% | `SCALAR cycles%` | Global |
| `[Pipe Distribution]` VECTOR cycles% | `VECTOR cycles%` | Global |
| `[Pipe Distribution]` MTE2 cycles% | `MTE2 cycles%` | Global |
| `[Pipe Distribution]` MTE3 cycles% | `MTE3 cycles%` | Global |
| `[Pipe Distribution]` ALL cycles% | `ALL cycles%` | Global |
| `[Pipe Distribution]` FLOWCTRL cycles% | `FLOWCTRL cycles%` | Global |
| `[Key Ratios]` SCALAR:VECTOR_instr ratio | `SCALAR:VECTOR_instr` | Global |
| `[Key Ratios]` SCALAR:VECTOR_cycles ratio | `SCALAR:VECTOR_cycles` | Global |
| `[Key Ratios]` SCALAR_instr_pct | `SCALAR_instr_pct` | Global |
| `[Key Ratios]` SCALAR_cycles_pct | `SCALAR_cycles_pct` | Global |
| `[Key Ratios]` VECTOR_instr_pct | `VECTOR_instr_pct` | Global |
| `[Key Ratios]` VECTOR_cycles_pct | `VECTOR_cycles_pct` | Global |
| `[Key Ratios]` MTE2_instr_pct | `MTE2_instr_pct` | Global |
| `[Key Ratios]` MTE2_cycles_pct | `MTE2_cycles_pct` | Global |
| `[VECTOR Unit]` Read Conflict | `UB Read Conflict` | Global |
| `[VECTOR Unit]` Write Conflict | `UB Write Conflict` | Global |
| `[VECTOR Unit]` Conflict Total | `UB Conflict Total` | Global |
| `[VECTOR Unit]` Utilization avg | `VECTOR utilization avg` | Global |
| `[TRACE Events]` Total events | `Total events` | Global |
| `[TRACE Events]` DIV count | `DIV events` | Global |
| `[TRACE Events]` Arithmetic events% | `Arithmetic events%` | Global |
| `[TRACE Events]` Arithmetic breakdown | `Arithmetic breakdown` | Global |
| `[WAIT_FLAG / BAR Sync]` WAIT_FLAG total | `WAIT_FLAG total` | Global |
| `[WAIT_FLAG / BAR Sync]` BAR total | `BAR total` | Global |
| `[Pipeline Flows]` counts | `Pipeline Flows` | Global |
| `[Pipe Overlap Ratio]` (all 18) | *(same names)* | Global |
| `[Pipe Distribution Over Each Core]` CACHEMISS | `CACHEMISS` | Per-core |
| `[Pipe Distribution Over Each Core]` per-pipe | Per-core pipe% | Per-core |
| `[Pipe Overlap Ratio Over Each Core]` | Per-core overlap% | Per-core |
