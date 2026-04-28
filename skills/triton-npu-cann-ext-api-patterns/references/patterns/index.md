# CANN Extension API Pattern Index

Use this file to choose an extension-API-specific optimization direction before reading any detailed pattern reference.

Read this index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

## How To Use This Index

1. Identify the dominant symptom from code inspection, benchmark evidence, profiling evidence, or IR evidence.
2. Decide whether the bottleneck specifically suggests a CANN Triton extension API rewrite instead of a generic optimize pattern.
3. Pick the most relevant extension pattern.
4. Read only that detailed pattern file unless multiple independent extension-API bottlenecks are clearly present.
5. Record why the selected pattern is plausible for the current round.

## Pattern Selection Table

### `sub-vec-id-1to2`

- Use when:
  - the kernel mixes vector work and cube work
  - the mixed kernel matches a vector-plus-cube structure where vector work can be split without changing cube math
  - the kernel still needs full-tile `tl.dot` semantics
- Signals:
  - vector-heavy staging or epilogue work around a full-tile cube path
  - a plausible opportunity to split vector work across the two `sub_vec_id()` lanes without changing cube math
- Expected benefit:
  - better vector-unit utilization on A5 mixed kernels
  - larger viable tiles
  - better vector-heavy tail behavior
- Main risk:
  - incorrect lane ownership
  - unsound vector-to-cube handoff
  - partial-dot rewrites that break full-dot semantics
- Read next:
  - [sub_vec_id_1to2.md](sub_vec_id_1to2.md)
