# Gen Eval Force Overwrite

## Summary

Add `--force-overwrite` support to `gen-eval`.

## User-visible behavior

- By default, `gen-eval` protects existing generated outputs and fails if the target generated test file or benchmark file already exists.
- With `--force-overwrite`, `gen-eval` deletes the generated test file and generated benchmark file that it is about to recreate.
- With `--force-overwrite`, `gen-eval` also deletes the archived execution artifacts for the same operator: `<operator-stem>_result.pt` and `<operator-stem>_perf.txt`.
- `gen-eval --force-overwrite` does not delete the original operator file.

## Implementation notes

- Extend CLI parsing so `gen-eval` accepts `--force-overwrite`.
- Reuse the existing overwrite protection for single-output generators.
- Add a `gen-eval`-specific preparation path that resolves both generated target files and removes them only when overwrite is explicitly requested.
- In overwrite mode, remove operator-associated archived execution outputs with stable names derived from the operator stem.
- Update prompt text so `gen-eval` describes overwriting both generated outputs and archived execution outputs instead of a single requested output file.
