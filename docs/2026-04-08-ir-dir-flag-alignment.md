# IR Directory Flag Alignment

## Goal

Align the IR analyzer scripts with the optimize workflow's round artifact layout so the user-facing flag matches the standard directory name `opt-round-N/ir/`.

## User-visible behavior

- Replace the public `--archive-dir` flag with `--ir-dir` in:
  - `skills/triton-npu-analyze-ir/scripts/capture_ir.py`
  - `skills/triton-npu-analyze-ir/scripts/inspect_ir.py`
- Update skill guidance and repository docs to show `--ir-dir` in examples and command references.
- Keep the underlying IR directory layout unchanged:
  - `triton_dump/`
  - `bishengir_stages/`
  - `all-ir.txt`
  - `capture-manifest.json`

## Rationale

- The optimize skill already standardizes round-local IR evidence under `opt-round-N/ir/`.
- A flag named `--archive-dir` forces the agent to translate between two names for the same concept.
- A flag named `--ir-dir` makes local ad hoc use and optimize-round use read the same way.

## Design notes

- This is a naming alignment change, not a layout or behavior redesign.
- Internal helper names may continue to use "archive" where helpful, but user-facing help text, examples, and argument parsing should consistently say `--ir-dir`.
- Error messages shown to users should prefer "IR directory" over "archive directory".

## Verification

- Update unit tests so parser coverage explicitly checks `--ir-dir`.
- Run the targeted IR-analyzer tests, repository unittest suite, `pyright`, targeted `ruff`, and skill validation.
