# Strict Opt Round Artifact Naming Design

## Context

The current optimize round contract is permissive in two ways:

- a round-local optimized operator file may be inferred from whatever non-metadata file exists in `opt-round-N/`
- a round-local perf artifact may be declared as `perf.txt` or inferred from a fallback search

This permissiveness makes round artifact ownership ambiguous and weakens the connection between:

- the original operator selected for the optimize session
- the optimized operator snapshot produced for a round
- the perf file produced by `run-bench`

At the same time, `run-bench` already saves perf output beside the operator file as `<operator-stem>_perf.txt`, but the CLI only reports that path as a generic `Perf file:` line instead of explicitly telling the user that the file has been saved.

## Goals

- Make round-local optimize artifacts deterministic and name-derived from the original operator.
- Require each optimize round to use the generated optimized operator filename `opt_<original>.py`.
- Require each optimize round to use the generated perf filename `opt_<original>_perf.txt`.
- Make `run-bench` end with an explicit saved-path message for the perf artifact.

## Non-Goals

- No backward compatibility for historical round layouts.
- No baseline naming changes. `baseline/perf.txt` remains the canonical optimize baseline perf artifact.
- No changes to benchmark parsing or perf comparison math.

## Design

### Round operator naming

For an optimize session rooted at an original operator file `<original>.py`:

- the generated optimize output path remains `opt_<original>.py`
- every completed `opt-round-N/` directory must contain exactly that round operator filename:
  - `opt_<original>.py`

The round checker must stop inferring the operator from arbitrary files. Instead, it should resolve the expected filename from the workspace operator selected for the optimize session and require that exact round-local file.

### Round perf naming

For the same round-local optimized operator file `opt_<original>.py`:

- `run-bench` saves perf output beside the operator as `opt_<original>_perf.txt`
- every completed `opt-round-N/` directory must use that exact perf artifact filename
- `round-state.json["perf_artifact"]` must equal `opt_<original>_perf.txt`

The round checker and optimize-status logic must stop accepting `perf.txt` for round artifacts.

### Baseline naming

Baseline remains unchanged:

- canonical baseline perf artifact: `baseline/perf.txt`
- canonical baseline comparison target recorded in rounds: `baseline/perf.txt`

### Run-bench user message

After a successful `run-bench`, the command should still report the perf path, but it must also end with a direct saved-path message:

- `Saved perf file to: <abs-path>`

This message should appear in both the main CLI command handler and the skill-local `run-command.py` entrypoint so the behavior is consistent regardless of which entrypoint the user or skill invokes.

## Affected Areas

- optimize round contract and checker
- optimize status round perf discovery
- verify round operator selection and copied filenames
- CLI `run-bench` output
- skill-local `run-command.py` output
- optimize docs and workflow references

## Validation

- unit tests for `run-bench` output messaging
- unit tests for strict round operator/perf artifact naming in optimize-check
- unit tests for optimize-status using only `opt_<original>_perf.txt` in round directories
- unit tests for verify selecting `opt_<original>.py` from the best round
