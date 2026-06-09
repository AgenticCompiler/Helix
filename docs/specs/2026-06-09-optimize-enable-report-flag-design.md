# Optimize Enable Report Flag Design

## Goal

Change `optimize` and `optimize-batch` so report generation is disabled by default.
Users must pass `--enable-report` to opt in to automatic `report.md` generation after
successful optimize runs.

## User-Visible Behavior

- Remove the current opt-out flag shape from the optimize commands.
- Add `--enable-report` as the explicit opt-in flag.
- Keep all other optimize behavior unchanged, including:
  - auto-upload defaults
  - interactive mode suppressing auto-report
  - report generation still being best-effort and not changing optimize exit codes

## Implementation Notes

- Update CLI argument registration for optimize command families to expose
  `--enable-report` instead of `--no-report`.
- Update optimize option mapping so `OptimizeRunOptions.report` defaults to `False`
  and becomes `True` only when `--enable-report` is present and `--interact` is not set.
- Update focused tests that currently pass `--no-report` only to suppress the old
  default behavior, and add coverage for the new default and explicit opt-in path.
