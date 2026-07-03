# Optimize Workflow State Bootstrap Review Follow-Up Design

## Goal

Record which review findings against the resume-bootstrap design are now stale,
which are non-bugs, and which still require a real code fix.

## Findings Triage

- The earlier "plugin packaging is underspecified" concern is now outdated.
  The follow-on `hook_runtime` refactor defines the packaging boundary and the
  current plugin build copies `src/hook_runtime/` into the built payload.
- The earlier "`source_operator` removal forgot skill-side changes" concern is
  now outdated. The optimize state skill no longer stores `source_operator` in
  `.triton-agent/state.json`.
- The "baseline-phase edit policy is out of scope" concern is valid as a review
  of the older resume-bootstrap design document, but it is not a runtime bug.
  Baseline-phase edit relaxation was intentionally implemented and later split
  across follow-on guard-policy design work.
- The "runner and plugin invalid-state UX differ" concern is not treated as a
  bug. The two hosts intentionally expose different recovery surfaces: runner
  startup fails explicitly, while plugin hooks return repair guidance.

## Real Bug To Fix

When optimize bootstrap runs without a caller-provided `source_operator` hint,
the current helper tries to recover that path from `baseline/state.json` before
classifying the workspace.

If `baseline/state.json` is unreadable but the workspace still contains optimize
session markers such as `opt-note.md` or `opt-round-*`, the helper currently
falls back to "fresh baseline" bootstrap instead of treating the workspace as a
partial or invalid resumable session.

That behavior is wrong for Claude plugin startup because it silently replaces a
recoverable guidance path with a new baseline-phase runtime state.

## Required Behavior

- When `source_operator` is explicitly provided, keep the current classification
  flow unchanged.
- When no hint is provided and `baseline/state.json` cannot supply a valid
  source operator:
  - if the workspace has no optimize-session markers, keep bootstrapping a fresh
    baseline state
  - if the workspace already has optimize-session markers or baseline artifacts,
    raise a `ValueError` so the caller can surface repair guidance instead of
    silently bootstrapping a fresh baseline state

## Validation

Add focused tests for:

- the shared bootstrap helper raising on optimize-looking workspaces when the
  baseline metadata is unreadable and no operator hint exists
- Claude plugin startup returning repair guidance for that same workspace shape
  instead of writing a fresh baseline-phase runtime state
