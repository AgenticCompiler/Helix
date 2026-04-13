# Optimize Supervised Log Archive Design

## Summary

When `optimize` runs with `--supervise on`, the runtime currently creates a live `.triton-agent/` directory for shared guidance, role briefs, the current round brief, and the current supervisor report. Those files are deleted during cleanup. This keeps later runs clean, but it also discards the supervisor handoff trail that would be useful when reviewing how an optimize session progressed.

This design keeps `.triton-agent/` as an ephemeral runtime directory and adds a persistent archive under `optimize-logs/triton-agent/<run-id>/`. The archive stores immutable snapshots of supervised orchestration artifacts without leaving the live `.triton-agent/` directory behind for future runs to accidentally read.

## Goals

- Preserve supervised optimize orchestration logs after the run finishes.
- Keep live `.triton-agent/` semantics unchanged: it remains a runtime-only working directory.
- Avoid contaminating later optimize runs with stale round briefs or supervisor reports.
- Archive enough information to reconstruct the supervision flow across rounds.
- Apply the behavior only to `--supervise on`.

## Non-Goals

- Do not archive unsupervised optimize runs.
- Do not move round-local operator artifacts, perf artifacts, profile directories, or IR directories into the new archive.
- Do not introduce retention, rotation, compression, or pruning in the first version.
- Do not add a new user-facing CLI flag in the first version.

## Current Behavior

- Supervised optimize writes these live files under `.triton-agent/`:
  - `roles/optimize-worker.md`
  - `roles/optimize-supervisor.md`
  - `round-brief.md`
  - `supervisor-report.md`
- The top-level shared guidance file (`AGENTS.md` or `CLAUDE.md`) is rendered into the workspace for the duration of the run and then removed or restored.
- `round-brief.md` and `supervisor-report.md` are overwritten as the session progresses.
- Cleanup removes the live `.triton-agent/` tree and restores the workspace guidance file.

## Proposed Behavior

### Runtime Directory

The live `.triton-agent/` directory continues to exist only for the current supervised run. Agents still read and write the same live paths during execution.

### Persistent Archive

At the end of a supervised run, the runtime writes a persistent archive under:

`optimize-logs/triton-agent/<run-id>/`

`<run-id>` should be unique per supervised run. A timestamp-based identifier is sufficient for the first version.

### Archived Content

The archive contains:

- `shared-guidance.md`
  - A snapshot of the shared optimize guidance rendered for this run.
- `roles/optimize-worker.md`
- `roles/optimize-supervisor.md`
- `final/round-brief.md`
  - The final live round brief content at shutdown.
- `final/supervisor-report.md`
  - The final live supervisor report content at shutdown.
- `history/round-NNN-brief.md`
  - One immutable brief snapshot per completed supervisor handoff.
- `history/round-NNN-supervisor-report.md`
  - One immutable supervisor report snapshot per completed supervisor handoff.

## Snapshot Timing

### Per-Round History

Each time the supervised loop writes the live `round-brief.md` and `supervisor-report.md`, it must also write immutable copies into `.triton-agent/history/`.

This avoids losing intermediate handoff state when:

- later rounds overwrite the live files
- the run fails after some rounds have already completed
- the user interrupts the optimize session before normal cleanup

### Final Archive

During supervised cleanup, before removing `.triton-agent/`, the runtime copies the live history and final snapshots into `optimize-logs/triton-agent/<run-id>/`.

After the archive is written, cleanup proceeds as it does today:

- remove live `.triton-agent/` files
- remove empty live `.triton-agent/` directories
- remove or restore the workspace guidance file

## Failure And Interrupt Semantics

If a supervised run created `.triton-agent/`, the runtime should attempt to archive the supervised logs on:

- successful completion
- supervisor-driven stop
- hard failure
- user interrupt, when shutdown reaches cleanup

If archive creation itself fails, the runtime should:

- emit a short actionable warning
- continue best-effort cleanup of temporary runtime files
- avoid deleting or overwriting unrelated user files

## Archive Layout Example

```text
optimize-logs/
  triton-agent/
    20260413-153045/
      shared-guidance.md
      roles/
        optimize-worker.md
        optimize-supervisor.md
      final/
        round-brief.md
        supervisor-report.md
      history/
        round-001-brief.md
        round-001-supervisor-report.md
        round-002-brief.md
        round-002-supervisor-report.md
```

## Implementation Notes

- Keep `.triton-agent/` role-neutral and runtime-only.
- Add archive paths to optimize guidance state so cleanup can archive before deletion.
- Write per-round immutable history from the supervised loop when gate handoff files are updated.
- Preserve current behavior for unsupervised optimize.

## Testing

Add coverage for:

- supervised optimize writes immutable `history/` snapshots without overwriting previous rounds
- supervised cleanup creates `optimize-logs/triton-agent/<run-id>/`
- supervised cleanup removes the live `.triton-agent/` directory after archiving
- unsupervised optimize does not create supervised archives
- failure or interruption after at least one supervisor pass still leaves an archive when cleanup runs
