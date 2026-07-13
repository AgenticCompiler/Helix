# Optimize Supervisor Handoff Archive Naming Design

## Goal

Make optimize supervisor handoff archives more explicit and avoid creating empty archive directories when no handoff snapshots were produced.

## User-Visible Semantics

- Supervised optimize archives should store immutable supervisor handoff snapshots under `supervisor-handoffs/` instead of the generic `history/`.
- Checked optimize runs and supervised runs with no recorded handoff snapshots should not create an empty `supervisor-handoffs/` directory.
- Existing archive contents such as `shared-guidance.md`, `supervisor-report.md`, `show-output.log`, trace files, and agent-session files keep their current locations.

## Design

- Rename the live hidden runtime snapshot directory from `.helix/supervisor-history/` to `.helix/supervisor-handoffs/` so internal names match the artifact semantics.
- Rename the archive copy destination from `helix-logs/<run-id>/history/` to `helix-logs/<run-id>/supervisor-handoffs/`.
- Update runtime state, function arguments, and tests to use `handoff` terminology instead of `history` where they specifically describe supervisor snapshot files.
- Create the archive handoff directory only after confirming at least one snapshot file exists to copy.

## Non-Goals

- Do not rename unrelated optimize history concepts such as `opt-note.md` round history.
- Do not change the supervisor snapshot file naming scheme (`round-001-supervisor-report.md`).
- Do not redesign the rest of the optimize archive layout.

## Testing

- Supervised cleanup without any recorded handoff snapshots leaves no `supervisor-handoffs/` archive directory.
- A real supervisor handoff snapshot is archived under `supervisor-handoffs/`.
- Live runtime snapshot paths and archive-copy helpers use the renamed handoff terminology consistently.
