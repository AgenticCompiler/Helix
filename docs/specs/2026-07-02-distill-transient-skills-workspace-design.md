# Distill Transient Skills Workspace Design

## User-Visible Semantics

`helix distill` should leave one skill-related result directory by
default: the `--output-dir` directory containing updated pattern cards and the
manifest for this run. The full editable skills workspace is an implementation
workspace used while agents iterate, not a final artifact.

The command should create the editable knowledge skill under a transient
`.helix/distill-skills` workspace and remove it before returning. The CLI
should not expose a `--skills-dir` option because the editable copy is an
internal workspace.

## Implementation

The CLI always points `DistillConfig.skills_dir` at the transient workspace and
marks it as cleanup-owned. `run_distill()` removes only cleanup-owned transient
skills directories, and only from the `.helix` workspace under the input
root.

The export flow stays unchanged: changed pattern cards are copied to
`--output-dir`, and `updated_patterns.json` records what changed.

## Testing

Tests should verify that the CLI default no longer points at `<input>/skills`,
that `--skills-dir` and `--export-dir` are not accepted, and that a completed
distill run removes the transient skills workspace while preserving the output
directory.
