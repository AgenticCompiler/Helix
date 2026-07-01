# Optimize Workflow State Phase Tracking Design

## Goal

Add a temporary optimize workflow state file under `.triton-agent/state.json` so the optimize runtime and optimize skill scripts share one runner-owned source of truth for:

- the current optimize workflow phase
- whether baseline has been formally accepted in the current run
- which round is currently active
- per-round start and end timestamps

This design covers workflow state, phase transitions, and a cleanup-time archive summary of completed round timings. It does not implement hook-based write enforcement.

## User-Visible Semantics

- The workflow-state features in this design are enabled only when optimize runs with `--enable-agent-hook`.
- Without `--enable-agent-hook`, optimize should keep today's behavior: no temporary workflow-state file, no phase-summary injection derived from workflow state, and no cleanup-time `round-timings.json` archive.
- Optimize startup should create a temporary `.triton-agent/state.json` alongside the existing temporary `.triton-agent/` runtime files.
- Optimize cleanup should remove `.triton-agent/state.json` together with the rest of `.triton-agent/`.
- Optimize cleanup should derive `triton-agent-logs/<run-id>/round-timings.json` from the temporary workflow state when one or more rounds completed successfully.
- `round-timings.json` should contain only completed rounds. Unfinished rounds must not be archived there.
- The workflow phase names should be:
  - `baseline`
  - `awaiting_round_start`
  - `round_active`
- After baseline submission passes, the workflow must enter `awaiting_round_start` rather than allowing the next round to start implicitly.
- `triton-npu-optimize-start-round` should formally open one specific round and record that round's `started_at` timestamp.
- `triton-npu-optimize-submit-round` should keep the round active on failure and record that round's `ended_at` timestamp only on success.
- The state file is behavioral for optimize runtime and optimize workflow scripts, not just a post-hoc log.
- The agent should learn the current workflow phase from runtime-injected guidance or prompt text derived from `.triton-agent/state.json`; the agent should not be required to parse the JSON directly for correctness.
- Existing Codex and OpenCode hook behavior should remain unchanged in this design.

## Problem

The current optimize flow relies on prompts, skill instructions, and artifact inspection to keep rounds sequential, but there is no runner-owned workflow state file that records:

- whether the run is still repairing baseline
- whether the run is between rounds and waiting for the next formal round start
- which round is currently active
- when a round actually started and ended

The repository already has durable optimize artifacts such as `baseline/state.json` and `opt-round-N/round-state.json`, but those files serve baseline and round artifact contracts. They should not become temporary workflow-control state.

This design also deliberately avoids coupling workflow-state delivery to hook enforcement. The current guard implementations are read-denial systems: they deny selected `Read` and `Bash` access to protected paths, and they only trace built-in edit tools. Converting those guards into phase-aware built-in edit guards is a separate major capability in both Python and JavaScript. That work should follow once the workflow state mechanism exists.

## Design

### Temporary Workflow State Artifact

Add one temporary workflow-state file:

```text
.triton-agent/state.json
```

This file should be owned by the current optimize run and removed during ordinary optimize cleanup. It should only exist for optimize runs launched with `--enable-agent-hook`. It should not be treated as a durable optimize artifact and should not itself be uploaded or archived as a contract artifact, even though cleanup may derive a separate minimal `round-timings.json` summary from it. The workflow state file should not be reused across future optimize runs.

The file should use a small schema oriented around workflow control rather than artifact description:

```json
{
  "schema_version": 1,
  "run_id": "optimize-20260622-123456-abcdef",
  "phase": "baseline",
  "source_operator": "kernel.py",
  "current_round": null,
  "baseline": {
    "status": "pending",
    "submitted_at": null
  },
  "rounds": {}
}
```

Field semantics:

- `schema_version`: current workflow-state schema version
- `run_id`: bootstrap trace metadata for the current optimize run
- `phase`: one of `baseline`, `awaiting_round_start`, or `round_active`
- `source_operator`: workspace-relative path to the original operator file for this optimize run
- `current_round`: active round number when `phase=round_active`, otherwise `null`
- `baseline.status`: `pending` or `passed`
- `baseline.submitted_at`: UTC ISO 8601 timestamp when the current optimize run observes a passing baseline submission through the baseline submit workflow, otherwise `null`
- `rounds`: mapping keyed by decimal round number strings such as `"1"` or `"12"`

Each round entry should contain only workflow-state data:

```json
{
  "status": "active | passed",
  "round_dir": "opt-round-1",
  "started_at": "2026-06-22T12:40:00Z",
  "ended_at": "2026-06-22T12:55:00Z"
}
```

This file should not duplicate `round-state.json` fields such as perf paths, comparison targets, or artifact metadata.

`run_id` and `source_operator` are informational bootstrap fields in this design:

- runtime should write them when it creates the state skeleton
- later workflow transitions should preserve them unchanged
- the shared helper should not independently validate them against live runner state in this first implementation
- cleanup-time round-timing archiving should not recompute the archive path from `state.json["run_id"]`; it should use the runtime-owned archive path that optimize already prepared for the current run

### Shared Workflow State Helper Placement

The workflow-state helper should not live under `src/triton_agent/` because the optimize skill scripts under `skills/*/scripts/` must not import `triton_agent`.

Instead, add a skill-side shared helper at:

```text
skills/triton/triton-npu-optimize/scripts/optimize_workflow_state.py
```

This helper should be the single owner of:

- loading and validating `.triton-agent/state.json`
- enforcing legal phase transitions
- normalizing round numbers and round directory names
- producing UTC ISO 8601 timestamps
- writing the updated JSON atomically

The helper must not import `triton_agent`.

Runtime code under `src/` that needs the same logic should load this helper through the existing skill-loader bridge rather than creating a reverse dependency from the skill back into `src/`.

The optimize skill scripts that need this helper should use a new cross-skill import pattern. This is not the same as the repository's existing same-directory sibling imports, so the resolution algorithm should be specified explicitly:

1. Start from the current script file path.
2. Resolve `skills_root = Path(__file__).resolve().parents[2]`.
3. Resolve `shared_optimize_scripts_dir = skills_root / "triton-npu-optimize" / "scripts"`.
4. Validate that `shared_optimize_scripts_dir / "optimize_workflow_state.py"` exists.
5. Temporarily prepend `shared_optimize_scripts_dir` to `sys.path` only when it is not already present.
6. Import `optimize_workflow_state`.
7. Remove the temporary `sys.path` entry after import when this script call inserted it.

If the shared helper directory or module is missing, the script should fail explicitly instead of silently falling back.

This algorithm must work both from the repository checkout and from staged backend workspaces such as `.codex/skills/` and `.opencode/skills/`, because those staged layouts preserve the same `skills/<skill-name>/scripts/` shape.

### Atomic Write Rules

Workflow-state writes should be atomic.

The helper should:

- render the full next JSON payload in memory
- write it to a temporary file in `.triton-agent/`
- replace `.triton-agent/state.json` with a final rename

This avoids future readers observing partial JSON.

The design assumes workflow-state access is single-writer. No locking is required in this first implementation because optimize already enforces one-round-at-a-time workflow progression and the same runner-owned workflow scripts are the only intended writers.

### Optimize Runtime Bootstrap And Cleanup

Optimize runtime should keep bootstrap ownership narrow.

At optimize startup:

- when `--enable-agent-hook` is enabled, create `.triton-agent/` through the existing runtime artifact preparation flow
- when `--enable-agent-hook` is enabled, write the minimal `.triton-agent/state.json` skeleton
- when `--enable-agent-hook` is enabled, initialize `phase=baseline` and `baseline.status=pending`
- when `--enable-agent-hook` is disabled, do not bootstrap workflow state and leave optimize runtime behavior unchanged

If optimize startup has bootstrapped workflow state and determines that the baseline is already reusable before any baseline repair launch is needed, runtime may immediately update the bootstrap state to:

- `baseline.status=passed`
- `baseline.submitted_at=null`
- `phase=awaiting_round_start`
- `current_round=null`

This keeps runtime responsible only for minimum bootstrap and teardown while leaving workflow transitions to the skill-side helper and workflow scripts.

`baseline.status=passed` with `baseline.submitted_at=null` is a valid bootstrap shortcut state. It means the current optimize run reused an already-valid baseline instead of observing a fresh passing baseline-submit workflow. Downstream consumers may distinguish that case for observability, but both forms count as passed baseline for phase progression.

At optimize cleanup:

- if workflow state was bootstrapped for this run and one or more rounds have reached `status=passed`, derive a minimal timing summary file at `triton-agent-logs/<run-id>/round-timings.json` before removing live runtime state
- if workflow state was bootstrapped for this run, remove `.triton-agent/state.json` together with the existing `.triton-agent/` runtime directory cleanup
- preserve current fail-fast behavior when stale `.triton-agent/` data already exists before startup

`reset_optimize_workspace()` and clean-subcommand behavior do not need new special cases because `.triton-agent/` is already cleanup-owned.

### Cleanup-Time Round Timings Archive

Optimize cleanup should project completed round timings into the existing per-run archive directory:

```text
triton-agent-logs/<run-id>/round-timings.json
```

Implementation ownership and timing:

- extend the existing optimize runtime cleanup path rather than introducing a new skill script
- specifically, wire this into the same `OptimizeSessionArtifactsManager.archive(...)` / `ArchiveManager` flow that already owns `triton-agent-logs/<run-id>/`
- use `OptimizeSessionArtifactsState.archive.run_archive_dir` as the authoritative archive destination for the current run
- read `.triton-agent/state.json` while the live runtime directory still exists
- write `round-timings.json` before removing `.triton-agent/state.json`
- if there are no completed rounds, skip writing `round-timings.json` entirely

This archive file should be derived from `.triton-agent/state.json`, but it should not reuse the workflow-state schema verbatim.

The file should contain a single JSON array. Each element should contain only:

- `round`
- `started_at`
- `ended_at`

The file should not include:

- `run_id`
- `source_operator`
- baseline status fields
- unfinished-round placeholders such as `ended_at=null`
- any `round-state.json` artifact metadata

Example:

```json
[
  {
    "round": 1,
    "started_at": "2026-06-22T12:40:00Z",
    "ended_at": "2026-06-22T12:55:00Z"
  },
  {
    "round": 2,
    "started_at": "2026-06-22T13:10:00Z",
    "ended_at": "2026-06-22T13:28:00Z"
  }
]
```

Archive projection rules:

- include only rounds whose workflow-state entry has `status=passed`
- require both `started_at` and `ended_at` to be non-null for an archived round
- sort entries by canonical numeric round order
- do not include unfinished or still-active rounds
- do not archive `.triton-agent/state.json` itself
- treat archive write failures as cleanup warnings, not as a fatal optimize failure
- continue removing live `.triton-agent/` runtime state even when `round-timings.json` could not be written

Error-handling behavior should mirror the existing optimize archive flow:

- if the archive directory does not exist yet, let the existing archive manager create it
- if the archive directory already exists but is invalid for reuse, report the same style of warning the archive path already uses
- if writing `round-timings.json` fails, append a warning and continue cleanup

This keeps the run archive useful for later timing analysis without redefining baseline or round artifact contracts, and it keeps the write path aligned with the runtime's existing archive infrastructure instead of making `state.json["run_id"]` a second archive-path authority.

### Agent Phase Visibility

The state file is not purely observational.

It changes behavior at workflow boundaries because:

- baseline submit advances phase only on success
- start-round requires `phase=awaiting_round_start`
- round submit requires `phase=round_active`

These behavioral rules apply only for optimize runs that bootstrapped workflow state under `--enable-agent-hook`.

The agent does not need to parse `.triton-agent/state.json` directly for correctness. Instead, when workflow state is enabled, optimize runtime should surface a compact phase summary through the existing optimize prompt builders in [src/triton_agent/optimize/prompts.py](/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/prompts.py) on each optimize agent launch. If the temporary optimize guidance file also mirrors this summary, it should reuse the same rendered content rather than creating a second independent wording source. That summary should include at least:

- current phase
- current round when present
- whether baseline was reused or freshly passed in this run
- the state file path for debugging

This makes the phase model visible to the agent while keeping enforcement ownership in runtime and workflow scripts rather than in ad hoc free-form file reads.

### Phase Model

The workflow phases should behave as follows.

#### `baseline`

Use this phase while the optimize session is still establishing or repairing baseline artifacts.

This phase includes the current baseline-minimal behavior envelope:

- baseline artifact creation and repair
- root-level test and benchmark harness creation or reuse
- minimum repair of the original operator when baseline preparation requires it

#### `awaiting_round_start`

Use this phase after baseline has passed or after an active round has passed, but before the next round has been formally opened through `triton-npu-optimize-start-round`.

This phase exists to model the important workflow pause between "the last phase is complete" and "the next round is officially open".

#### `round_active`

Use this phase only after `triton-npu-optimize-start-round` has formally opened one specific round.

When this phase is active:

- `current_round` must be a non-null integer
- the matching `rounds["N"]` entry must exist
- that round entry must record `status=active` and a non-null `started_at`

### Round Number Normalization

The helper should follow the repository's current round-directory shape:

- accept names matching `opt-round-(\d+)$`
- parse the captured digits as a base-10 integer
- require the numeric round number to be at least `1`

Normalization behavior:

- `current_round` should store the canonical integer value
- `rounds` keys should use the canonical unpadded decimal string such as `"1"` or `"12"`
- `round_dir` should preserve the original directory spelling that the caller passed

That means an input such as `opt-round-01` remains the stored `round_dir`, but its numeric identity is round `1`. This keeps the helper aligned with the existing `opt-round-(\d+)$` matching behavior instead of introducing a separate directory-name policy in this change.

### Baseline Submission State Transition

Extend the baseline submit skill script:

```text
skills/triton-npu-optimize-submit-baseline/scripts/optimize_submit_baseline.py
```

The existing `check-baseline` behavior should remain the baseline validation authority. But phase advancement must not depend on an optional caller-supplied flag because that would create a silent dead-end when a caller forgets to pass it.

Caller contract:

- if `.triton-agent/state.json` is absent, `check-baseline` behaves like today's pure validation command
- if `.triton-agent/state.json` is present, `check-baseline` must treat workflow-state advancement as part of the optimize workflow contract
- callers should not need to pass a separate `--update-optimize-state` flag

Optimize runtime should only make `.triton-agent/state.json` available to this script when the optimize session was started with `--enable-agent-hook`.

State-file discovery for baseline submit should use the direct-child workspace layout only:

- `baseline_dir = Path(args.baseline_dir).expanduser().resolve()`
- `workspace_root = baseline_dir.parent`
- `state_path = workspace_root / ".triton-agent" / "state.json"`

This design assumes optimize runs only inside an operator workspace root, with `baseline/` as a direct child of that root. No ancestor search or alternative discovery path should be performed. If the caller points `--baseline-dir` at some other layout, that invocation is outside this optimize workflow contract.

This makes optimize workflow state mutation automatic when optimize runtime has bootstrapped state, while preserving the existing script behavior for non-optimize contexts that have no workflow-state file.

When `check-baseline` runs in a workspace that has `.triton-agent/state.json`:

- if the baseline check fails, do not mutate workflow state
- if the baseline check passes:
  - set `baseline.status=passed`
  - set `baseline.submitted_at=<timestamp>`
  - set `phase=awaiting_round_start`
  - set `current_round=null`

If the workflow-state file exists but cannot be loaded or updated, the command should fail explicitly rather than returning success with stale phase state.

This transition should not write round entries.

### Round Start State Transition

Add a new helper script under the start-round skill:

```text
skills/triton-npu-optimize-start-round/scripts/optimize_start_round.py
```

This script should be called by the start-round workflow immediately before the new round begins.

CLI shape:

```bash
python3 scripts/optimize_start_round.py start-round --round-dir opt-round-1
```

Expected behavior:

- require `.triton-agent/state.json` to exist and be valid
- require `phase=awaiting_round_start`
- require `baseline.status=passed`
- require `--round-dir` to match `opt-round-<integer>`
- set `current_round=<integer>`
- create or update `rounds["<integer>"]`
- set that round entry to:
  - `status=active`
  - `round_dir=opt-round-N`
  - `started_at=<timestamp>`
  - `ended_at=null`
- set `phase=round_active`

If a round entry already exists for that number with `status=passed`, the script should fail explicitly rather than silently reopening a completed round.

If `start-round` is called again for the same active round while:

- `phase=round_active`
- `current_round` already equals the requested round number
- the matching round entry exists with `status=active`

then the command should be idempotent:

- return success
- preserve the original `started_at`
- skip the workflow-state write entirely

If `phase=round_active` but the requested round does not match `current_round`, the command should fail explicitly.

### Round Submission State Transition

Extend the round submit script:

```text
skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round.py
```

The existing `check-round` contract should remain the round validation authority. As with baseline submit, optimize workflow state mutation should not depend on an optional caller-supplied flag.

Caller contract:

- if `.triton-agent/state.json` is absent, `check-round` behaves like today's pure validation command
- if `.triton-agent/state.json` is present, `check-round` must treat workflow-state advancement as part of the optimize workflow contract
- callers should not need to pass a separate `--update-optimize-state` flag

Optimize runtime should only make `.triton-agent/state.json` available to this script when the optimize session was started with `--enable-agent-hook`.

State-file discovery for round submit should use the same direct-child workspace layout:

- `round_dir = Path(args.round_dir).expanduser().resolve()`
- `workspace_root = round_dir.parent`
- `state_path = workspace_root / ".triton-agent" / "state.json"`

This design assumes optimize runs only inside an operator workspace root, with each `opt-round-N/` as a direct child of that root. No ancestor search or alternative discovery path should be performed. If the caller points `--round-dir` at some other layout, that invocation is outside this optimize workflow contract.

When `check-round` runs in a workspace that has `.triton-agent/state.json`:

- require a valid `.triton-agent/state.json`
- require `phase=round_active`
- require `current_round` to match the submitted `opt-round-N`

When `.triton-agent/state.json` is present, it is the authoritative source of the active round. Existing CLI arguments keep their current meaning for reporting and min-round guidance, but they must not override workflow state:

- if `--current-round` is omitted, use `state.json["current_round"]` as the only workflow round authority
- if `--current-round` is provided, it must equal `state.json["current_round"]`; otherwise fail explicitly
- `--final-round` remains independent because this design does not add a corresponding workflow-state field for final batch bounds

If `check-round` fails:

- keep `phase=round_active`
- keep `current_round=N`
- do not set `ended_at`

If `check-round` passes:

- set `rounds["N"].status=passed`
- set `rounds["N"].ended_at=<timestamp>`
- set `phase=awaiting_round_start`
- set `current_round=null`

If the workflow-state file exists but cannot be loaded or updated, the command should fail explicitly rather than returning success with stale phase state.

This produces the required round end timestamp while still keeping round artifact details in `opt-round-N/round-state.json`.

### Workflow State Validity Rules

The shared helper should reject invalid combinations such as:

- unknown `schema_version`
- unknown `phase`
- `phase=round_active` with `current_round=null`
- `phase=awaiting_round_start` with `current_round` still set
- `current_round=N` but no matching `rounds["N"]`
- a round entry with `status=passed` but missing `ended_at`
- malformed JSON in `.triton-agent/state.json`

These rules should fail the workflow-state operation explicitly rather than attempting silent repair.

Malformed JSON should be treated the same way as semantic invalidity: fail explicitly, do not re-bootstrap, and require the operator to restart or clean the temporary optimize runtime state.

`baseline.status=passed` with `baseline.submitted_at=null` remains a valid state for the whole run when optimize bootstrapped from an already reusable baseline.

### Deferred Hook Guard Follow-Up

Hook-based write enforcement is intentionally deferred to a follow-up design.

That follow-up should explicitly scope and estimate the net-new work required in both existing guard implementations:

- built-in edit tool recognition in the Codex Python guard and OpenCode JavaScript plugin
- `.triton-agent/state.json` loading and parsing in both languages
- phase-specific path allowlist evaluation
- workspace-relative path normalization and symlink handling for write targets
- denial message generation from the current workflow phase

The follow-up design can consume this workflow state file once the state mechanism is implemented and stabilized.

### Skill Staging Coverage

No new stage-directive entries are required for this design.

The optimize command already stages both:

- `triton-npu-optimize`
- `triton-npu-optimize-start-round`

through [src/triton_agent/skill_staging.py](/Users/cdj/Projects/triton-agent/src/triton_agent/skill_staging.py), so adding new scripts under those skills rides along automatically. The implementer should still verify that no additional skill names or new staged command surfaces were introduced.

## Testing

Add focused tests instead of depending on a live agent session.

- Add workflow-state helper tests for:
  - valid loads
  - malformed JSON
  - unsupported `schema_version`
  - invalid phase combinations
  - atomic write behavior
  - legal and illegal phase transitions
- Add baseline submit tests showing:
  - failed baseline checks do not mutate workflow state
  - successful baseline checks set `phase=awaiting_round_start`
  - state mutation happens automatically when `.triton-agent/state.json` exists, without a caller flag
  - direct-child workspace-root discovery resolves `.triton-agent/state.json` correctly
  - state discovery does not walk ancestor directories
- Add start-round tests showing:
  - valid `awaiting_round_start -> round_active` transition
  - timestamp creation
  - rejection of already-passed rounds
  - idempotent success when the same active round is started twice
- Add round submit tests showing:
  - failed checks remain in `round_active`
  - successful checks set `ended_at` and return to `awaiting_round_start`
  - mismatch between active round and submitted round is rejected
  - state mutation happens automatically when `.triton-agent/state.json` exists, without a caller flag
  - `--current-round` must agree with `state.json["current_round"]` when state is present
  - state discovery does not walk ancestor directories
- Add optimize runtime tests showing:
  - `--enable-agent-hook` startup bootstrap writes `.triton-agent/state.json`
  - reusable-baseline startup enters `awaiting_round_start` when workflow state is enabled
  - `--enable-agent-hook` cleanup writes `triton-agent-logs/<run-id>/round-timings.json` for completed rounds
  - unfinished rounds are excluded from `round-timings.json`
  - `round-timings.json` matches the minimal array schema exactly
  - cleanup creates the archive directory when needed before writing `round-timings.json`
  - archive write failure becomes a cleanup warning while live runtime cleanup still continues
  - ordinary optimize cleanup removes the state file when workflow state was enabled
  - optimize prompts or guidance include a compact current-phase summary derived from the state file only when workflow state is enabled
  - optimize runs without `--enable-agent-hook` skip workflow-state bootstrap, phase-summary injection, and `round-timings.json`

Because this change modifies Python under `skills/*/scripts/`, implementation verification should include the repository's required strict file-scoped skill-script pyright checks in addition to the usual repository lint, type-check, and test commands.

## Scope Boundaries

- Do not treat `.triton-agent/state.json` as a durable optimize artifact.
- Do not archive `.triton-agent/state.json` itself; archive only the derived `round-timings.json` summary.
- Do not add timestamps to `baseline/state.json` or `opt-round-N/round-state.json`.
- Do not change upload, status, verify, report, or hook features to consume `.triton-agent/state.json` in this design.
- Do not change upload, status, verify, report, or hook features to consume `round-timings.json` in this design.
- Do not enable workflow-state bootstrap, phase-summary injection, or `round-timings.json` archive for optimize runs that do not use `--enable-agent-hook`.
- Do not change current Codex or OpenCode hook denial behavior in this design.
- Do not change current `opt-note.md`, `learned_lessons.md`, or supervisor-report ownership semantics in this design.
- Do not merge this temporary workflow state with any user-owned pre-existing `.triton-agent/` data. Existing fail-fast startup behavior should remain.
- Do not describe this feature as a security boundary. It is temporary workflow state for optimize orchestration.
