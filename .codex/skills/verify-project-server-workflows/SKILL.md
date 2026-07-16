---
name: verify-project-server-workflows
description: Validate this repository's Ascend NPU evaluation workflows through real local and remote execution on a user-specified SSH server. Use only when the user explicitly requests real testing on a server or machine for changes to run-test, run-bench, profile-bench, compare-result, staging, remote execution, or shared evaluation runtime.
---

# Verify Project Server Workflows

Perform evidence-backed validation of the changed evaluation workflows on a real NPU-capable server. Do not claim a workflow passed merely because a helper, mock, or isolated worker passed.

## Activation Constraint

Use this skill only after the user explicitly asks to test on a server, remote machine, or specified machine. Do not infer that authorization from a general request to test, validate, run pytest, review a refactor, or verify a change.

Until that explicit request exists, do not open SSH connections, run `rsync`, create a remote `/tmp` directory, or execute `--remote` commands. Perform ordinary local checks instead. If the user requests server testing but does not identify a target, ask for the SSH target before acting.

## Required Inputs

Obtain these values before connecting:

- SSH target, including any required port or SSH config alias.
- Whether copying the current project to the server is authorized.
- The workflow scope: changed commands, relevant fixtures or generated artifacts, and any expected device/runtime setup.

Do not embed a host name, address, user name, credential, server-specific environment setup, or device inventory in this skill. Do not assume an SSH alias configured on the controller also exists on the remote server.

Default to a new, unique verification root beneath the supplied server's `/tmp`. Use another root only when the user explicitly requests it.

## Prepare

1. Inspect `git status --short`; preserve unrelated worktree changes.
2. Read the affected CLI/parser, runner, staging, and comparison code before selecting rows. Derive the matrix from changed behavior, then include all affected local and remote branches.
3. Record the controller Python used by the CLI. For Tensor payloads and result comparisons, use an interpreter that can import `torch`; a system Python without it cannot unpickle or load the resulting artifacts.
4. Create one remote session root before copying or testing. Never reuse or empty an existing `/tmp` directory:

```bash
VERIFY_ROOT=$(ssh <ssh-target> "mktemp -d /tmp/helix-server-verify-XXXXXX")
```

Use `${VERIFY_ROOT}/project` for a synchronized project and `${VERIFY_ROOT}/workspaces` as the value passed to `--remote-workdir`.

5. Synchronize source only when server-local validation needs current changes:

```bash
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' \
  --exclude '.pytest_cache/' --exclude '.mypy_cache/' --exclude '.ruff_cache/' \
  --exclude 'node_modules/' --exclude 'dist/' --exclude 'build/' \
  --exclude '*.pt' --exclude '*.pyc' --exclude '.DS_Store' \
  <project-root>/ <ssh-target>:${VERIFY_ROOT}/project/
```

Exclude persistent remote fixture directories from `--delete`, or keep them beneath `${VERIFY_ROOT}` and outside the synchronized project.

6. Use the smallest valid fixture or generated artifact for every selected command. For differential run-test fixtures, return Tensor or supported nested Tensor outputs; strings are not valid differential outputs. Use at least two named cases when validating `--case-id`.

## Select The Matrix

Execute every affected row in both locations: first directly on the server against `${VERIFY_ROOT}/project` (local mode), then from the controller with `--remote <ssh-target> --remote-workdir ${VERIFY_ROOT}/workspaces`. Prefer the real skill CLI over direct helper calls; API probes may supplement a CLI row but never replace it.

| Workflow | Server-local coverage | Required `--remote` coverage |
| --- | --- | --- |
| `run-test-baseline` | standalone, `--verbose`, selected accuracy mode | standalone remote execution and staging |
| `run-test-convert` / `run-test-optimize` | differential whole-test, reference artifact/operator, `--case-id`, selected accuracy mode | reference/operator direct comparison, `--case-id`, `--keep-remote-workdir` |
| `compare-result` | compare valid local result artifacts in every changed accuracy mode | compare uploaded artifacts remotely when remote comparison changed |
| `run-bench` | configured bench mode, output artifact, baseline comparison if changed | remote execution, output retrieval, `--keep-remote-workdir`, device selection when changed |
| `profile-bench` / `profile-report` | generated profile and report parsing | remote profile retrieval, report generation, retained workspace |
| Staging and runtime helpers | copied scripts and command construction | retained remote workspace contains the required staged scripts and artifacts |

For every selected command that exposes `--remote`, its `--remote` row is mandatory. Do not run irrelevant or unsafe benchmarks merely to fill a table; instead omit the entire workflow and state why. For performance workflows, use the command's intended benchmark contract and report artifacts/return codes; do not assert performance improvement unless that is the requested behavior.

## Robust Execution

- Run long matrices in a background process with unbuffered logging and an explicit exit-status file. Print `START <row>` and `DONE <row>` around each row.
- Poll logs, status files, process trees, and retained remote workspaces at intervals no longer than 60 seconds. SSH/SCP can remain silent for a long time; do not mistake a lost interactive tool session for an evaluation hang.
- Invoke `--remote` rows from the controller. A remote machine may not recognize the controller's SSH alias. Pass `${VERIFY_ROOT}/workspaces` explicitly as `--remote-workdir` so every generated workspace remains inside the session root.
- Check CLI exit status, emitted `Return code`, and the expected artifact. A wrapper process exiting successfully is not sufficient.
- Keep a remote workspace only when it is evidence for a requested row. Confirm its staged files and clean up only directories created by this verification.

## Diagnose Before Retrying

- **Worker succeeds but payload/result cannot be read:** verify the controller interpreter has `torch` and that the expected archive/payload is present before changing serialization.
- **Differential output rejected:** correct the fixture to satisfy the result contract; do not weaken production validation.
- **Long silent upload or execution:** inspect active SSH/SCP children and the remote workspace; reproduce the exact remote command before changing code.
- **Option appears ignored:** trace it through parser, local execution, remote execution, reference generation, artifact comparison, and output. Add a regression test for every missing propagation path.
- **Remote artifact missing:** distinguish failed remote execution, staging omission, cleanup, and copy-back failure from each other using the preserved workspace and command output.

## Finish

Run targeted tests for the changed workflow, strict pyright for every changed skill script, then the repository gates:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
git diff --check
```

Report the selected and omitted matrix rows, exact commands/parameter combinations, return codes, artifact and retained-workspace evidence, server-side versus controller-side execution location, `${VERIFY_ROOT}`, and every failure fixed during validation. State any untested behavior plainly. Clean up only `${VERIFY_ROOT}` after reporting, unless retained workspaces are requested as evidence.
