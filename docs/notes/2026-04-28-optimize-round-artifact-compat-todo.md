# TODO: Remove Legacy Optimize Round Artifact Compatibility

## Context

The optimize workflow now writes canonical round artifact names:

- `opt_<original-operator>.py`
- `opt_<original-operator>_perf.txt`

To avoid breaking existing workspaces, the current implementation still reads legacy round artifact names during resume, status inspection, round checks, and verification.

The temporary compatibility paths include:

- a round-local operator file that keeps the original operator name instead of using the `opt_` prefix
- a round-local perf artifact named `perf.txt`
- older round-state payloads that still declare those legacy artifact names

## Why This Is Temporary

This compatibility layer exists only to keep already-created workspaces usable after the strict naming change introduced on 2026-04-28.

Newly produced round artifacts should continue to use only the canonical `opt_...` names.

## Cleanup Target

When the repository is ready to drop legacy workspace support, remove the fallback readers and restore strict canonical-name enforcement in these areas:

- [src/triton_agent/optimize/naming.py](/Users/cdj/Projects/triton-agent/src/triton_agent/optimize/naming.py)
- [skills/triton-npu-optimize-check/scripts/optimize_check_contract.py](/Users/cdj/Projects/triton-agent/skills/triton-npu-optimize-check/scripts/optimize_check_contract.py)
- [src/triton_agent/status/core.py](/Users/cdj/Projects/triton-agent/src/triton_agent/status/core.py)
- [src/triton_agent/verification/core.py](/Users/cdj/Projects/triton-agent/src/triton_agent/verification/core.py)

## Exit Criteria

Remove the compatibility layer only after all of the following are true:

- existing optimize workspaces have been migrated, archived, or intentionally declared unsupported
- round checks require only `opt_<original-operator>.py` and `opt_<original-operator>_perf.txt`
- status and verify no longer accept legacy round artifact names
- legacy compatibility tests are removed and canonical-name tests remain green
- required verification still passes, including:
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py`
  - the focused optimize/status/verify unit test suite
