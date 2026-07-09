# Optimize Profile Artifact Cleanup

## User-Visible Semantics

- Successful `submit-round` cleanup must:
  - prune the round-local declared `profile/` artifact tree so only `.csv`
    files remain
  - delete root-level `PROF_*` and `OPPROF_*` artifacts from the operator
    workspace
- Optimize session teardown must run the same cleanup again as a fallback, even
  when an agent skipped or failed `submit-round`.
- The cleanup must not run during normal profile collection. Raw profiler
  artifacts may still exist while a round is in progress.

## Problem

The current optimize flow only removes root-level `PROF_*` artifacts during
successful `submit-round` validation. That leaves two gaps:

- round-local `profile/` directories keep large non-CSV profiler byproducts
- if `submit-round` is skipped or fails, workspace-root `PROF_*` and
  `OPPROF_*` artifacts can survive until the whole optimize session ends

## Decision

- Keep `submit-round` as the primary cleanup trigger.
- Add a second cleanup trigger in optimize session teardown so cleanup still
  happens when the worker path misses `submit-round`.
- Centralize the cleanup behavior in the optimize-state round helper so both
  triggers reuse the same implementation through the existing skill-loader
  bridge.
- Treat round-local profile pruning and workspace-root profiler artifact
  deletion as separate operations:
  - round-local profile pruning keeps `.csv` files recursively and removes all
    other files plus empty subdirectories, while preserving the declared
    profile root itself
  - workspace-root cleanup removes only direct child entries whose names start
    with `PROF_` or `OPPROF_`
- For optimize session teardown, scan every `opt-round-*` directory and prune:
  - the declared `profile_dir` when `round-state.json` is readable
  - otherwise the conventional `profile/` directory when present

## Verification

- Add regression coverage for successful `submit-round` cleanup:
  - declared `profile/` trees keep only `.csv`
  - workspace-root `PROF_*` and `OPPROF_*` artifacts are removed
- Add regression coverage for optimize session teardown fallback cleanup:
  - cleanup still runs without `submit-round`
  - conventional `opt-round-*/profile/` directories are pruned
- Run focused unit tests plus the required strict Pyright check for the modified
  optimize-state skill script.
