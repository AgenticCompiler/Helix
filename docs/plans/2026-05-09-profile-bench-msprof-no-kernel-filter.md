# Profile Bench Msprof No Kernel Filter Implementation Plan

**Goal:** Fix `profile-bench --bench-mode msprof` so it no longer passes unsupported kernel-filter arguments to `msprof`.

**Architecture:** Keep `profile-bench` as a case-level profiler. `run-bench` remains responsible for kernel-aware aggregation, while `profile-bench` only selects the benchmark case and copies back the profiler directory.

## Tasks

- [ ] Update `tests/test_profile_runner.py` so local `msprof` profile commands are expected to omit `--kernel-name`.
- [ ] Add remote `msprof` profile command coverage that also rejects `--kernel-name`.
- [ ] Run the focused profile runner tests and confirm the new expectations fail against the current implementation.
- [ ] Remove profile-time kernel-name resolution and command arguments from `skills/triton-npu-run-eval/scripts/profile_runner.py`.
- [ ] Update `skills/triton-npu-run-eval/references/profile-bench.md`, `README.md`, and relevant docs tests.
- [ ] Run focused tests, strict pyright for the edited skill script, and the standard affected test set.
