# Run-Eval Review Follow-Ups Design

## Goal

Address the confirmed `skills/triton-npu-run-eval/scripts/` review follow-ups without changing user-visible benchmark semantics.

## Scope

- Deduplicate `ResultPayload` and `make_result` into a shared sibling script module.
- Keep standalone runtime execution working for local, remote, and staged-script flows.
- Replace the `bench_runner.py` globals-backed dependency facade with an explicit, typed dependency object.
- Scope `sys.path` mutations in `run-command.py` to imports only.
- Remove confirmed dead code that no longer carries behavior.

## Non-Goals

- Rewriting `run-command.py` into a dispatch table.
- Refactoring unrelated benchmark/profile execution flows.
- Changing perf parsing or timeout semantics.

## Design

### Shared result payload helper

Create `skills/triton-npu-run-eval/scripts/result_payload.py` as the canonical skill-local home for `ResultPayload` and `make_result`. Runtime scripts in this skill import from that module instead of redefining the same shape.

Because `standalone_bench_runtime.py` is copied into isolated workspaces and remote temp directories, every path that stages that runtime must also stage `result_payload.py`.

### Explicit bench runner dependency facade

Replace `_BenchRunnerDeps.__getattr__ -> globals()` with an explicit dependency contract module plus a concrete dependency namespace assembled in `bench_runner.py`. The helper submodules keep their current injection pattern, but the contract becomes grep-friendly and typeable.

### Scoped import path handling

Replace persistent `sys.path.insert(0, ...)` behavior in `run-command.py` with a context manager that temporarily exposes the script directory only while importing sibling helper modules or the profile reporter.

## Verification

- Targeted unit tests for shared payload imports, scoped `sys.path`, and bench-runner dependency structure.
- Skill-script strict pyright for touched `skills/*/scripts/*.py`.
- Focused unittest coverage for run-eval command, benchmark, profile, and remote staging paths.
