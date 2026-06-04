# Force Recompile Design

## Summary

- Triton kernel recompilation is **always forced** for `run-test`, `run-bench`, `profile-bench`, `verify`, and `verify-batch`.
- `TRITON_ALWAYS_COMPILE=1` is always injected into the Triton kernel compilation environment.
- There is no CLI flag to opt out.

## Rationale

Triton kernels are cached by default. Cached kernels may mask code changes during optimize rounds, leading to misleading benchmark results. Force recompilation ensures measurements always reflect the current code.

## Design

### Environment Variable

The canonical Triton flag for forcing recompilation is `TRITON_ALWAYS_COMPILE=1`.

### Injection Strategy

| Execution Mode | Injection Method |
|---|---|
| Local subprocess | `extra_env={"TRITON_ALWAYS_COMPILE": "1"}` |
| Remote subprocess | `extra_env={"TRITON_ALWAYS_COMPILE": "1"}` |
| In-process | `os.environ["TRITON_ALWAYS_COMPILE"] = "1"` with save/restore |

### Threading Chain

```
CLI -> force_recompile=True (hardcoded)
  -> run_local_test / run_remote_test
    -> test_runner / bench_runner
      -> subprocess with TRITON_ALWAYS_COMPILE=1
```

## Commands Affected

| Command | Behavior |
|---|---|
| `run-test` | Always `TRITON_ALWAYS_COMPILE=1` |
| `run-bench` | Always `TRITON_ALWAYS_COMPILE=1` |
| `profile-bench` | Always `TRITON_ALWAYS_COMPILE=1` |
| `verify` | Always `TRITON_ALWAYS_COMPILE=1` |
| `verify-batch` | Always `TRITON_ALWAYS_COMPILE=1` |
