# Evaluation Wall-Clock Timeout Design

## User-Visible Semantics

`HELIX_EVAL_TIMEOUT_SECONDS` limits the wall-clock duration of one operator
evaluation. It applies to `run-test` and to each operator evaluation performed
by `run-bench` (the baseline and candidate are separate evaluations).

- The default limit remains 300 seconds.
- `0` disables the limit.
- Output from the operator does not extend the limit.
- On timeout, Helix terminates the local evaluation process tree, returns a
  non-zero result, and emits an actionable diagnostic that names
  `HELIX_EVAL_TIMEOUT_SECONDS` and says that the current operator execution
  exceeded the limit.
- The diagnostic is returned through the normal command result, so an agent
  invoking `run-test` or `run-bench` can use it as a repair signal.

For `run-bench --baseline-operator-file`, the baseline and candidate each have
their own limit. A run whose baseline has already completed may therefore take
up to two evaluation limits before completing or failing.

## Implementation

The skill-local process runner keeps its existing idle-stall argument for
non-evaluation callers and gains a separate wall-clock timeout argument. Its
buffered mode must monitor pipes from reader threads rather than block on
`readline()`, so a silent process can still be terminated at the deadline.

`run-test` already executes its local work in a worker process and will pass
the evaluation wall-clock limit to that worker. Remote test commands use the
same limit on their SSH invocation.

Local `run-bench` will execute its existing benchmark orchestration in a
dedicated worker process. This covers the default in-process profiler path as
well as serial and parallel benchmark modes without duplicating their case
logic. The parent reads the worker result payload on success; if the worker is
timed out, it returns the timeout diagnostic directly.

## Verification

Focused tests will verify that:

- process output does not reset an evaluation wall-clock timeout;
- a local `run-test` receives the evaluation timeout and exposes its diagnostic;
- a local `run-bench` runs through the timeout-supervised worker and returns a
  timeout result instead of an unbounded in-process benchmark;
- user-facing help and README describe the variable as an execution timeout.
