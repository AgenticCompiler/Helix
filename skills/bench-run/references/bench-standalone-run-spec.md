# Standalone Benchmark Run Spec

## Execution rule

- Run the generated benchmark file directly with bash.
- Preferred form:

```bash
python3 bench_<op>.py
```

- Run from the directory containing the benchmark file when neighboring files are required.

## Artifact expectations

- Success is indicated by exit code `0`.
- The script should print benchmark output directly, including latency lines if it follows the generation spec.
- Parse every stdout line that starts with `latency:`.
- Save those lines as-is, one per line, into the target perf file under `bench_results/`.
- If no `latency:` lines are present, treat the run as failed or non-compliant.

## What to report

- Exact command used
- Exit code
- Primary timing output
- Saved perf file path
- Any failure stderr
