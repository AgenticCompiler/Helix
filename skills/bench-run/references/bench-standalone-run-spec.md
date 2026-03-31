# Standalone Benchmark Run Spec

## Execution rule

- Run the generated benchmark file directly with bash.
- Preferred form:

```bash
python3 bench_<op>.py --operator-file <operator-file> --api-name <api-name>
```

- The `--operator-file` argument specifies the operator source file to benchmark (e.g. `abs.py` or `opt_abs.py`).
- The `--api-name` argument specifies the operator API function name.
- Run from the directory containing the benchmark file when neighboring files are required.

## Artifact expectations

- Success is indicated by exit code `0`.
- The script should print benchmark output directly, including latency lines if it follows the generation spec.
- Parse every stdout line that starts with `latency:`.
- Save those lines as-is, one per line, into the target perf file under `bench_results/`.
- If no `latency:` lines are present, treat the run as failed or non-compliant.

## What to report

- Exact command used (including `--operator-file` and `--api-name` values)
- Exit code
- Primary timing output
- Saved perf file path
- Any failure stderr

## Summary report

After the run completes, produce a concise summary including:

- Operator file and API name benchmarked
- Benchmark mode (standalone)
- Number of benchmark cases executed
- Latency values per case
- Saved perf file path
- If comparing baseline vs optimized: both latency sets and speedup ratio per case
- If failed: failure classification and suspected root cause
