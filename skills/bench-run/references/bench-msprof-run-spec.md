# Msprof Benchmark Run Spec

## Execution rule

- Use the benchmark file's own CLI directly from bash.
- First query the number of benchmark cases:

```bash
python3 bench_<op>.py --num-bench
```

- Then run one benchmark case at a time with the required parameters:

```bash
python3 bench_<op>.py --operator-file <operator-file> --api-name <api-name> --bench <N>
```

- Use the benchmark file's directory as the working directory when file-relative module loading matters.

## Artifact expectations

- `--num-bench` must work without requiring other arguments.
- Each benchmark case run should return exit code `0` on success.
- If the benchmark file is spec-compliant, the script itself defines how many cases exist and how each case is addressed.
- For each executed case, inspect stdout and stderr for a line beginning with `Task Duration(us):`.
- Convert each extracted value into a normalized line of the form `latency: <value>`.
- Save the normalized latency lines, one per case, into the target perf file under `bench_results/`.
- If any case does not produce `Task Duration(us):`, treat the run as failed or non-compliant.

## What to report

- Result of `--num-bench`
- Exact case command used
- Exit code
- Saved perf file path
- Relevant stdout or stderr

## Summary report

After all cases complete, produce a concise summary including:

- Operator file and API name benchmarked
- Benchmark mode (msprof)
- Number of benchmark cases executed
- Latency values per case (normalized from `Task Duration(us):`)
- Saved perf file path
- If comparing baseline vs optimized: both latency sets and speedup ratio per case
- If failed: which case(s) failed, failure classification and suspected root cause
