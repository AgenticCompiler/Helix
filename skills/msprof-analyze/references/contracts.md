# Validation and Troubleshooting Contract

## Pre-flight Checks

Before running binary analysis:
- Verify the file exists and is readable
- Check file size is reasonable (profiler .bin files are typically 1KB-10MB)
- Confirm it's actually a MindStudio Profiler output (contains ZZ{ markers)

Before running benchmark comparison:
- Verify base_dir exists
- Check at least one {version}-{round} subdirectory exists
- Confirm CSV files are present under mindstudio_profiler_output/
- Validate CSV has expected columns (Device_id, OP Type, etc.)

## Validation After Execution

Binary parsing:
- Expect exactly 5 JSON blocks; fewer means incomplete profiling data
- All block fields should be non-null
- Duration should be positive
- Block dim should be non-negative

Benchmark comparison:
- comparison_table.csv should have rows for each target_op
- NaN values indicate missing data for that version/round combination
- improvement_% only meaningful for 2-version comparisons
- Standard deviation > 50% of mean suggests high variance — recommend more rounds

## Common Failures and Repairs

| Failure | Cause | Repair |
|---------|-------|--------|
| No ZZ{ markers found | Not a MindStudio profiler .bin file | Verify file source; try hex dump to check format |
| Fewer than 5 JSON blocks | Incomplete profiling run | Re-run profiling with MindStudio |
| JSON decode error | Corrupted data or encoding issue | Try extract_and_output() to isolate valid blocks |
| No CSV files found | Wrong directory structure | Check path convention: {version}-{round}/mindstudio_profiler_output/ |
| Target OP not in data | Typo or OP not profiled | List unique OP Types from CSV first |
| Empty plots | Metric column has no numeric data | Check metric name matches CSV column exactly |
| High std deviation | Unstable benchmark environment | Increase rounds, check for thermal throttling |

## Dependencies

Binary parsing (parse_bin.py):
- Python 3.9+
- tabulate (pip install tabulate)

Benchmark analysis (benchmark_analyzer.py):
- Python 3.7+
- pandas >= 1.3.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- numpy >= 1.19.0
