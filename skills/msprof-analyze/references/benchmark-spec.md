# Benchmark CSV Format and Analysis Pipeline

## CSV Format

Expected columns from MindStudio Profiler `op_statistic` CSV:

```
Device_id, OP Type, Core Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)
```

Example row:

```
0,count_nonzero_combin_kernel_1,AI_VECTOR_CORE,297,235837.392,439.489,794.065,1881.798,47.374
```

## Directory Convention

```
{base_dir}/
├── {version}-{round}/
│   └── mindstudio_profiler_output/
│       └── op_statistic_{timestamp}.csv
```

- **version**: tool version name (e.g., `new`, `old`, `v1`, `v2`)
- **round**: integer starting from 1

## Config JSON Schema

```json
{
  "base_dir": "string - root directory",
  "versions": ["string array - version names"],
  "rounds": "int - rounds per version",
  "target_ops": ["string array - OP Type values to analyze"],
  "output_dir": "string - output directory",
  "plot_format": "png|pdf|svg",
  "dpi": "int - plot resolution",
  "metric": "string - column name to analyze"
}
```

## Supported Metrics

- **Avg Time(us)** -- average execution time (default)
- **Min Time(us)** -- minimum execution time
- **Max Time(us)** -- maximum execution time
- **Total Time(us)** -- total execution time across all calls
- **Count** -- number of invocations
- **Ratio(%)** -- percentage of total runtime

## Output Files

1. **comparison_table.csv**: columns are `OP Type`, `{Version}_mean`, `{Version}_std`, `{Version}_median`, `improvement_%`
2. **detailed_analysis.json**: nested dict `[version][round][op_type]` -> `{mean, median, std, min, max, count}`
3. **{op_type}_comparison.png**: line plot with rounds on X-axis, metric on Y-axis, one line per version
4. **summary_heatmap.png**: seaborn heatmap of all numeric columns

## Statistics Computed

- mean, median, std, min, max, count per (version, round, op_type)
- `improvement_% = (old_mean - new_mean) / old_mean * 100` (only for 2-version comparison)

## CLI Usage

```bash
python3 benchmark_analyzer.py --config config.json
python3 benchmark_analyzer.py --base-dir ./profile_dir --versions new old --rounds 5 --target-ops op1 op2 --metric "Avg Time(us)"
```
