---
name: msprof-analyze
description: Analyze MindStudio Profiler (msprof) output data for Huawei Ascend NPU operators. Use when the user wants to parse binary profiling files (.bin), compare benchmark CSV results across versions, generate performance reports, or understand operator-level compute/memory bottlenecks on Ascend hardware. Trigger this skill whenever the user mentions msprof, profiler output, profiling data, op_statistic CSV, visualize_data.bin, benchmark comparison, performance analysis, pipe utilization, memory bandwidth, or Ascend NPU profiling.
---

# MindStudio Profiler Analysis

Parse and analyze MindStudio Profiler output data for Ascend NPU operators — from raw binary profiling dumps to multi-version benchmark comparisons.

Use this skill when the user wants to:
- Parse `.bin` files from MindStudio Profiler and extract structured performance data
- Compare `op_statistic_*.csv` benchmark results across tool versions and rounds
- Generate performance reports with statistics, tables, and visualizations
- Understand compute/memory bottlenecks of Ascend NPU operators

## Inputs

- **Binary profiling file** (`.bin`): Raw output from MindStudio Profiler containing embedded JSON blocks (e.g., `visualize_data.bin`)
- **Benchmark CSV directory**: A directory tree following the convention `{base_dir}/{version}-{round}/mindstudio_profiler_output/op_statistic_*.csv`
- **Config file** (optional): A JSON config specifying versions, rounds, target ops, metrics, and output preferences
- **Analysis parameters** (optional): Specific block IDs, operation names, or metrics to focus on

## Outputs

- **In-conversation Markdown report**: Structured performance analysis with tables covering base info, compute workload, and memory workload
- **File outputs** (when benchmark comparison is requested):
  - `comparison_table.csv` — per-version statistical comparison
  - `detailed_analysis.json` — full intermediate data
  - `{op_type}_comparison.png` — line plots per operation
  - `summary_heatmap.png` — heatmap of all metrics

## Two Analysis Modes

### Mode 1: Binary Profiler Parsing

For single-operator deep-dive profiling analysis from `.bin` files.

**When to use:** The user has a `visualize_data.bin` or similar binary file from MindStudio Profiler and wants to understand the operator's performance characteristics.

**Workflow:**

1. Run the bundled parser script to extract and analyze the binary data:
   ```bash
   python3 <skill-path>/scripts/parse_bin.py <path-to-bin-file> [--block-id <N>]
   ```
   If no `--block-id` is specified, defaults to block 0.

2. The script outputs a Markdown report covering three sections:
   - **Base Info**: Operator name, duration, type (vector/cube/mix), block dimensions and detail table
   - **Compute Workload Analysis**: Pipe utilization per block, instruction-level breakdown (instructions, duration, data volume)
   - **Memory Workload Analysis**: Core memory map (20 data paths with bandwidth and requests), L2 cache hit rate, Vector/Cube utilization ratios, per-block memory workload tables with optimization advice

3. Present the report in the conversation. Highlight:
   - Which pipes (Vector, Cube) are underutilized — low utilization signals optimization opportunity
   - Memory bandwidth bottlenecks — compare actual vs. theoretical bandwidth on each data path
   - L2 cache hit rates below 80% — suggests data reuse issues
   - Any advice strings from the profiler (these are Ascend-specific optimization hints)

4. Read [binary-format-spec.md](references/binary-format-spec.md) for details on the binary format, JSON block structure, and how to interpret each section of the output.

### Mode 2: Benchmark Comparison

For comparing operator performance across multiple tool versions and benchmark rounds.

**When to use:** The user has `op_statistic_*.csv` files organized by version and round, and wants to compare performance (e.g., before/after optimization).

**Workflow:**

1. Confirm the directory structure follows the expected convention:
   ```
   {base_dir}/
   ├── {version1}-1/mindstudio_profiler_output/op_statistic_*.csv
   ├── {version1}-2/mindstudio_profiler_output/op_statistic_*.csv
   ├── {version2}-1/mindstudio_profiler_output/op_statistic_*.csv
   └── ...
   ```

2. Create or locate a config JSON (or construct parameters from the user's request):
   ```json
   {
     "base_dir": "./profile_dir",
     "versions": ["new", "old"],
     "rounds": 5,
     "target_ops": ["kernel_name_1", "kernel_name_2"],
     "output_dir": "./analysis_output",
     "metric": "Avg Time(us)"
   }
   ```

3. Run the benchmark analyzer:
   ```bash
   python3 <skill-path>/scripts/benchmark_analyzer.py --config <config.json>
   ```
   Or with CLI arguments:
   ```bash
   python3 <skill-path>/scripts/benchmark_analyzer.py \
     --base-dir <dir> --versions new old --rounds 5 \
     --target-ops <op1> <op2> --output-dir ./analysis_output \
     --metric "Avg Time(us)"
   ```

4. Present key findings in the conversation:
   - Per-operation mean/median comparison across versions
   - Improvement percentages (positive = faster, negative = regression)
   - Standard deviation to assess stability
   - Point the user to generated plots for visual trends

5. Read [benchmark-spec.md](references/benchmark-spec.md) for CSV format details, supported metrics, and output schema.

## Interpreting Results — Ascend NPU Context

The Ascend NPU has a hierarchical memory system and dual compute pipelines. When analyzing results, keep these in mind:

- **Vector vs. Cube**: Vector units handle element-wise ops; Cube handles matrix multiply. An operator marked "mix" uses both — check if one pipe is idle while the other is busy.
- **Data path bandwidth**: The 20 data paths (GM ↔ L2 ↔ L1 ↔ L0A/L0B ↔ Cube/Vector) each have theoretical bandwidth limits. Low actual bandwidth relative to theoretical suggests the operator isn't fully utilizing the memory bus.
- **L2 Cache hit rate**: Critical for memory-bound operators. Below ~80% indicates poor data locality — suggest tiling or data layout changes.
- **Block dimension**: More blocks generally means better parallelism, but too many small blocks can increase scheduling overhead.

For the full Ascend memory hierarchy and data path reference, see [binary-format-spec.md](references/binary-format-spec.md).

## Quality Rules

- Always present the Markdown report in the conversation — don't just save files silently.
- When running benchmark comparison, verify the directory structure before invoking the analyzer. Missing CSV files produce partial results.
- If the user provides a `.bin` file without specifying a block ID, analyze block 0 first, then offer to analyze other blocks.
- For benchmark comparisons with more than 2 versions, note that improvement percentages are only calculated for exactly 2 versions. For 3+ versions, present the statistics table and let the user draw conclusions.
- If the user's CSV files don't match the expected `{version}-{round}` naming pattern, help them reorganize or adjust the config accordingly.

## Failure Handling

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| "No CSV files found for X-N" | Directory structure doesn't match `{version}-{round}/` convention | Verify directory layout, adjust `base_dir` or `versions` in config |
| Binary parser returns empty results | File doesn't contain `ZZ{` markers, or is not a MindStudio profiler output | Confirm file origin; try `extract_and_output()` to split JSON blocks for manual inspection |
| Missing columns in CSV | Profiler version mismatch or non-standard CSV format | Check CSV headers; the expected columns are: `Device_id, OP Type, Core Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)` |
| Plots show empty/NaN | Target operation not found in CSV data | List available OP Types first (`pandas.unique(df['OP Type'])`) and confirm with user |
