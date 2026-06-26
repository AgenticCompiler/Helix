---
name: triton-npu-case-weighting
description: Derive and apply representative benchmark case weights for Triton NPU optimize workflows. Use when a small benchmark subset should be scored against a larger JSON/JSONL case distribution, when full-case benchmark results disagree with representative-case results, or when optimize rounds need weighted compare-perf summaries.
---

# Triton NPU Case Weighting

## Goal

Create a workspace-local `case_weights.json` that maps representative benchmark latency ids to weights derived from the full case distribution.

## Workflow

1. Identify the full case JSON/JSONL file and the benchmark file.
2. Run the bundled script:

   ```bash
   python3 .codex/skills/triton-npu-case-weighting/scripts/case_weighting.py \
     derive \
     --cases-json <operator>.json \
     --bench-file bench_<operator>.py \
     --output case_weights.json
   ```

3. If a full-case baseline perf artifact exists, prefer latency-backed weighting:

   ```bash
   python3 .codex/skills/triton-npu-case-weighting/scripts/case_weighting.py \
     derive \
     --cases-json <operator>.json \
     --bench-file bench_<operator>.py \
     --full-perf baseline/<operator>_full_case_perf.txt \
     --output case_weights.json
   ```

4. During optimize comparisons, pass the weight file to `compare-perf`:

   ```bash
   python3 .codex/skills/triton-npu-run-eval/scripts/run-command.py compare-perf \
     --baseline baseline/<operator>_perf.txt \
     --compare opt-round-N/opt_<operator>_perf.txt \
     --metric-source kernel \
     --case-weights case_weights.json
   ```

## Rules

- Keep the normal perf files unchanged.
- Treat weighted metrics as the representative-subset estimate of full-case impact.
- Keep unweighted metrics visible because they are still useful for debugging individual representative cases.
- Regenerate `case_weights.json` when `REPRESENTATIVE_INDICES`, the full case JSON/JSONL file, or the full-case perf basis changes.
