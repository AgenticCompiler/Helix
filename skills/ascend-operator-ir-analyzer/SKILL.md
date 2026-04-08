---
name: ascend-operator-ir-analyzer
description: Use when an agent needs to capture, archive, or inspect Triton Ascend compiler IR for an operator workflow, especially to analyze dumped Triton or Bisheng IR stages, reason about likely performance issues from generated IR artifacts, or collect complete IR from a local or remote execution.
---

# Ascend Operator IR Analyzer

## Overview

Capture complete Triton Ascend compiler IR into a stable archive directory, then inspect the archived stages for likely performance problems. Use the bundled script first instead of re-explaining the capture mechanics in the conversation.

## Default Workflow

1. Run the bundled capture helper with an archive directory, a generated benchmark harness, and the operator file you want to inspect.
   - Local:
     ```bash
     python3 ./scripts/capture_ir.py --archive-dir ir-archive --bench-file bench_matmul.py --operator-file matmul.py
     ```
   - Remote:
     ```bash
     python3 ./scripts/capture_ir.py --archive-dir ir-archive --bench-file bench_matmul.py --operator-file matmul.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
     ```

2. Inspect the resulting archive:
   - `triton_dump/`: copied Triton dump directory
   - `bishengir_stages/`: full Bisheng MLIR stage tree
   - `all-ir.txt`: compiler stderr from the replayed Bisheng command
   - `capture-manifest.json`: original command, extracted dump path, original compile command, and replay command

3. Analyze likely performance issues directly from the archived IR.
   - Start from `bishengir_stages/` to see how the IR changes across passes.
   - Use `all-ir.txt` for the complete pass-by-pass print stream.
   - Use `capture-manifest.json` when you need the exact replay context.

4. If the user also needs hotspot evidence or operator timing attribution, use [`../ascend-npu-operator-profiler/SKILL.md`](../ascend-npu-operator-profiler/SKILL.md) as a companion skill.
   - Use this pairing when IR suggests a likely bottleneck but you still need runtime evidence to confirm where time is spent.
   - Use this pairing when profiling already identified a hot operator and you want the IR archive to explain why that hotspot exists.

## Working Rules

- Prefer `python3 ./scripts/capture_ir.py --bench-file ... --operator-file ...` over ad hoc shell sequences so archive layout and replay flags stay consistent.
- For remote capture, the helper stages the benchmark harness and operator file into the remote workspace before running the benchmark command there.
- Keep the archive directory immutable once captured unless the user explicitly asks to replace it.
- Present analysis in terms of concrete artifacts and passes, not only intuition. Call out the relevant archive paths and stage names you inspected.
- Do not invent a fixed Ascend IR tuning methodology yet. Analyze the archived IR directly and be explicit when a conclusion is a hypothesis rather than a proven bottleneck.
