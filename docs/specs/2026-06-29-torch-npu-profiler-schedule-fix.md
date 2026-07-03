# Torch NPU Profiler Schedule Fix

## Summary

Fix the standalone `torch-npu-profiler` benchmark helper so profiler `warmup`
steps do not reduce the number of recorded `active` iterations.

## Problem

The current helper runs one case iteration before entering the profiler, then
configures `torch_npu.profiler.schedule()` with:

- `skip_first = 1 + warmup`
- `warmup = case.warmup`
- `active = case.repeats`

Under `torch_npu` step semantics, `skip_first` is applied before `warmup`.
Counting `warmup` in both places shifts part of the intended active window out
of the executed loop, so fewer active iterations are recorded than
`case.repeats`.

## Decision

Keep the existing pre-profiler dry run, but treat the in-profiler skip and
warmup phases separately:

- keep one pre-profiler dry run
- set `skip_first = 1`
- keep `warmup = case.warmup`
- execute `skip_first + warmup + repeats` profiler steps

This preserves the current intent of discarding the first in-profiler iteration
while ensuring the active region still contains exactly `case.repeats`
iterations.

## Verification

- Add a regression test that simulates `torch_npu` schedule transitions and
  proves `warmup=1, repeats=3` still produces three active iterations worth of
  profiler data.
- Run the focused runtime tests.
