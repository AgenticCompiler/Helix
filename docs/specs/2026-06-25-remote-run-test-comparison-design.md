## Summary

Remote `run-test`, `run-test-baseline`, and `run-test-optimize` executions already produce archived result files on the remote target and copy those archives back to the control machine. The current automatic post-run comparison step incorrectly performs result comparison locally even when the test itself ran remotely.

## Problem

Local comparison imports the staged `compare_result.py` helper, which imports `torch` at module import time so it can load `.pt` payloads. This makes a successful remote differential test fail afterward on any control machine that does not have local PyTorch installed.

## Intended Behavior

When a `run-test*` command runs remotely and both a reference result and new archived result are available, the automatic comparison must execute through the existing remote comparison helper. Local comparison remains the default only for local `run-test*` executions.

## Implementation Notes

- Update the shared CLI execution handler to route remote automatic comparisons through `compare_remote_result_files(...)`.
- Update the staged skill script `skills/common/ascend-npu-run-eval/scripts/run-command.py` to follow the same rule so direct skill-script usage matches the CLI behavior.
- Add regression coverage proving remote optimize runs use remote comparison and do not require local `torch`.
