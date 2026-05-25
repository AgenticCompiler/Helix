# Buffered Process Stderr Drain Design

## Summary

Non-interactive agent runs without `--show-output` currently use the shared buffered process runner. That runner reads `stdout` synchronously and only reads `stderr` after the child exits. If a backend writes enough progress or diagnostics to `stderr`, the child can block before it reaches the file-writing step, which makes commands such as `gen-test` appear to "do nothing" unless `--show-output` switches the run onto the PTY streaming path.

## Goals

- Keep the existing non-interactive CLI contract for buffered runs.
- Prevent buffered agent runs from deadlocking when the child emits substantial `stderr`.
- Preserve separate `stdout` and `stderr` collection in the returned `AgentResult`.
- Keep session-id extraction and stdout filtering working for buffered runs.

## Non-Goals

- Do not change prompt construction, generation skill content, or output-path resolution.
- Do not force all non-interactive runs onto the `--show-output` PTY path.
- Do not redesign interactive-mode process handling.

## Design

Update the shared buffered runner to drain both pipes concurrently while still returning buffered output at the end.

- Launch the child with both `stdout` and `stderr` piped as today.
- Read `stdout` and `stderr` on separate background readers so neither pipe can block the child.
- Keep applying the existing stdout output filter only to `stdout`.
- Continue extracting session ids from stdout text, but make the extraction line-aware so chunked reads do not break detection.
- Treat activity on either stream as stall-timeout progress so a child that is still producing diagnostics is not misclassified as idle.

## Verification

- Add a regression test proving buffered mode completes when the child writes a large `stderr` burst before creating a file and printing stdout.
- Keep existing buffered-process tests passing for stdout capture, diff filtering, interrupt handling, and return-code normalization.
