# Codex Show Output Hunk Filter Design

## Summary

`codex exec --show-output` currently strips full unified diffs only when the streamed text includes file-level diff headers such as `diff --git`. Recent Codex output can also emit a bare patch hunk after a successful tool call, which leaks `+`/`-` patch lines into the terminal and show-output log.

This change extends the Codex-specific output filter so non-interactive `--show-output` continues to hide patch content when Codex emits either:

- a normal unified diff with file headers
- a bare hunk fragment that starts at an `@@ ... @@` header without the preceding file metadata

## Goals

- Keep readable command output visible in `--show-output`.
- Hide Codex-emitted patch hunks even when they arrive without `diff --git` headers.
- Preserve normal non-diff lines that happen to be indented after a diff block.
- Keep the change local to the Codex backend filter.

## Non-Goals

- Do not change interactive Codex behavior.
- Do not introduce a generic patch parser shared across backends.
- Do not suppress arbitrary `+` or `-` lines unless surrounding evidence indicates a patch hunk.

## Design

The Codex filter should track hunk state more precisely.

For standard unified diffs:

- Enter diff mode on `diff --git`.
- Enter hunk mode on `@@`.
- Parse the hunk header counts so the filter knows how many old/new lines belong to the hunk.
- Consume exactly that many deletion, addition, and context lines, including indented context lines.
- Exit hunk mode once the parsed counts are exhausted, then resume normal output handling.

For bare hunk fragments:

- If not already inside a diff, treat a standalone `@@ ... @@` line as the start of a synthetic diff hunk.
- Apply the same hunk-count parsing and line consumption rules.
- Exit back to normal output as soon as the hunk is complete.

This keeps filtering narrow: the backend only hides patch-looking output when it sees an actual hunk header, which avoids swallowing ordinary command output that merely contains leading `+` or `-` characters.

## Verification

- Add buffered-process coverage for a bare hunk fragment without `diff --git`.
- Add streaming-process coverage for the same case, including chunked PTY delivery.
- Keep existing regression coverage that normal indented text after a diff remains visible.
