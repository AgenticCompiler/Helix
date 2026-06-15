# Convert Standalone Test Mode Support

## Summary

Allow `convert` and `convert-batch` to accept `--test-mode standalone` in addition to the existing default `differential`.

## User-Visible Behavior

- `convert --test-mode standalone` and `convert-batch --test-mode standalone` are valid CLI invocations.
- Conversion prompts and skill/docs must describe the requested validation mode instead of hard-coding differential-only wording.
- Convert verification should prefer standalone test artifacts when the request asks for standalone, while preserving the current differential-first behavior for the default mode.

## Implementation Notes

- Keep the default convert test mode as `differential`.
- Remove the convert-specific CLI restriction that narrows `--test-mode` choices to only `differential`.
- Make convert prompt text branch on the requested test mode so standalone requests do not instruct the agent to generate a differential test.
- Update convert verification test-file resolution to prefer the requested mode's default filename and reusable test candidates before falling back to the alternate mode.

## Verification

- Focused parser coverage for `convert` and `convert-batch`.
- Focused convert verification tests covering mode preference when both standalone and differential test files exist.
- Contract coverage for prompt text, skill wording, and README command docs.
