# Optimize Minimum Rounds

## Summary

- Add an `optimize` CLI option to require at least a target number of optimization rounds.
- If the code agent exits before the workspace reaches that minimum round count, restart the agent with continuation guidance instead of treating the run as complete.
- Continuation prompts must explicitly tell the agent to continue the existing optimization session and reuse prior progress recorded in `opt-note.md` and existing `opt-round-N/` directories.

## User-Visible Behavior

- `uv run helix optimize ... --min-rounds N` requires the optimize workflow to produce at least `N` `opt-round-*` directories in the operator workspace before the command may finish successfully.
- The default behavior remains unchanged when `--min-rounds` is not provided.
- If the agent stalls, the existing recovery behavior still applies.
- If the agent exits successfully but the workspace still contains fewer than `N` round directories, the supervisor launches another optimize pass automatically.
- Continuation prompts must clearly say that the agent should continue the optimization instead of restarting from scratch, and should inspect `opt-note.md`, completed round summaries, and existing round artifacts before deciding the next round.

## Counting Policy

- Count directories matching `opt-round-*` directly under the optimize workspace.
- The count is based on round directory presence, not on validating `summary.md`, because the CLI contract for this feature is expressed in terms of round folder count.
- Ignore non-directory matches.

## Implementation Notes

- Extend the optimize parser with `--min-rounds`.
- Thread `min_rounds` through `AgentRequest`.
- Extend optimize prompt generation so the initial prompt states the minimum round requirement when present.
- Move successful-but-incomplete continuation handling into `OptimizeRunLoop`, because it already owns optimize lifecycle recovery.
- Add a dedicated continuation prompt builder so both restart-after-stall and restart-after-insufficient-rounds share the same “continue from prior progress” language.
- Update both backend resume prompts so they explicitly mention `opt-note.md` and prior round artifacts.

## Verification

- Parser tests for the new `--min-rounds` option.
- Prompt tests for initial optimize guidance with minimum rounds.
- Supervisor tests for successful exit with insufficient rounds causing an additional run.
- Runner tests for continuation prompt wording.
- Update `README.md` and `AGENTS.md` to describe the new optimize behavior.
