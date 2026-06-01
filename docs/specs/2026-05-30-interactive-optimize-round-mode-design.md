# Interactive Optimize Round Mode Design

## Goal

Make `optimize --interact` behave like a single attached interactive session: once the user exits the agent UI, `triton-agent` should stop instead of launching another optimize invocation.

## Problem

Today `optimize --interact` can still participate in the continuous optimize resume loop. If the attached agent exits before `min_rounds` is satisfied, the CLI treats that as a normal continuous-session stop and automatically issues a resume invocation. From the user's perspective, exiting the interactive agent appears to relaunch it immediately.

The other multi-invocation round modes are also a poor fit for `--interact`. `checked` and `supervised` intentionally split optimize work across multiple CLI-managed invocations, while `--interact` implies one directly attached session. Allowing both together creates ambiguous lifecycle semantics.

## User-Visible Behavior

- `optimize --interact` is only valid with `--round-mode continuous`.
- `optimize --interact --round-mode checked` fails fast with a CLI validation error.
- `optimize --interact --round-mode supervised` fails fast with a CLI validation error.
- In `continuous` mode, when the interactive agent process exits, the optimize command returns that result immediately and does not auto-resume to satisfy `min_rounds`.
- Non-interactive continuous optimize runs keep the current auto-resume behavior.

## Design

Keep the change narrow:

- Add argument validation in `src/triton_agent/commands/optimize.py` so `--interact` rejects any round mode other than `continuous`.
- Update `src/triton_agent/optimize/run_loop.py` so the round-satisfaction resume path short-circuits for interactive requests before attempting any automatic `resume()`.
- Preserve the existing recovery behavior for non-interactive stalled runs.

## Testing

- Add a command-level regression test that `--interact` rejects `checked`.
- Add a command-level regression test that `--interact` rejects `supervised`.
- Add a run-loop regression test that `continuous + interact=True` returns after the first successful interactive run without calling `resume()`, even when `min_rounds` is not yet satisfied.

## Scope

- Do not change non-interactive optimize semantics.
- Do not change backend command construction.
- Do not add interactive support for `checked` or `supervised`.
