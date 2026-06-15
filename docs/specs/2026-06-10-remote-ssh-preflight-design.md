# Remote SSH Preflight

## Summary

When a user explicitly passes `--remote user@host[:port]` to a top-level CLI command, the CLI should verify that the target is reachable through non-interactive SSH before starting the real workflow. If the machine is reachable but public-key authentication is not ready, the CLI should fail early with a short message that tells the user to run `ssh-copy-id` and enter the remote login password.

## User-Visible Behavior

- The preflight runs only when the user explicitly provides `--remote` on the top-level CLI command.
- Commands that inherit remote context only through environment variables keep their current behavior and do not gain this extra front-door check.
- If the remote target already accepts the current SSH key, the command continues normally with no extra output.
- If the target rejects the connection because key-based authentication is not set up, the CLI fails before launching the actual workflow and prints:
  - that the remote target is not ready for key-based SSH access
  - the exact `ssh-copy-id` command to run
  - that the user will need to enter the remote login password during that setup step
- If the remote target uses a custom port, the suggested command includes `-p <port>`.
- If the failure is a network, DNS, timeout, or other non-authentication SSH problem, the CLI surfaces that failure instead of incorrectly telling the user to run `ssh-copy-id`.

## Scope

- Apply this explicit-flag preflight to all top-level commands that already accept `--remote`.
- Keep the existing remote execution flow unchanged after the preflight succeeds.
- Do not automatically run `ssh-copy-id`.
- Do not add interactive password prompting to the CLI itself.
- Do not change remote behavior for commands that discover remote context only from injected environment variables.

## Detection Rules

- Use one shared helper for the top-level CLI preflight.
- The helper should run a fast SSH probe that forbids interactive prompts and password fallback.
- Treat authentication-style failures such as rejected public keys, permission denied, or prompt-required SSH failures as the “please run `ssh-copy-id`” case.
- Treat host resolution, routing, timeout, refusal, and similar transport failures as ordinary SSH errors and preserve the original context in the reported error.

## Implementation Shape

- Add a small shared module under `src/triton_agent/` that:
  - parses the existing `user@host[:port]` target format
  - runs the non-interactive SSH probe
  - returns either success or a short actionable failure
- Call that helper once from the top-level CLI dispatch path after argument parsing and before command handler execution.
- Reuse the same remote-target parsing semantics already used by the staged remote runtime so the user-facing target syntax stays consistent.

## Testing

- Add helper-level tests for:
  - host without port
  - host with port
  - authentication failure mapped to an `ssh-copy-id` hint
  - non-authentication SSH failures preserved as-is
- Add CLI-level tests for:
  - explicit `--remote` triggers the preflight
  - explicit `--remote` failure prevents handler execution
  - non-remote commands do not trigger the preflight
  - environment-only remote fallback does not trigger the preflight
