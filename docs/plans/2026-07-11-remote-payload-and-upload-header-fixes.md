# Remote Payload And Upload Header Fixes

## Goal

Close the last rename-related runtime gaps that still break real `helix` workflows:

- remote single-case `run-test` must return a clean serialized payload even when remote PTY output includes warning noise
- optimize upload client, service tests, and docs must agree on the renamed `X-Helix-*` header contract

## Steps

1. Normalize remote single-case payload serialization so tensor payloads are copied to CPU before pickling.
2. Harden payload extraction so marker blocks still parse when warning lines appear alongside the base64 payload.
3. Update optimize upload request headers and rename the stale client/server/doc references.
4. Re-run targeted tests, full local validation, and real remote `helix` commands on `R154_cdj`.
