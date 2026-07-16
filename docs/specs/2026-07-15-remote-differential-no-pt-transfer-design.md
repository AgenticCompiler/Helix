# Remote Differential Results Stay Remote

## User-visible behavior

For a remote differential `run-test`, the user must provide
`--ref-operator-file`; a local `--ref-result` is not accepted. Helix uploads
the test, reference operator, candidate operator, and comparison helpers once,
runs both operators in one remote temporary workspace, and compares their
archives there. No `.pt` payload is copied to or from the control machine.

The command returns the remote comparison output and exit code. It preserves
`--keep-remote-workdir`: the one workspace is retained only when requested.
This applies equally to the public CLI and the staged run-eval helper.

## Implementation

Add one test-runner operation owning the workspace lifecycle. It stages inputs
under distinct remote names, runs the existing self-contained differential
script twice with explicit archive names, and invokes the existing
`compare_result.py` helper in that same workspace. The public and staged CLIs
route remote differential commands to that operation before their existing
single-case payload flow.

Local differential behavior and remote standalone behavior remain unchanged.
