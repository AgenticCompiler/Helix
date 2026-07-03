# `run-test-baseline` and `run-test-optimize`

Use the `run-test-baseline` MCP tool for baseline or generation validation, and use the `run-test-optimize` MCP tool for optimize-round validation.

Rules:

- Always pass both `test_file` and `operator_file`.
- If `test_mode` is omitted, the tool reads `# test-mode: ...` from the test file.
- Use `test_mode="standalone"` or `test_mode="differential"` only when you need to override the embedded metadata.
- `run-test-baseline` must be used to validate the correctness of a baseline operator.
- `run-test-optimize` must be used to validate the correctness of an optimized operator.
- In optimize differential mode, `run-test-optimize` requires `ref_operator_file`.
- The comparison policy is controlled by the execution environment; invoke the run-test tool normally.
- There is no compare-level option.
- Remote execution uses `remote`, `remote_workdir`, and `keep_remote_workdir` when needed.

Argument examples:

- `run-test-baseline(test_file="test_<operator>.py", operator_file="<operator>.py", test_mode="standalone")`
- `run-test-baseline(test_file="differential_test_<operator>.py", operator_file="<operator>.py", test_mode="differential")`
- `run-test-optimize(test_file="differential_test_<operator>.py", operator_file="opt_<operator>.py", test_mode="differential", ref_operator_file="<operator>.py")`
- `run-test-optimize(test_file="differential_test_<operator>.py", operator_file="opt_<operator>.py", ref_operator_file="<operator>.py", remote="user@host:2222", remote_workdir="/tmp/triton-agent")`
