# `run-test-baseline`, `run-test-convert`, and `run-test-optimize`

Use `run-test-baseline` for baseline or generation validation, `run-test-convert` for convert validation, and `run-test-optimize` for optimize-round validation.


```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-baseline --test-file test_<operator>.py --operator-file <operator>.py --test-mode standalone
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-baseline --test-file differential_test_<operator>.py --operator-file <operator>.py --test-mode differential

python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert --test-file test_<operator>.py --operator-file triton_<operator>.py --test-mode standalone
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert --test-file differential_test_<operator>.py --operator-file triton_<operator>.py --test-mode differential --ref-operator-file <operator>.py

python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-optimize --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --test-mode differential --ref-operator-file <operator>.py
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-optimize --test-file test_<operator>.py --operator-file opt_<operator>.py --test-mode standalone
```

Rules:

- Always pass both `--test-file` and `--operator-file`.
- If `--test-mode` is omitted, the command reads `# test-mode: ...` from the test file.
- Use `--test-mode standalone` or `--test-mode differential` only when you need to override the embedded metadata.
- Standalone mode never accepts `--case-id`, `--ref-result`, or `--ref-operator-file`.
- Differential mode optionally accepts `--case-id <id>` when you want to rerun only one declared test case during a repair loop.
- If `--case-id <id>` is omitted in differential mode, the command runs every declared test case.
- If you combine `--case-id <id>` with `--ref-result <path>`, the reference payload must cover the same selected case; otherwise the command fails because there is no reference operator available to regenerate it.
- In `--case-id` mode, the command compares a single in-memory payload and does not write a new `*_result.pt` archive for the candidate or any reference rerun.
- If you combine `--case-id <id>` with `--ref-operator-file <path>`, the command first checks the derived `<ref_operator>_result.pt` for that one case and reruns only the missing reference case in memory when needed.
- `run-test-baseline` differential mode accepts at most one of `--ref-result` or `--ref-operator-file`, and it may omit both when you want to produce a reusable archived baseline result.
- `run-test-convert` differential mode requires exactly one of `--ref-result` or `--ref-operator-file`.
- In convert differential mode, `run-test-convert` requires `--ref-operator-file` or `--ref-result`.
- `run-test-optimize` differential mode requires exactly one of `--ref-result` or `--ref-operator-file`.
- Prefer `--verbose` while debugging failures so the command prints per-case progress and richer error context.

- `run-test-baseline` must be used to validate the correctness of a baseline operator.
- `run-test-convert` must be used to validate the correctness of a converted operator.
- `run-test-optimize` must be used to validate the correctness of an optimized operator.
- `run-test-baseline` preserves archived `.pt` result payloads so later optimize validation can reuse them.
- In optimize differential mode, `run-test-optimize` requires `--ref-operator-file` or `--ref-result`.
- The comparison policy is controlled by the execution environment; invoke the run-test command normally.
- There is no compare-level option.

Remote examples:

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-baseline --test-file test_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-baseline --test-file differential_test_<operator>.py --operator-file <operator>.py --test-mode differential --case-id case-0 --verbose
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-convert --test-file differential_test_<operator>.py --operator-file triton_<operator>.py --ref-operator-file <operator>.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py run-test-optimize --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --ref-operator-file <operator>.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
