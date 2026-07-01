# Run-Eval Skill Helper Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace current-directory-relative `./scripts/run-command.py` guidance in the live `ascend-npu-run-eval` skill docs with explicit staged-skill-path placeholders, and lock the behavior with a contract test.

**Architecture:** Keep the change documentation-only. First tighten `tests/test_generation_contracts.py` so it fails until the new `<ascend-npu-run-eval-skill-path>/scripts/run-command.py` contract is present. Then update the top-level run-eval skill doc and each focused reference under `skills/common/ascend-npu-run-eval/` to use the same backend-neutral placeholder and one shared explanation of what `<ascend-npu-run-eval-skill-path>` means.

**Tech Stack:** Markdown skill docs, `pytest`

---

## File Structure

- `tests/test_generation_contracts.py`: owns live skill documentation contract checks for `ascend-npu-run-eval`.
- `skills/common/ascend-npu-run-eval/SKILL.md`: owns the router-level helper entrypoint guidance.
- `skills/common/ascend-npu-run-eval/references/run-test.md`: owns documented `run-test-baseline` and `run-test-optimize` helper examples.
- `skills/common/ascend-npu-run-eval/references/run-bench.md`: owns documented `run-bench` helper examples.
- `skills/common/ascend-npu-run-eval/references/probe-bench.md`: owns documented helper fallback for `probe-bench`.
- `skills/common/ascend-npu-run-eval/references/profile-bench.md`: owns documented `profile-bench` and follow-up `profile-report` helper examples.
- `skills/common/ascend-npu-run-eval/references/profile-report.md`: owns documented `profile-report` helper examples.
- `skills/common/ascend-npu-run-eval/references/compare-result.md`: owns documented `compare-result` helper examples.
- `skills/common/ascend-npu-run-eval/references/compare-perf.md`: owns documented `compare-perf` helper examples.

### Task 1: Lock the new helper-path contract in tests

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing contract assertions**

```python
    def test_run_eval_skill_uses_explicit_skill_path_helper(self) -> None:
        skill = _read("skills/common/ascend-npu-run-eval/SKILL.md")
        run_test = _read("skills/common/ascend-npu-run-eval/references/run-test.md")
        run_bench = _read("skills/common/ascend-npu-run-eval/references/run-bench.md")
        probe_bench = _read("skills/common/ascend-npu-run-eval/references/probe-bench.md")
        profile_bench = _read("skills/common/ascend-npu-run-eval/references/profile-bench.md")
        profile_report = _read("skills/common/ascend-npu-run-eval/references/profile-report.md")
        compare_result = _read("skills/common/ascend-npu-run-eval/references/compare-result.md")
        compare_perf = _read("skills/common/ascend-npu-run-eval/references/compare-perf.md")

        self.assertIn("<ascend-npu-run-eval-skill-path>/scripts/run-command.py", skill)
        self.assertIn("call `python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py <subcommand> ...` directly", skill)

        for doc in (
            skill,
            run_test,
            run_bench,
            probe_bench,
            profile_bench,
            profile_report,
            compare_result,
            compare_perf,
        ):
            self.assertNotIn("python3 ./scripts/run-command.py", doc)
```

- [ ] **Step 2: Run the focused contract test and verify it fails**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py -k run_eval_skill`

Expected: FAIL because the live run-eval skill docs still use `python3 ./scripts/run-command.py`.

### Task 2: Update the live run-eval skill docs to use `<ascend-npu-run-eval-skill-path>`

**Files:**
- Modify: `skills/common/ascend-npu-run-eval/SKILL.md`
- Modify: `skills/common/ascend-npu-run-eval/references/run-test.md`
- Modify: `skills/common/ascend-npu-run-eval/references/run-bench.md`
- Modify: `skills/common/ascend-npu-run-eval/references/probe-bench.md`
- Modify: `skills/common/ascend-npu-run-eval/references/profile-bench.md`
- Modify: `skills/common/ascend-npu-run-eval/references/profile-report.md`
- Modify: `skills/common/ascend-npu-run-eval/references/compare-result.md`
- Modify: `skills/common/ascend-npu-run-eval/references/compare-perf.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Update the top-level router guidance**

```md
Use the bundled helper script in this skill. Treat
`<ascend-npu-run-eval-skill-path>` as the staged path to the
`ascend-npu-run-eval` skill for the active backend:

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py <subcommand> ...
```

...

- call `python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py <subcommand> ...` directly
```

- [ ] **Step 2: Rewrite each focused reference example with the explicit placeholder**

```md
python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py run-test-baseline --test-file test_<operator>.py --operator-file <operator>.py --test-mode standalone
python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py run-bench --bench-file bench_<operator>.py --operator-file <operator>.py
python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py compare-perf \
python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py probe-bench \
```

- [ ] **Step 3: Preserve existing semantics while removing only the ambiguous path form**

```md
If the staged helper script in this skill already exposes the subcommand, the equivalent helper form is:

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/run-command.py probe-bench \
  --bench-file bench_<operator>.py \
  --operator-file opt_<operator>.py \
  --baseline-operator-file baseline/<operator>.py
```
```

- [ ] **Step 4: Re-run the focused contract test and verify it passes**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py -k run_eval_skill`

Expected: PASS with the live run-eval docs using `<ascend-npu-run-eval-skill-path>/scripts/run-command.py` and no remaining `python3 ./scripts/run-command.py` strings in the covered files.

### Task 3: Run final verification for the touched contract area

**Files:**
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Run the full generation-contract suite**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_generation_contracts.py`

Expected: PASS with no regressions in other live skill doc contracts.
