# Round Performance Analysis Deepening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen `triton-npu-analyze-round-performance` into a general Triton NPU operator profiler-first analysis workflow that uses `op_summary`, `task_time`, `api_statistic`, `msprof` JSON, `.bin`, and IR to produce a more actionable `opt-round-N/perf-analysis.md`.

**Architecture:** Keep the existing round-analysis skill as the single entrypoint. Extend `skills/triton-npu-profile-operator/scripts/profile_summary.py` into the unified structured profiler signal extractor, extend `skills/triton-npu-profile-operator/scripts/parse_bin.py` into a stable deep-signal parser, keep `skills/triton-npu-analyze-ir/scripts/inspect_ir.py` as the IR-side evidence source, and enrich the round-analysis skill plus references so the final analysis follows a layered evidence model instead of ad hoc symptom reading.

**Tech Stack:** Python 3.11, `argparse`, `csv`, `json`, `pathlib`, existing profiler and optimize skill docs, Python `unittest`, Markdown references

---

### Task 1: Add failing tests for richer `op_summary`-driven profiler signals

**Files:**
- Modify: `tests/test_ascend_npu_operator_profiler.py`
- Modify: `skills/triton-npu-profile-operator/scripts/profile_summary.py`
- Reuse fixture: `tests/fixtures/ascend_npu_operator_profiler/realistic_parent_layout/mindstudio_profiler_output/op_summary_20260402160352.csv`

- [ ] **Step 1: Add a failing test for operator-type and bound guesses from `op_summary`**

Add a test that feeds representative `op_summary` rows and asserts the JSON payload includes stable fields such as:

- `operator_type_guess`
- `bound_analysis`
- `pipeline_signals`

- [ ] **Step 2: Add a failing test for pipeline-ratio summaries**

Add a test that asserts `aic_mac_ratio`, `aic_scalar_ratio`, `aic_mte*`, `aiv_*`, `cube_utilization(%)`, `Block Dim`, and `Task Wait Time(us)` are summarized into structured signal sections rather than being left only in raw CSV rows.

- [ ] **Step 3: Run the targeted profiler tests to confirm they fail**

Run: `uv run python -m unittest tests.test_ascend_npu_operator_profiler -v`

Expected: FAIL because `profile_summary.py` does not yet expose the new structured `op_summary`-driven signals.

### Task 2: Add failing tests for `task_time`, `api_statistic`, and `msprof` timeline signals

**Files:**
- Modify: `tests/test_ascend_npu_operator_profiler.py`
- Modify: `skills/triton-npu-profile-operator/scripts/profile_summary.py`

- [ ] **Step 1: Add a failing test for task timeline gap and overlap summaries**

Use a synthetic `task_time` CSV fixture inside the test file and assert the JSON payload exposes:

- `task_timeline_signals`
- gap or idle indicators
- task sequencing summaries

- [ ] **Step 2: Add a failing test for host API overhead summaries**

Use a synthetic `api_statistic` CSV fixture and assert the JSON payload exposes:

- `host_api_signals`
- top expensive APIs
- tiling, workspace, or launch-related summaries when they are present

- [ ] **Step 3: Add a failing test for `msprof` JSON timeline summaries**

Use a small synthetic `msprof` JSON payload and assert the profiler summary can surface basic timeline structure or concurrency clues through a stable JSON section.

- [ ] **Step 4: Run the targeted profiler tests to confirm they fail**

Run: `uv run python -m unittest tests.test_ascend_npu_operator_profiler -v`

Expected: FAIL because timeline and host-overhead signals are not yet included in the structured profiler summary.

### Task 3: Add failing tests for structured `.bin` deep-signal extraction

**Files:**
- Modify: `tests/test_msprof_parse_bin.py`
- Modify: `skills/triton-npu-profile-operator/scripts/parse_bin.py`
- Modify: `skills/triton-npu-profile-operator/scripts/profile_summary.py`

- [ ] **Step 1: Add a failing test for stable structured summaries from parsed binary blocks**

Add tests that validate new helper outputs for:

- base operator info
- pipe utilization
- vector wait or instruction-level details
- memory path or bandwidth summaries
- memory load signals

- [ ] **Step 2: Add a failing integration test that `profile_summary.py` can include binary-derived signals**

Use a synthetic or stubbed parse-bin path so the profiler summary JSON can expose a `binary_signals` section without needing a full real profiler binary fixture in the repo.

- [ ] **Step 3: Run the targeted binary tests to confirm they fail**

Run: `uv run python -m unittest tests.test_msprof_parse_bin tests.test_ascend_npu_operator_profiler -v`

Expected: FAIL because `.bin` parsing is still display-oriented and not yet integrated into the richer profiler summary path.

### Task 4: Implement the unified profiler signal extractor in `profile_summary.py`

**Files:**
- Modify: `skills/triton-npu-profile-operator/scripts/profile_summary.py`
- Modify: `skills/triton-npu-profile-operator/SKILL.md`

- [ ] **Step 1: Refactor the summary script around layered evidence sections**

Split the implementation so the script can independently load and summarize:

- `op_statistic`
- `op_summary`
- `task_time`
- `api_statistic`
- `msprof` JSON
- `.bin`-derived signals

- [ ] **Step 2: Implement `op_summary`-driven operator-type and bound heuristics**

Add stable, clearly heuristic logic for:

- `cube` / `vector` / `mix` operator-type guesses
- compute-bound vs memory-bound vs scalar-overhead vs mixed diagnoses
- pipeline signal extraction

- [ ] **Step 3: Implement timeline and host API signal extraction**

Add helpers that summarize:

- task gaps and weak overlap
- host-side high-cost APIs
- basic `msprof` timeline clues

- [ ] **Step 4: Integrate optional binary-derived summaries**

If a compatible profiler binary or parsed block source is available, include a `binary_signals` section in the JSON payload. Keep the script resilient when binary artifacts are absent.

- [ ] **Step 5: Keep Markdown output readable and backward-compatible**

Preserve the current summary use case while expanding Markdown to reflect the richer layered signals where it adds value.

- [ ] **Step 6: Update the profiler skill guidance**

Document that `profile_summary.py` is now the preferred structured profiler signal extractor for round-level deep analysis.

- [ ] **Step 7: Run the targeted profiler tests to verify they pass**

Run: `uv run python -m unittest tests.test_ascend_npu_operator_profiler -v`

Expected: PASS

### Task 5: Implement structured deep-signal extraction in `parse_bin.py`

**Files:**
- Modify: `skills/triton-npu-profile-operator/scripts/parse_bin.py`
- Modify: `tests/test_msprof_parse_bin.py`

- [ ] **Step 1: Add helpers that normalize parsed binary blocks into stable structured sections**

Define clear helpers for:

- `base_info`
- `pipe_utilization`
- `instruction_wait_signals`
- `memory_path_signals`
- `memory_load_signals`

- [ ] **Step 2: Keep raw or Markdown-style block display as a secondary view**

Do not remove existing display-oriented functionality unless a test explicitly shows it is dead or misleading.

- [ ] **Step 3: Run the targeted binary tests to verify they pass**

Run: `uv run python -m unittest tests.test_msprof_parse_bin -v`

Expected: PASS

### Task 6: Deepen the round-analysis skill and add a profiling reference

**Files:**
- Modify: `skills/triton-npu-analyze-round-performance/SKILL.md`
- Create: `skills/triton-npu-analyze-round-performance/references/ascend-npu-profiling-analysis.md`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add a failing contract test for the new evidence model**

Extend skill contract tests so they require the round-analysis skill to mention:

- profiler-first layered analysis
- `.bin` as a first-class deep-analysis path
- IR as explanation and attribution
- the richer `perf-analysis.md` structure

- [ ] **Step 2: Add the profiling reference document**

Translate the accepted parts of `workspace/matmul/ascend-npu-profiling-analysis-guide.md` into a concise reusable reference under the skill, keeping it general for Triton NPU operators.

- [ ] **Step 3: Update `SKILL.md` to point at the reference only when needed**

Keep `SKILL.md` concise and procedural:

- when to read the profiling reference
- when to escalate from CSVs into `.bin`
- when to escalate from profiler evidence into IR
- how to structure `perf-analysis.md`

- [ ] **Step 4: Run the targeted skill contract tests**

Run: `uv run python -m unittest tests.test_generation_contracts -v`

Expected: PASS

### Task 7: Update the `perf-analysis.md` contract and optimize-facing docs

**Files:**
- Modify: `skills/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton-npu-optimize/references/workflow.md`
- Modify: `skills/triton-npu-optimize/references/artifacts.md`
- Modify: `src/triton_agent/prompts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing prompt and contract tests for the deeper analysis structure**

Require optimize-facing guidance to mention:

- profiler-first layered analysis
- standalone `perf-analysis.md`
- `.bin` deep-analysis as part of the round-analysis path

- [ ] **Step 2: Update prompt guidance and optimize docs**

Make optimize guidance explicitly describe the deep round-analysis workflow as:

- profiler-first
- `.bin`-aware
- IR-backed for attribution

- [ ] **Step 3: Update artifact documentation**

Reflect the richer `perf-analysis.md` content contract, including `Binary Signals` and the deeper diagnosis subsections.

- [ ] **Step 4: Run the targeted prompt and contract tests**

Run: `uv run python -m unittest tests.test_cli tests.test_generation_contracts -v`

Expected: PASS

### Task 8: Verify the full change set without committing

**Files:**
- Modify only as needed from earlier tasks after verification feedback

- [ ] **Step 1: Run the focused test modules**

Run:

```bash
uv run python -m unittest \
  tests.test_ascend_npu_operator_profiler \
  tests.test_msprof_parse_bin \
  tests.test_generation_contracts \
  tests.test_cli -v
```

Expected: PASS

- [ ] **Step 2: Run the full unit suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS

- [ ] **Step 3: Run type checking**

Run: `uv run pyright`

Expected: PASS

- [ ] **Step 4: Run lint for touched Python files**

Run:

```bash
uv run --group dev ruff check \
  skills/triton-npu-profile-operator/scripts/profile_summary.py \
  skills/triton-npu-profile-operator/scripts/parse_bin.py \
  skills/triton-npu-analyze-round-performance/SKILL.md \
  tests/test_ascend_npu_operator_profiler.py \
  tests/test_msprof_parse_bin.py \
  tests/test_cli.py \
  tests/test_generation_contracts.py
```

Expected: PASS

- [ ] **Step 5: Run skill validation**

Run: the skill validation workflow available in your environment against `skills/triton-npu-analyze-round-performance`

Expected: PASS

- [ ] **Step 6: Review `git diff` and stop for human approval before any commit**

Do not create a commit in this implementation pass unless the user explicitly asks for one after review.
