# Generated Harness PyTorch Entrypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend generated test and benchmark harness contracts so they support Triton wrapper APIs, PyTorch functions, and no-argument PyTorch modules through explicit `api-kind` metadata.

**Architecture:** Keep the CLI unchanged and evolve the generation-side contract instead. Update the skills and normative spec files to describe entrypoint resolution and runtime loading by `api-kind`, then align repository docs and contract tests with the richer metadata header.

**Tech Stack:** Markdown skill specs, repository docs, Python `unittest` contract tests

---

### Task 1: Lock Contract Expectations In Tests

**Files:**
- Modify: `tests/test_generation_contracts.py`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing test**

Add assertions that:
- every generation skill requires `# api-kind:`
- every normative test/bench spec requires `# api-kind:`
- spec wording uses neutral entrypoint language instead of wrapper-only `resolved_wrapper_api`
- specs document the three supported values `triton-wrapper`, `torch-function`, and `torch-module`

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: FAIL because the current skill/spec documents do not mention `api-kind`

- [ ] **Step 3: Write minimal implementation**

Update the skill/spec/docs files referenced below until the new assertions pass.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: PASS

### Task 2: Update Generation Skill Contracts

**Files:**
- Modify: `skills/test-gen/SKILL.md`
- Modify: `skills/bench-gen/SKILL.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing test**

Extend Task 1 assertions so the top-level skills must:
- resolve a public entrypoint instead of only a wrapper API
- allow `triton-wrapper`, `torch-function`, and `torch-module`
- reject guessing constructor arguments for `torch-module`

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: FAIL because the skill docs still require wrapper-only APIs

- [ ] **Step 3: Write minimal implementation**

Revise both skill guides to:
- rename wrapper-only language to entrypoint language
- define `api-kind`
- describe `torch-module` as no-argument construction only
- keep raw Triton kernels out of scope as direct harness targets

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: PASS

### Task 3: Update Normative Test And Benchmark Specs

**Files:**
- Modify: `skills/test-gen/references/test-standalone-spec.md`
- Modify: `skills/test-gen/references/test-differential-spec.md`
- Modify: `skills/bench-gen/references/bench-standalone-spec.md`
- Modify: `skills/bench-gen/references/bench-msprof-spec.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Write the failing test**

Add assertions that the specs:
- include `# api-kind:`
- define runtime behavior for each supported `api-kind`
- show explicit `torch-module` no-argument instantiation behavior
- keep `# kernel:` required

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: FAIL because the current examples and wording are wrapper-only

- [ ] **Step 3: Write minimal implementation**

Revise the four normative spec files to use neutral entrypoint terminology, update sample headers/code, and document `api-kind`-driven loading behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: PASS

### Task 4: Align Repository Documentation

**Files:**
- Modify: `docs/2026-04-01-generated-harness-metadata.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/2026-04-02-generated-harness-pytorch-entrypoints.md`

- [ ] **Step 1: Write the failing test**

Reuse the contract tests when possible and note any remaining documentation gaps by manual review.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: FAIL until documentation references and examples are aligned

- [ ] **Step 3: Write minimal implementation**

Update the metadata doc, README, and AGENTS guidance to describe public entrypoints and `api-kind`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: PASS

### Task 5: Final Verification

**Files:**
- Modify: none
- Test: `tests/test_generation_contracts.py`, repo-wide checks

- [ ] **Step 1: Run focused tests**

Run: `uv run python -m unittest tests.test_generation_contracts -v`
Expected: PASS

- [ ] **Step 2: Run full verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
