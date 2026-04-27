# Round Check Triton Kernel Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a round-check validation that rejects optimize rounds whose round-local operator no longer preserves a recognizable Triton kernel launch path.

**Architecture:** Keep the new enforcement inside the existing `triton-npu-optimize-check` skill contract flow. Add one dedicated helper module under `skills/triton-npu-optimize-check/scripts/` to analyze a round-local operator file, then have `check_round()` translate helper failures into a `revise-required` result.

**Tech Stack:** Python 3, `dataclasses`, `pathlib`, Python `unittest`, existing skill-loader-backed optimize checks, `uv run pyright`.

---

### Task 1: Lock the behavior with failing round-check tests

**Files:**
- Modify: `tests/test_optimize_checks.py`
- Test: `tests/test_optimize_checks.py`

- [ ] **Step 1: Write failing tests for mixed Triton-plus-PyTorch pass and pure PyTorch failure**

Add round-check coverage that:

```python
result = optimize_checks.check_round(round_dir)
self.assertTrue(result.ok)
```

for a round-local operator that still contains a recognizable Triton launch path, and:

```python
result = optimize_checks.check_round(round_dir)
self.assertFalse(result.ok)
self.assertEqual(result.decision, "revise-required")
self.assertIn(
    "round operator no longer preserves a recognizable Triton kernel launch path",
    result.issues,
)
```

for a round-local operator rewritten to pure PyTorch.

- [ ] **Step 2: Run the targeted test command and verify the new failure**

Run: `uv run python -m unittest tests.test_optimize_checks -v`
Expected: FAIL because the new pure-PyTorch rejection behavior is not implemented yet.

### Task 2: Add the dedicated Triton continuity helper and wire it into round check

**Files:**
- Create: `skills/triton-npu-optimize-check/scripts/kernel_continuity_check.py`
- Modify: `skills/triton-npu-optimize-check/scripts/optimize_check.py`
- Modify: `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`
- Test: `tests/test_optimize_checks.py`

- [ ] **Step 1: Add a small helper result type and static detection function**

Create a helper shaped like:

```python
@dataclass(frozen=True)
class KernelContinuityResult:
    ok: bool
    reason: str | None


def analyze_triton_kernel_continuity(operator_path: Path) -> KernelContinuityResult:
    ...
```

The helper should pass when the file contains recognizable Triton continuity signals such as Triton imports, `@triton.jit`, and launch syntax like `kernel[...](`, while allowing PyTorch wrapper code to coexist.

- [ ] **Step 2: Call the helper from round check after artifact discovery**

Update `check_round()` so that once `inspect_round_artifacts(round_dir)` succeeds and `operator_path` exists, it runs the helper and converts a failure into:

```python
return _build_result(
    kind="round",
    decision="revise-required",
    issues=("round operator no longer preserves a recognizable Triton kernel launch path",),
)
```

- [ ] **Step 3: Export helper symbols needed by skill-loaded tests**

If needed, update `optimize_check.py` exports so the helper stays accessible through the skill module surface without changing the runtime API shape elsewhere.

- [ ] **Step 4: Run the targeted test command and verify it passes**

Run: `uv run python -m unittest tests.test_optimize_checks -v`
Expected: PASS.

### Task 3: Run verification for touched Python code and repo-targeted checks

**Files:**
- Verify: `skills/triton-npu-optimize-check/scripts/kernel_continuity_check.py`
- Verify: `skills/triton-npu-optimize-check/scripts/optimize_check.py`
- Verify: `skills/triton-npu-optimize-check/scripts/optimize_check_contract.py`
- Verify: `tests/test_optimize_checks.py`

- [ ] **Step 1: Run strict file-scoped pyright for the touched skill scripts**

Run a file-scoped strict check covering:

```bash
bash -lc 'tmpdir=$(mktemp -d); printf "[tool.pyright]\npythonVersion = \"3.11\"\ninclude = [\"%s\", \"%s\", \"%s\"]\ntypeCheckingMode = \"strict\"\n" "$PWD/skills/triton-npu-optimize-check/scripts/kernel_continuity_check.py" "$PWD/skills/triton-npu-optimize-check/scripts/optimize_check.py" "$PWD/skills/triton-npu-optimize-check/scripts/optimize_check_contract.py" > "$tmpdir/pyproject.toml"; uv run pyright --project "$tmpdir/pyproject.toml"'
```

Expected: `0 errors`.

- [ ] **Step 2: Run the focused unittest modules that exercise optimize round contracts**

Run: `uv run python -m unittest tests.test_optimize_checks tests.test_optimize_round_contract -v`
Expected: PASS.
