# TileLang Capture IR Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify and harden the TileLang AscendC capture workflow so users get explicit prerequisites, cache-cleanup steps, and actionable failure guidance.

**Architecture:** Keep the behavior change local to the TileLang capture helper and the two TileLang skill documents that describe or consume this workflow. Use focused regression tests to lock the new helper messages and documentation contracts before editing implementation text.

**Tech Stack:** Python, unittest, Markdown skills/docs, file-scoped Pyright

---

### Task 1: Lock The New Guidance With Failing Tests

**Files:**
- Modify: `tests/test_ascend_operator_ir_analyzer.py`
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add failing TileLang helper tests**

```python
def _load_tilelang_capture_ir_module():
    script = (
        REPO_ROOT
        / "skills"
        / "tilelang"
        / "tilelang-npu-analyze-ir"
        / "scripts"
        / "capture_ir.py"
    )
    spec = importlib.util.spec_from_file_location("tilelang_capture_ir_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tilelang_main_reports_missing_compilation_and_name_guidance(self) -> None:
    module = _load_tilelang_capture_ir_module()
    # create a temp operator file that defines no discoverable compiled kernels
    # assert exit_code == 1 and stderr includes:
    # - module-level @tilelang.jit call guidance
    # - non-underscore export guidance


def test_tilelang_main_reports_import_failure_with_cache_guidance(self) -> None:
    module = _load_tilelang_capture_ir_module()
    # create a temp operator file that raises RuntimeError("Compilation Failed")
    # assert exit_code == 1 and stderr includes:
    # - the original failure text
    # - __pycache__
    # - .pkl_memoize_py3


def test_tilelang_main_reports_get_kernel_source_failure_with_cache_guidance(self) -> None:
    module = _load_tilelang_capture_ir_module()
    # patch _load_operator_module() to return a namespace containing
    # a fake compiled kernel whose get_kernel_source() raises RuntimeError("Compilation Failed")
    # assert exit_code == 1 and stderr includes cache cleanup guidance
```

- [ ] **Step 2: Run the TileLang helper tests to verify RED**

Run:

```bash
uv run python -m unittest \
  tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_tilelang_main_reports_missing_compilation_and_name_guidance \
  tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_tilelang_main_reports_import_failure_with_cache_guidance \
  tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_tilelang_main_reports_get_kernel_source_failure_with_cache_guidance -v
```

Expected: FAIL because `skills/tilelang/tilelang-npu-analyze-ir/scripts/capture_ir.py` does not yet emit the requested troubleshooting guidance and import failures are not normalized into a user-facing exit path.

- [ ] **Step 3: Add failing documentation contract tests**

```python
def test_tilelang_capture_ir_skill_documents_cache_cleanup_and_prerequisites(self) -> None:
    skill = _read("skills/tilelang/tilelang-npu-analyze-ir/SKILL.md")
    self.assertIn("module-level", skill)
    self.assertIn("does not start with `_`", skill)
    self.assertIn('find <workspace> -name "__pycache__" -type d -exec rm -rf {} +', skill)
    self.assertIn(".pkl_memoize_py3", skill)
    self.assertIn("No compiled kernels found", skill)
    self.assertIn("Compilation Failed", skill)


def test_tilelang_optimize_artifacts_describe_ascendc_source_capture(self) -> None:
    artifacts = _read("skills/tilelang/tilelang-npu-optimize/references/artifacts.md")
    self.assertIn("ascendc_source.cpp", artifacts)
    self.assertNotIn("bishengir_stages/", artifacts)
    self.assertNotIn("triton_dump/", artifacts)
```

- [ ] **Step 4: Run the documentation contract tests to verify RED**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_tilelang_capture_ir_skill_documents_cache_cleanup_and_prerequisites \
  tests.test_generation_contracts.GenerationContractTests.test_tilelang_optimize_artifacts_describe_ascendc_source_capture -v
```

Expected: FAIL because the current TileLang skill text and artifact reference do not yet include the new prerequisites, cleanup steps, troubleshooting guidance, or AscendC-specific round-local evidence wording.

### Task 2: Implement The TileLang Helper Guidance

**Files:**
- Modify: `skills/tilelang/tilelang-npu-analyze-ir/scripts/capture_ir.py`
- Test: `tests/test_ascend_operator_ir_analyzer.py`

- [ ] **Step 1: Add minimal helper-formatting functions for actionable failures**

```python
def _cache_cleanup_guidance() -> str:
    return (
        'Clear workspace "__pycache__" directories and stale TileLang memoize files '
        '(for example `.pkl_memoize_py3`) before retrying the capture.'
    )


def _no_kernels_found_message(operator_file: Path) -> str:
    return (
        f"No compiled TileLang kernels found in {operator_file}. "
        "Add a module-level call that triggers @tilelang.jit compilation during import, "
        "for example `compiled_kernel = kernel_func(...)`, and expose the compiled kernel "
        "through a name that does not start with `_`."
    )


def _compilation_failure_message(prefix: str, exc: Exception) -> str:
    return f"{prefix}: {exc}. {_cache_cleanup_guidance()}"
```

- [ ] **Step 2: Normalize operator import failures into a controlled exit path**

```python
try:
    module = _load_operator_module(operator_file)
except Exception as exc:
    print(_compilation_failure_message("Failed to load operator", exc), file=sys.stderr)
    return 1
```

- [ ] **Step 3: Use the new no-kernel and get-kernel-source messages**

```python
if not kernels:
    print(_no_kernels_found_message(operator_file), file=sys.stderr)
    return 1

for name, kernel in sorted(kernels.items()):
    try:
        print(kernel.get_kernel_source())
    except Exception as exc:
        print(_compilation_failure_message(f"Compilation failed for kernel '{name}'", exc), file=sys.stderr)
        return 1
```

- [ ] **Step 4: Run the TileLang helper tests to verify GREEN**

Run:

```bash
uv run python -m unittest \
  tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_tilelang_main_reports_missing_compilation_and_name_guidance \
  tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_tilelang_main_reports_import_failure_with_cache_guidance \
  tests.test_ascend_operator_ir_analyzer.AscendOperatorIrAnalyzerTests.test_tilelang_main_reports_get_kernel_source_failure_with_cache_guidance -v
```

Expected: PASS.

### Task 3: Update Skill Contracts And Verify

**Files:**
- Modify: `skills/tilelang/tilelang-npu-analyze-ir/SKILL.md`
- Modify: `skills/tilelang/tilelang-npu-optimize/references/artifacts.md`
- Test: `tests/test_generation_contracts.py`

- [ ] **Step 1: Update the TileLang analyze-ir skill workflow text**

```markdown
## Prerequisites

- Trigger `@tilelang.jit` compilation during module import with a module-level call such as `compiled_kernel = kernel_func(...)`.
- Do not store the compiled kernel only in `_ = kernel_func(...)` or another underscore-prefixed name, because `capture_ir.py` skips names that start with `_`.

## Default Workflow

1. Clear stale caches before capture:
   ```bash
   find <workspace> -name "__pycache__" -type d -exec rm -rf {} +
   ```
   Also remove stale TileLang memoize files such as `.pkl_memoize_py3` if a prior JIT failure is being replayed.
2. Print the AscendC source for an operator:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file matmul.py
   ```

## Troubleshooting

| Error | Likely cause | Fix |
| --- | --- | --- |
| `No compiled kernels found` | No module-level JIT trigger | Add a module-level compiled-kernel call |
| `No compiled kernels found` with `_` export | Underscore-prefixed compiled-kernel name | Rename the exported variable |
| `Compilation Failed` | Stale failed cache state | Clear `__pycache__` and `.pkl_memoize_py3`, then retry |
```

- [ ] **Step 2: Correct the TileLang optimize artifact wording**

```markdown
- When source capture is needed for a round decision, keep the resulting AscendC source under `opt-round-N/ir/`.
- A standard round-local TileLang source-capture workflow uses the `tilelang-npu-analyze-ir` skill's helper with shapes like:
  ```text
  python3 ./scripts/capture_ir.py --operator-file opt-round-N/opt_<operator>.py > opt-round-N/ir/ascendc_source.cpp
  ```
- Compare round-local AscendC source snapshots directly when source-level evidence informs later rounds.
```

- [ ] **Step 3: Run the documentation contract tests to verify GREEN**

Run:

```bash
uv run python -m unittest \
  tests.test_generation_contracts.GenerationContractTests.test_tilelang_capture_ir_skill_documents_cache_cleanup_and_prerequisites \
  tests.test_generation_contracts.GenerationContractTests.test_tilelang_optimize_artifacts_describe_ascendc_source_capture -v
```

Expected: PASS.

- [ ] **Step 4: Run focused final verification**

Run:

```bash
uv run python -m unittest tests.test_ascend_operator_ir_analyzer -v
uv run python -m unittest tests.test_generation_contracts -v
bash scripts/run-skill-script-pyright.sh skills/tilelang/tilelang-npu-analyze-ir/scripts/capture_ir.py
```

Expected: all commands pass.
