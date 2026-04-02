# Differential Test Comparison — Bug Report & Fixes (2026-04-02)

## Bug 1 (CRITICAL): Scalar float NaN comparison silently passes

**Files:** `skills/run-validation/scripts/test_runner.py:268`, `scripts/compare_result_payloads.py:103`

**Problem:** The scalar float comparison used `abs(expected - float(actual)) > threshold`. When either side is NaN, all arithmetic produces NaN, and `NaN > anything` is always `False`. This meant:
- NaN oracle vs any actual value → **silently passes**
- Any oracle vs NaN actual → **silently passes**

This contradicts the tensor path which uses `torch.allclose(..., equal_nan=True)` to only allow NaN-to-NaN matching.

**Fix:** Added explicit NaN guard using `math.isnan`. NaN matches NaN (consistent with `equal_nan=True` on tensors), but NaN vs non-NaN now correctly reports a mismatch.

## Bug 2 (IMPORTANT): int vs float type asymmetry causes false failures

**Files:** Same two files, same location.

**Problem:** The guard was `isinstance(expected, float)`, so:
- `float(3.0)` oracle vs `int(3)` actual → tolerance comparison → **passes**
- `int(3)` oracle vs `float(3.0000001)` actual → falls through to exact `!=` → **fails**

The same pair of values could pass or fail depending on which was oracle vs compare.

**Fix:** Changed the guard to `isinstance(expected, (int, float)) and isinstance(actual, (int, float))`. Both sides are promoted to `float` for tolerance comparison, making the check symmetric.

## Bug 3 (IMPORTANT): Follow-up regression — bool scalars treated as approximate floats

**Files:** Same two files, same location.

**Problem:** The widened numeric guard in Bug 2 also matched `bool`, because Python `bool` is a subclass of `int`. That caused values such as:
- `True` oracle vs `1.00001` actual → **incorrectly passes**
- `False` oracle vs `1e-5` actual → **incorrectly passes**

This is undesirable for scalar result payloads because boolean values should stay exact, not be folded into float-like tolerance matching.

**Fix:** Added a dedicated `bool` branch before numeric tolerance handling. Booleans now use exact value/type comparison, while `int`/`float` non-boolean scalars still use the symmetric float-tolerance path from Bug 2.

## Bug 4 (MINOR): Duplicated `ORACLE_COMPARE_LEVELS`

**Files:** `skills/run-validation/scripts/test_runner.py:22`, `scripts/compare_result_payloads.py:9`

**Problem:** The same tolerance dict is defined independently in both files. If one is updated without the other, local and remote comparisons will use different tolerances.

**Status:** Not fixed. Noted for future cleanup — `compare_result_payloads.py` is only used for remote execution and ideally should import from a shared location.

## What changed

### `skills/run-validation/scripts/test_runner.py` (line 268)

Before:
```python
if isinstance(expected, float):
    if not isinstance(actual, (int, float)):
        return f"{path} type mismatch: ..."
    if abs(expected - float(actual)) > (atol + rtol * abs(expected)):
        return f"{path} scalar mismatch: ..."
    return None
```

After:
```python
if isinstance(expected, bool) or isinstance(actual, bool):
    if type(expected) is not type(actual) or expected != actual:
        return f"{path} value mismatch: ..."
    return None

if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
    exp_f, act_f = float(expected), float(actual)
    import math
    exp_nan, act_nan = math.isnan(exp_f), math.isnan(act_f)
    if exp_nan or act_nan:
        if exp_nan != act_nan:
            return f"{path} NaN mismatch: ..."
        return None
    if abs(exp_f - act_f) > (atol + rtol * abs(exp_f)):
        return f"{path} scalar mismatch: ..."
    return None
```

### `scripts/compare_result_payloads.py` (line 103)

Same change applied identically.

## Regression tests added

- `NaN` vs `NaN` scalar comparison passes.
- `NaN` vs non-`NaN` scalar comparison fails in both directions.
- `int` vs near-equal `float` scalar comparison is symmetric.
- `bool` scalars no longer pass via numeric tolerance.
- Both the local compare path and the remote helper script enforce the same scalar contract.

## Verification

Current verification after the follow-up fix:
```
$ uv run python -m unittest tests.test_test_runner tests.test_remote_execution -v
Ran 19 tests in 0.009s — OK

$ uv run --group dev ruff check tests/test_test_runner.py skills/run-validation/scripts/test_runner.py scripts/compare_result_payloads.py
All checks passed!

$ uv run pyright
0 errors, 0 warnings, 0 informations
```
