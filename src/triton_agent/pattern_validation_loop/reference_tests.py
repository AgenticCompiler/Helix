from __future__ import annotations

REFERENCE_TEST_SUFFIX = ".py.txt"


def build_pattern_validation_optimize_reference_test_prompt() -> str:
    return (
        "Pattern-validation workspace: reference test files are provided as `test_*.py.txt` "
        "(not runnable pytest). Each file is copied from the target repo as a **reference only** "
        "for dtype, tensor shapes, and how the operator is exercised. "
        "Use them when generating or updating `test_*.py` / `bench_*` for optimize rounds; "
        "do not treat `.py.txt` files as tests to execute directly."
    )
