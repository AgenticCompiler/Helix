"""Reference test artifacts for pattern-validation workspaces (not runnable pytest)."""

from __future__ import annotations

REFERENCE_TEST_SUFFIX = ".py.txt"


def reference_test_destination_name(source_name: str) -> str:
    """Map repo test_foo.py to workspace reference test_foo.py.txt."""
    name = source_name.strip()
    if name.endswith(REFERENCE_TEST_SUFFIX):
        return name
    if name.endswith(".py"):
        return f"{name}.txt"
    return f"{name}.py.txt"


def is_reference_test_filename(name: str) -> bool:
    return name.endswith(REFERENCE_TEST_SUFFIX)
