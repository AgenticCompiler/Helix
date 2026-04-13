from __future__ import annotations

from test_runner import compare_remote_result_files as _compare_remote_result_files
from test_runner import compare_result_files as _compare_result_files


def compare_result_files(*args, **kwargs):
    return _compare_result_files(*args, **kwargs)


def compare_remote_result_files(*args, **kwargs):
    return _compare_remote_result_files(*args, **kwargs)
