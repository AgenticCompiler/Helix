from __future__ import annotations


def compare_result_files(*args, **kwargs):
    from test_runner import compare_result_files as _compare_result_files

    return _compare_result_files(*args, **kwargs)


def compare_remote_result_files(*args, **kwargs):
    from test_runner import compare_remote_result_files as _compare_remote_result_files

    return _compare_remote_result_files(*args, **kwargs)
