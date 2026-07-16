"""Shared result-path and output-filtering helpers for run-test APIs."""

from __future__ import annotations

from pathlib import Path

from result_payload import ResultPayload, make_result


_WARNING_PREFIX = "[WARNING]"


def differential_archive_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def filter_result_payload(result: ResultPayload, *, verbose: bool) -> ResultPayload:
    if verbose:
        return result
    filtered_stdout = _filter_known_warning_lines(str(result["stdout"]))
    filtered_stderr = _filter_known_warning_lines(str(result["stderr"]))
    if filtered_stdout == result["stdout"] and filtered_stderr == result["stderr"]:
        return result
    return make_result(
        return_code=int(result["return_code"]),
        stdout=filtered_stdout,
        stderr=filtered_stderr,
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )


def _filter_known_warning_lines(text: str) -> str:
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not line.rstrip("\r\n").startswith(_WARNING_PREFIX)
    )
