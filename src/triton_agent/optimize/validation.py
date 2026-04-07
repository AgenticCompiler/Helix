from __future__ import annotations

from triton_agent.models import CommandKind


def validate_optimize_options(
    command_kind: CommandKind,
    *,
    min_rounds: int | None,
    max_concurrency: int | None,
    continue_optimize: bool,
    test_mode: str | None,
    bench_mode: str | None,
) -> None:
    if min_rounds is not None and min_rounds < 1:
        raise ValueError("--min-rounds must be at least 1")
    if command_kind == CommandKind.OPTIMIZE_BATCH and max_concurrency is not None and max_concurrency < 1:
        raise ValueError("--max-concurrency must be at least 1")
    if continue_optimize:
        if test_mode is not None:
            raise ValueError("--continue cannot be combined with --test-mode")
        if bench_mode is not None:
            raise ValueError("--continue cannot be combined with --bench-mode")
