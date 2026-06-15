from __future__ import annotations

from triton_agent.models import CommandKind


def validate_optimize_options(
    command_kind: CommandKind,
    *,
    min_rounds: int,
    round_batch_size: int,
    max_concurrency: int | None,
    resume_mode: str,
    reset_optimize: bool,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str,
    enable_cann_ext_api: bool,
) -> None:
    if min_rounds < 1:
        raise ValueError("--min-rounds must be at least 1")
    if round_batch_size < 1:
        raise ValueError("--round-batch-size must be at least 1")
    if command_kind == CommandKind.OPTIMIZE_BATCH and max_concurrency is not None and max_concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if reset_optimize and resume_mode != "fresh":
        raise ValueError("--reset-optimize requires --resume fresh")
    if enable_cann_ext_api and target_chip != "A5":
        raise ValueError("--enable-cann-ext-api requires --target-chip A5")
