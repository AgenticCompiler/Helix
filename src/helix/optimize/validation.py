from __future__ import annotations

from helix.models import CommandKind

_LANGUAGES_REQUIRING_A5_FOR_CANN_EXT = frozenset({"triton"})


def validate_optimize_options(
    command_kind: CommandKind,
    *,
    min_rounds: int,
    min_speedup: float | None,
    round_batch_size: int,
    max_concurrency: int | None,
    resume_mode: str,
    reset_optimize: bool,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str,
    enable_cann_ext_api: bool,
    language: str = "triton",
) -> None:
    if min_rounds < 1:
        raise ValueError("--min-rounds must be at least 1")
    if min_speedup is not None and min_speedup <= 0:
        raise ValueError("--min-speedup must be greater than 0")
    if round_batch_size < 1:
        raise ValueError("--round-batch-size must be at least 1")
    if command_kind == CommandKind.OPTIMIZE_BATCH and max_concurrency is not None and max_concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if reset_optimize and resume_mode != "fresh":
        raise ValueError("--reset-optimize requires --resume fresh")
    if enable_cann_ext_api:
        if language in _LANGUAGES_REQUIRING_A5_FOR_CANN_EXT and target_chip != "A5":
            raise ValueError("--enable-cann-ext-api requires --target-chip A5")
