from __future__ import annotations

import warnings


TORCH_NPU_COLLECT_ENV_MODULE_PATTERN = r"torch_npu\.utils\.collect_env"
TORCH_NPU_OWNER_MISMATCH_MESSAGE_PATTERN = r"Warning: The .* owner does not match the current owner\."


def suppress_torch_npu_owner_mismatch_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=TORCH_NPU_OWNER_MISMATCH_MESSAGE_PATTERN,
        category=UserWarning,
        module=TORCH_NPU_COLLECT_ENV_MODULE_PATTERN,
    )
