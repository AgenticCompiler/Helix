# Default: Triton version.  When language=tilelang, the staging step replaces
# this file with scripts/tilelang/kernel_continuity_check.py.
from triton.kernel_continuity_check import (
    KernelContinuityResult,
    analyze_kernel_continuity,  # noqa: F401
)
