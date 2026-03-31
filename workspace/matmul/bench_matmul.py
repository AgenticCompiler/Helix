import torch
import triton

from matmul import matmul


CASES = [
    {"m": 512, "n": 512, "k": 512, "dtype": torch.float16},
    {"m": 1024, "n": 1024, "k": 1024, "dtype": torch.float16},
]


def _make_inputs(m: int, n: int, k: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn((m, k), device="npu", dtype=dtype)
    b = torch.randn((k, n), device="npu", dtype=dtype)
    return a, b


def _select_bench_config(fn) -> tuple[int, int]:
    estimated_latency = triton.testing.do_bench_npu(fn, warmup=5, active=10)
    if estimated_latency < 10.0:
        return 1000, 10000
    return 100, 1000


def run_bench() -> None:
    torch.manual_seed(0)

    for case in CASES:
        a, b = _make_inputs(case["m"], case["n"], case["k"], case["dtype"])

        def bench_fn() -> torch.Tensor:
            return matmul(a, b)

        warmup, active = _select_bench_config(bench_fn)
        latency = triton.testing.do_bench_npu(bench_fn, warmup=warmup, active=active)
        print(
            f"case: m={case['m']}, n={case['n']}, k={case['k']}, "
            f"dtype={case['dtype']}, warmup={warmup}, active={active}"
        )
        print(f"latency: {latency}")


if __name__ == "__main__":
    run_bench()
