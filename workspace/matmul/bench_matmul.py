import argparse
import importlib.util
import time

import torch
import triton.backends.ascend.testing

# bench-mode: standalone
# api-name: matmul
# kernel: matmul_kernel

API_NAME = "matmul"
CASES = [
    ("case-1", {"m": 32, "k": 64, "n": 128, "dtype": torch.float32, "seed": 42}),
    ("case-2", {"m": 64, "k": 64, "n": 64, "dtype": torch.float32, "seed": 123}),
    ("case-3", {"m": 64, "k": 128, "n": 256, "dtype": torch.float32, "seed": 456}),
    ("case-4", {"m": 128, "k": 256, "n": 64, "dtype": torch.float32, "seed": 789}),
]


def load_operator_api(operator_file: str, api_name: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load operator module from {operator_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, api_name)
    except AttributeError as exc:
        raise RuntimeError(f"Operator file does not define API '{api_name}': {operator_file}") from exc


def make_inputs(case: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = case["dtype"]
    seed = case["seed"]
    m = case["m"]
    k = case["k"]
    n = case["n"]
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=dtype, device="npu")
    b = torch.randn((k, n), dtype=dtype, device="npu")
    return a, b


def select_bench_config(bench_fn) -> tuple[int, int]:
    started_at = time.perf_counter()
    bench_fn()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    if elapsed_ms < 10.0:
        return 1000, 10000
    return 100, 1000


def run_bench(operator_api) -> None:
    for case_id, case in CASES:
        inputs = make_inputs(case)

        def bench_fn():
            return operator_api(*inputs)

        warmup, active = select_bench_config(bench_fn)
        latency = triton.backends.ascend.testing.do_bench_npu(
            bench_fn,
            warmup=warmup,
            active=active,
        )
        print(f"latency-{case_id}: {latency}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", required=True)
    args = parser.parse_args()
    operator_api = load_operator_api(args.operator_file, API_NAME)
    run_bench(operator_api)


if __name__ == "__main__":
    main()
