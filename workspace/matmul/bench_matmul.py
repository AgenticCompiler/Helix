# bench-mode: standalone
# api-name: matmul
# kernel: matmul_kernel

import argparse
import importlib.util
import time
from pathlib import Path

import torch
import triton

API_NAME = "matmul"
CASES = (
    (
        "m256_n256_k256_fp16",
        {
            "m": 256,
            "n": 256,
            "k": 256,
            "dtype": torch.float16,
        },
    ),
    (
        "m512_n512_k512_fp16",
        {
            "m": 512,
            "n": 512,
            "k": 512,
            "dtype": torch.float16,
        },
    ),
    (
        "m1024_n512_k256_fp16",
        {
            "m": 1024,
            "n": 512,
            "k": 256,
            "dtype": torch.float16,
        },
    ),
)


def load_operator_api(operator_file: str, api_name: str):
    operator_path = Path(operator_file)
    spec = importlib.util.spec_from_file_location("operator_module", operator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load operator module from {operator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return getattr(module, api_name)
    except AttributeError as exc:
        raise AttributeError(f"Operator API '{api_name}' not found in {operator_path}") from exc


def make_inputs(case):
    dtype = case["dtype"]
    a = torch.randn((case["m"], case["k"]), device="npu", dtype=dtype)
    b = torch.randn((case["k"], case["n"]), device="npu", dtype=dtype)
    return a, b


def estimate_runtime_ms(bench_fn):
    start = time.perf_counter()
    bench_fn()
    end = time.perf_counter()
    return (end - start) * 1000.0


def select_bench_config(bench_fn):
    estimated_ms = estimate_runtime_ms(bench_fn)
    if estimated_ms < 10.0:
        return 1000, 10000
    return 100, 1000


def run_bench(operator_api):
    for case_id, case in CASES:
        inputs = make_inputs(case)

        def bench_fn():
            return operator_api(*inputs)

        warmup, active = select_bench_config(bench_fn)
        latency = triton.testing.do_bench_npu(bench_fn, warmup=warmup, active=active)
        print(f"latency-{case_id}: {latency}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", required=True)
    args = parser.parse_args()
    operator_api = load_operator_api(args.operator_file, API_NAME)
    run_bench(operator_api)


if __name__ == "__main__":
    main()
