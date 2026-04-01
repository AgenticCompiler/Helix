# bench-mode: standalone
# api-name: matmul
# kernel: matmul_kernel

import argparse
import importlib.util
from pathlib import Path

import torch
import triton

API_NAME = "matmul"
CASES = (
    {
        "case_id": "case-1",
        "seed": 101,
        "m": 16,
        "n": 16,
        "k": 16,
        "dtype": torch.float32,
    },
    {
        "case_id": "case-2",
        "seed": 202,
        "m": 32,
        "n": 48,
        "k": 32,
        "dtype": torch.float32,
    },
    {
        "case_id": "case-3",
        "seed": 303,
        "m": 64,
        "n": 64,
        "k": 32,
        "dtype": torch.float32,
    },
)


def load_operator_api(operator_file: str, api_name: str):
    operator_path = Path(operator_file)
    spec = importlib.util.spec_from_file_location("operator_module", operator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load operator module from {operator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return getattr(module, api_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Operator file does not define API '{api_name}': {operator_path}"
        ) from exc


def make_inputs(case: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(int(case["seed"]))
    dtype = case["dtype"]
    m = int(case["m"])
    n = int(case["n"])
    k = int(case["k"])
    a = torch.randn((m, k), device="npu", dtype=dtype)
    b = torch.randn((k, n), device="npu", dtype=dtype)
    return a, b


def select_bench_config(case: dict[str, object]) -> tuple[int, int]:
    # Heuristic runtime estimate from GEMM work. Small cases use the longer policy.
    m = int(case["m"])
    n = int(case["n"])
    k = int(case["k"])
    estimated_ops = 2 * m * n * k
    if estimated_ops < 2_000_000:
        return 1000, 10000
    return 100, 1000


def run_bench(operator_api) -> None:
    for case in CASES:
        case_id = str(case["case_id"])
        a, b = make_inputs(case)

        def bench_fn():
            return operator_api(a, b)

        warmup, active = select_bench_config(case)
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
