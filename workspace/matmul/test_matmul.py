import argparse
import importlib.util
from pathlib import Path
import torch

# test-mode: differential
# api-name: matmul
# kernel: matmul_kernel

API_NAME = "matmul"


def _load_operator_api(operator_file: str):
    spec = importlib.util.spec_from_file_location("operator_module", operator_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load operator module from {operator_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, API_NAME)
    except AttributeError as exc:
        raise RuntimeError(f"Operator file does not define API '{API_NAME}': {operator_file}") from exc


def test_matmul(operator_api):
    results = []

    torch.manual_seed(42)

    case1_a = torch.randn(32, 64, dtype=torch.float32, device="npu")
    case1_b = torch.randn(64, 128, dtype=torch.float32, device="npu")
    case1_c = operator_api(case1_a, case1_b)
    results.append(case1_c)

    torch.manual_seed(123)

    case2_a = torch.randn(16, 32, dtype=torch.float32, device="npu")
    case2_b = torch.randn(32, 64, dtype=torch.float32, device="npu")
    case2_c = operator_api(case2_a, case2_b)
    results.append(case2_c)

    torch.manual_seed(456)

    case3_a = torch.randn(64, 128, dtype=torch.float32, device="npu")
    case3_b = torch.randn(128, 256, dtype=torch.float32, device="npu")
    case3_c = operator_api(case3_a, case3_b)
    results.append(case3_c)

    torch.manual_seed(789)

    case4_a = torch.randn(128, 256, dtype=torch.float32, device="npu")
    case4_b = torch.randn(256, 64, dtype=torch.float32, device="npu")
    case4_c = operator_api(case4_a, case4_b)
    results.append(case4_c)

    torch.save({"results": results}, Path(__file__).parent / "TEST_RESULT.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", required=True)
    args = parser.parse_args()
    test_matmul(_load_operator_api(args.operator_file))


if __name__ == "__main__":
    main()
