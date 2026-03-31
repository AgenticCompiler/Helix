import torch
from pathlib import Path

from matmul import matmul


def test_matmul():
    results = []

    torch.manual_seed(42)

    case1_a = torch.randn(32, 64, dtype=torch.float32, device="npu")
    case1_b = torch.randn(64, 128, dtype=torch.float32, device="npu")
    case1_c = matmul(case1_a, case1_b)
    results.append(case1_c)

    torch.manual_seed(123)

    case2_a = torch.randn(16, 32, dtype=torch.float32, device="npu")
    case2_b = torch.randn(32, 64, dtype=torch.float32, device="npu")
    case2_c = matmul(case2_a, case2_b)
    results.append(case2_c)

    torch.manual_seed(456)

    case3_a = torch.randn(64, 128, dtype=torch.float32, device="npu")
    case3_b = torch.randn(128, 256, dtype=torch.float32, device="npu")
    case3_c = matmul(case3_a, case3_b)
    results.append(case3_c)

    torch.manual_seed(789)

    case4_a = torch.randn(128, 256, dtype=torch.float32, device="npu")
    case4_b = torch.randn(256, 64, dtype=torch.float32, device="npu")
    case4_c = matmul(case4_a, case4_b)
    results.append(case4_c)

    torch.save({"results": results}, Path(__file__).parent / "TEST_RESULT.pt")


if __name__ == "__main__":
    test_matmul()
