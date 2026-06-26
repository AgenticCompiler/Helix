from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPRESENTATIVE_RE = re.compile(r"^REPRESENTATIVE_INDICES\s*=\s*(\[.*?\])", re.MULTILINE)
_LATENCY_RE = re.compile(r"^latency-(?:case-)?(?P<id>[^:]+):\s*(?P<value>NA|[-+0-9.eE]+)\s*$")
_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


@dataclass(frozen=True)
class TensorFeature:
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= max(int(dim), 1)
        return total

    @property
    def byte_cost(self) -> float:
        return float(self.numel * _DTYPE_BYTES.get(self.dtype, 4))


@dataclass(frozen=True)
class CaseFeature:
    one_based_index: int
    tensors: tuple[TensorFeature, ...]
    attrs: tuple[tuple[str, str], ...]

    @property
    def cost(self) -> float:
        tensor_cost = sum(tensor.byte_cost for tensor in self.tensors)
        return max(tensor_cost, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    derive = subparsers.add_parser("derive")
    derive.add_argument("--cases-json", required=True)
    derive.add_argument("--bench-file", required=True)
    derive.add_argument("--full-perf")
    derive.add_argument("--output", default="case_weights.json")

    args = parser.parse_args(argv)
    if args.command == "derive":
        output = Path(args.output)
        payload = derive_weights(
            cases_json=Path(args.cases_json),
            bench_file=Path(args.bench_file),
            full_perf=Path(args.full_perf) if args.full_perf else None,
        )
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote case weights: {output}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def derive_weights(
    *,
    cases_json: Path,
    bench_file: Path,
    full_perf: Path | None = None,
) -> dict[str, Any]:
    cases = _load_case_features(cases_json)
    representative_indices = _load_representative_indices(bench_file)
    _validate_representatives(representative_indices, case_count=len(cases))
    representative_cases = [cases[index - 1] for index in representative_indices]
    weight_basis = "log_cost_size_bucket"

    assigned: dict[int, list[CaseFeature]] = {index: [] for index in representative_indices}
    for case in cases:
        nearest_index = min(
            representative_indices,
            key=lambda rep_index: _case_distance(case, cases[rep_index - 1]),
        )
        assigned[nearest_index].append(case)

    raw_weights: dict[int, float] = {}
    for rep_index, covered_cases in assigned.items():
        raw_weights[rep_index] = sum(
            _case_importance(case) for case in covered_cases
        )
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        raise ValueError("case weights sum to zero")

    weights: list[dict[str, Any]] = []
    for bench_case_id, rep_index in enumerate(representative_indices):
        covered_cases = assigned[rep_index]
        weight = raw_weights[rep_index] / total_weight
        rep_case = representative_cases[bench_case_id]
        weights.append(
            {
                "latency_id": f"latency-{bench_case_id}",
                "bench_case_id": str(bench_case_id),
                "representative_case_index": rep_index,
                "weight": weight,
                "covered_case_indices": [case.one_based_index for case in covered_cases],
                "covered_case_count": len(covered_cases),
                "representative_cost": rep_case.cost,
            }
        )

    return {
        "version": 1,
        "weight_basis": weight_basis,
        "cases_json": _metadata_path(cases_json),
        "bench_file": _metadata_path(bench_file),
        "full_perf": _metadata_path(full_perf),
        "total_case_count": len(cases),
        "representative_indices": representative_indices,
        "weights": weights,
    }


def _load_case_features(path: Path) -> list[CaseFeature]:
    cases: list[CaseFeature] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} case must be a JSON object")
        cases.append(_case_feature_from_payload(len(cases) + 1, payload))
    if not cases:
        raise ValueError(f"{path} did not contain any cases")
    return cases


def _case_feature_from_payload(one_based_index: int, payload: dict[str, Any]) -> CaseFeature:
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list):
        raise ValueError(f"case {one_based_index} is missing an inputs list")
    tensors: list[TensorFeature] = []
    attrs: list[tuple[str, str]] = []
    for raw_input in raw_inputs:
        if not isinstance(raw_input, dict):
            continue
        name = str(raw_input.get("name", ""))
        input_type = str(raw_input.get("type", ""))
        if input_type == "tensor":
            raw_shape = raw_input.get("shape")
            if not isinstance(raw_shape, list):
                raise ValueError(f"case {one_based_index} tensor {name} is missing shape")
            shape = tuple(int(dim) for dim in raw_shape)
            tensors.append(TensorFeature(dtype=str(raw_input.get("dtype", "")), shape=shape))
        else:
            attrs.append((name, json.dumps(raw_input.get("value"), sort_keys=True)))
    return CaseFeature(
        one_based_index=one_based_index,
        tensors=tuple(tensors),
        attrs=tuple(sorted(attrs)),
    )


def _load_representative_indices(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    match = _REPRESENTATIVE_RE.search(text)
    if match is None:
        raise ValueError(f"{path} does not define REPRESENTATIVE_INDICES")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("REPRESENTATIVE_INDICES must be a list of integers")
    if not value:
        raise ValueError("REPRESENTATIVE_INDICES must not be empty")
    return list(value)


def _validate_representatives(indices: list[int], *, case_count: int) -> None:
    invalid = [index for index in indices if index < 1 or index > case_count]
    if invalid:
        raise ValueError(f"representative indices outside 1..{case_count}: {invalid}")
    if len(set(indices)) != len(indices):
        raise ValueError("REPRESENTATIVE_INDICES contains duplicates")


def _metadata_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.name


def _parse_full_perf(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    values: dict[int, float] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        match = _LATENCY_RE.match(stripped)
        if match is None:
            parsed_value = _parse_jsonl_perf_value(stripped, path=path, line_no=line_no)
            if parsed_value is not None:
                case_index, value = parsed_value
                values[case_index] = value
            continue
        raw_id = match.group("id")
        raw_value = match.group("value")
        if raw_value == "NA" or not raw_id.isdigit():
            continue
        value = float(raw_value)
        if value > 0:
            values[int(raw_id) + 1] = value
        else:
            raise ValueError(f"{path}:{line_no} has non-positive latency {value}")
    if not values:
        raise ValueError(f"{path} did not contain usable numeric latency entries")
    return values


def _parse_jsonl_perf_value(line: str, *, path: Path, line_no: int) -> tuple[int, float] | None:
    if not line.startswith("{"):
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    raw_label = payload.get("case_label")
    if not isinstance(raw_label, str):
        return None
    match = re.match(r"^group-(?P<id>\d+)$", raw_label)
    if match is None:
        return None
    value = _jsonl_numeric_metric(payload.get("total_op_avg_time_us"))
    if value is None:
        value = _jsonl_numeric_metric(payload.get("kernel_avg_time_us"))
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{path}:{line_no} has non-positive latency {value}")
    return int(match.group("id")), value


def _jsonl_numeric_metric(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _case_importance(case: CaseFeature) -> float:
    return math.log2(case.cost)


def _size_bucket(case: CaseFeature) -> int:
    max_numel = max((tensor.numel for tensor in case.tensors), default=1)
    if max_numel < 1_000:
        return 0
    if max_numel < 100_000:
        return 1
    if max_numel < 1_000_000:
        return 2
    return 3


def _case_distance(left: CaseFeature, right: CaseFeature) -> float:
    distance = 0.0
    left_bucket = _size_bucket(left)
    right_bucket = _size_bucket(right)
    if left_bucket != right_bucket:
        distance += 3.0 * abs(left_bucket - right_bucket)
    distance += 0.3 * abs(math.log2(left.cost) - math.log2(right.cost))
    distance += 0.5 * abs(len(left.tensors) - len(right.tensors))
    distance += 0.25 * abs(len(left.attrs) - len(right.attrs))
    if left.attrs != right.attrs:
        distance += 1.0
    for left_tensor, right_tensor in zip(left.tensors, right.tensors):
        if left_tensor.dtype != right_tensor.dtype:
            distance += 1.0
        distance += 0.25 * abs(len(left_tensor.shape) - len(right_tensor.shape))
        for left_dim, right_dim in zip(left_tensor.shape, right_tensor.shape):
            distance += 0.1 * abs(math.log2(max(left_dim, 1)) - math.log2(max(right_dim, 1)))
    return distance


if __name__ == "__main__":
    raise SystemExit(main())
