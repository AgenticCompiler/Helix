from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path


FactPayload = dict[str, object]


def _new_str_set() -> set[str]:
    return set()


def _new_evidence_list() -> list[dict[str, object]]:
    return []


@dataclass
class FactCollector(ast.NodeVisitor):
    facts: set[str] = field(default_factory=_new_str_set)
    evidence: list[dict[str, object]] = field(default_factory=_new_evidence_list)
    _load_assigned_names: set[str] = field(default_factory=_new_str_set)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_tl_load(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._load_assigned_names.add(target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if self._looks_like_manual_reduction(node):
            self._record(
                "manual_k_reduction",
                node,
                "Loop carries an accumulator update across iterations while loading operands inside the loop body.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_index_based_load(node):
            self._record(
                "index_based_load",
                node,
                "A tl.load address expression reuses a value that was itself loaded earlier, indicating index-driven memory access.",
            )
        self.generic_visit(node)

    def _record(self, fact: str, node: ast.AST, detail: str) -> None:
        if fact in self.facts:
            return
        self.facts.add(fact)
        self.evidence.append(
            {
                "fact": fact,
                "line": getattr(node, "lineno", None),
                "detail": detail,
            }
        )

    def _looks_like_manual_reduction(self, node: ast.For) -> bool:
        if not isinstance(node.iter, ast.Call):
            return False
        if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            return False
        saw_accumulator_update = False
        saw_load = False
        for child in ast.walk(node):
            if self._is_tl_load(child):
                saw_load = True
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                saw_accumulator_update = True
        return saw_accumulator_update and saw_load

    def _is_index_based_load(self, node: ast.Call) -> bool:
        if not self._is_tl_load(node):
            return False
        if not node.args:
            return False
        address = node.args[0]
        for child in ast.walk(address):
            if isinstance(child, ast.Name) and child.id in self._load_assigned_names:
                return True
        return False

    def _is_tl_load(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "tl"
        )


def extract_code_facts(operator_path: Path) -> FactPayload:
    source = operator_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(operator_path))
    collector = FactCollector()
    collector.visit(tree)
    return {
        "source": "code",
        "path": str(operator_path),
        "facts": sorted(collector.facts),
        "evidence": collector.evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract structured code facts for optimize pattern triage."
    )
    parser.add_argument("operator_file")
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)

    payload = extract_code_facts(Path(args.operator_file))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
