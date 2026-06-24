import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script = (
        REPO_ROOT
        / "skills"
        / "triton"
        / "triton-npu-analyze-compiler-source"
        / "scripts"
        / "inspect_compiler_source.py"
    )
    spec = importlib.util.spec_from_file_location("inspect_compiler_source_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_source_tree(root: Path) -> Path:
    source_root = root / "AscendNPU-IR"
    (source_root / "docs/source/en/developer_guide/passes").mkdir(parents=True)
    (source_root / "bishengir/lib/Conversion/HFusionToHIVM").mkdir(parents=True)
    (source_root / "bishengir/include/bishengir/Conversion").mkdir(parents=True)
    (source_root / "bishengir/test/Conversion/HFusionToHIVM").mkdir(parents=True)

    (
        source_root / "docs/source/en/developer_guide/passes/HFusionPasses.md"
    ).write_text(
        "Auto vectorize and hfusion lowering notes.\n",
        encoding="utf-8",
    )
    (
        source_root / "bishengir/lib/Conversion/HFusionToHIVM/Vectorize.cpp"
    ).write_text(
        "void runVectorizePass();\n",
        encoding="utf-8",
    )
    (source_root / "bishengir/include/bishengir/Conversion/Passes.td").write_text(
        'def HFusionVectorizePass : Pass<"hfusion-vectorize">;\n',
        encoding="utf-8",
    )
    (
        source_root / "bishengir/test/Conversion/HFusionToHIVM/vectorize.mlir"
    ).write_text(
        "// not part of default search scope\n",
        encoding="utf-8",
    )
    return source_root


class InspectCompilerSourceTests(unittest.TestCase):
    def test_build_parser_parses_locate_arguments(self) -> None:
        module = _load_module()

        args = module.build_parser().parse_args(
            [
                "locate",
                "--source-root",
                "AscendNPU-IR",
                "--term",
                "hfusion",
                "--term",
                "vectorize",
                "--hint",
                "pass",
                "--format",
                "json",
            ]
        )

        self.assertEqual(args.command, "locate")
        self.assertEqual(args.source_root, "AscendNPU-IR")
        self.assertEqual(args.term, ["hfusion", "vectorize"])
        self.assertEqual(args.hint, "pass")
        self.assertEqual(args.format, "json")

    def test_locate_payload_groups_docs_lib_and_include_matches(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            source_root = _make_source_tree(Path(tmp))
            payload = module.locate_payload(
                source_root,
                terms=["hfusion", "vectorize"],
                hint="pass",
                limit=10,
            )

        self.assertIn("docs", payload)
        self.assertIn("lib", payload)
        self.assertIn("include", payload)
        self.assertEqual(payload["docs"][0]["area"], "docs")
        self.assertTrue(payload["docs"][0]["path"].endswith("HFusionPasses.md"))
        self.assertTrue(payload["lib"][0]["path"].endswith("Vectorize.cpp"))
        self.assertTrue(payload["include"][0]["path"].endswith("Passes.td"))

    def test_locate_payload_omits_test_scope_by_default(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            source_root = _make_source_tree(Path(tmp))
            payload = module.locate_payload(
                source_root,
                terms=["vectorize"],
                hint="conversion",
                limit=10,
            )

        rendered = json.dumps(payload, sort_keys=True)
        self.assertNotIn("bishengir/test", rendered)

    def test_locate_text_renders_grouped_candidates(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as tmp:
            source_root = _make_source_tree(Path(tmp))
            rendered = module.locate_text(
                source_root,
                terms=["vectorize"],
                hint="pass",
                limit=5,
            )

        self.assertIn("docs:", rendered)
        self.assertIn("lib:", rendered)
        self.assertIn("include:", rendered)
        self.assertIn("matched_terms=", rendered)


if __name__ == "__main__":
    unittest.main()
