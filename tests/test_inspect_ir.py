import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_inspect_ir_module():
    script = (
        REPO_ROOT
        / "skills"
        / "triton-npu-analyze-ir"
        / "scripts"
        / "inspect_ir.py"
    )
    spec = importlib.util.spec_from_file_location("inspect_ir_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_archive(root: Path) -> Path:
    archive_dir = root / "ir-archive"
    stage_dir = archive_dir / "bishengir_stages" / "builtin_module_no-symbol-name"
    stage_dir.mkdir(parents=True)
    (archive_dir / "capture-manifest.json").write_text("{}\n", encoding="utf-8")

    (stage_dir / "10_hivm-plan-memory.mlir").write_text(
        "\n".join(
            [
                "func.func @kernel() {",
                "  %0 = memref.alloc() : memref<4xf32>",
                "  %1 = memref.alloc() : memref<4xf32>",
                "  memref.copy %0, %1 : memref<4xf32> to memref<4xf32>",
                "  return",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (stage_dir / "20_hfusion-auto-vectorize-v2.mlir").write_text(
        "\n".join(
            [
                "func.func @kernel() {",
                "  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xf32>, vector<8xf32>",
                "  %1 = arith.mulf %0, %0 : vector<8xf32>",
                "  vector.transfer_write %1, %arg1[%c0] : vector<8xf32>, memref<16xf32>",
                "  return",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (stage_dir / "30_hivm-inject-sync.mlir").write_text(
        "\n".join(
            [
                "func.func @kernel() {",
                "  hivm.set_flag %arg0 : index",
                "  hivm.wait_flag %arg0 : index",
                "  hivm.barrier",
                "  return",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_dir


class InspectIrTests(unittest.TestCase):
    def test_build_parser_uses_ir_dir_for_subcommands(self) -> None:
        module = _load_inspect_ir_module()

        list_args = module.build_parser().parse_args(["list-stages", "--ir-dir", "ir"])
        summary_args = module.build_parser().parse_args(
            ["stage-summary", "--ir-dir", "ir", "--stage", "10_hivm-plan-memory"]
        )
        diff_args = module.build_parser().parse_args(
            [
                "diff-stages",
                "--ir-dir",
                "ir",
                "--from",
                "10_hivm-plan-memory",
                "--to",
                "20_hfusion-auto-vectorize-v2",
            ]
        )
        change_args = module.build_parser().parse_args(["find-changes", "--ir-dir", "ir"])
        signal_args = module.build_parser().parse_args(["performance-signals", "--ir-dir", "ir"])

        self.assertEqual(list_args.ir_dir, "ir")
        self.assertEqual(summary_args.ir_dir, "ir")
        self.assertEqual(diff_args.ir_dir, "ir")
        self.assertEqual(change_args.ir_dir, "ir")
        self.assertEqual(signal_args.ir_dir, "ir")

    def test_list_stages_renders_sorted_stages_and_supports_grep_and_limit(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))

            all_output = module.list_stages_text(archive_dir)
            filtered_output = module.list_stages_text(archive_dir, grep="vectorize", limit=1)

        self.assertIn("10_hivm-plan-memory", all_output)
        self.assertIn("20_hfusion-auto-vectorize-v2", all_output)
        self.assertIn("30_hivm-inject-sync", all_output)
        self.assertLess(
            all_output.index("10_hivm-plan-memory"),
            all_output.index("20_hfusion-auto-vectorize-v2"),
        )
        self.assertIn("20_hfusion-auto-vectorize-v2", filtered_output)
        self.assertNotIn("10_hivm-plan-memory", filtered_output)

    def test_list_stages_supports_sort_by_size_and_interesting(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            by_size = module.list_stages_text(archive_dir, sort_by="size")
            by_interesting = module.list_stages_text(archive_dir, sort_by="interesting")

        size_lines = [line for line in by_size.splitlines() if line.strip() and not line.startswith("Stages:")]
        interesting_lines = [
            line for line in by_interesting.splitlines() if line.strip() and not line.startswith("Stages:")
        ]
        self.assertTrue(size_lines[0].strip().startswith("20_hfusion-auto-vectorize-v2"))
        self.assertIn("score=", interesting_lines[0])
        self.assertTrue(interesting_lines[0].strip().startswith("30_hivm-inject-sync"))

    def test_resolve_stage_selector_supports_relative_path_stem_and_unique_substring(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            stages_dir = module.resolve_stages_dir(archive_dir)
            by_rel = module.resolve_stage_selector(
                stages_dir,
                "builtin_module_no-symbol-name/10_hivm-plan-memory",
            )
            by_stem = module.resolve_stage_selector(stages_dir, "20_hfusion-auto-vectorize-v2")
            by_substring = module.resolve_stage_selector(stages_dir, "inject-sync")

        self.assertEqual(by_rel.name, "10_hivm-plan-memory.mlir")
        self.assertEqual(by_stem.name, "20_hfusion-auto-vectorize-v2.mlir")
        self.assertEqual(by_substring.name, "30_hivm-inject-sync.mlir")

    def test_stage_summary_renders_keyword_counts_and_highlights(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            rendered = module.stage_summary_text(archive_dir, "10_hivm-plan-memory")

        self.assertIn("Stage:", rendered)
        self.assertIn("10_hivm-plan-memory", rendered)
        self.assertIn("Keyword counts:", rendered)
        self.assertIn("alloc: 2", rendered)
        self.assertIn("copy: 1", rendered)
        self.assertIn("Highlights:", rendered)

    def test_diff_stages_renders_header_deltas_and_unified_diff(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            rendered = module.diff_stages_text(
                archive_dir,
                from_selector="10_hivm-plan-memory",
                to_selector="20_hfusion-auto-vectorize-v2",
                context=1,
            )

        self.assertIn("From: 10_hivm-plan-memory", rendered)
        self.assertIn("To: 20_hfusion-auto-vectorize-v2", rendered)
        self.assertIn("Keyword deltas:", rendered)
        self.assertIn("alloc:", rendered)
        self.assertIn("vector:", rendered)
        self.assertIn("---", rendered)
        self.assertIn("+++", rendered)

    def test_find_changes_renders_adjacent_stage_ranking(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            rendered = module.find_changes_text(archive_dir, limit=2)

        self.assertIn("Adjacent stage changes:", rendered)
        self.assertIn("10_hivm-plan-memory -> 20_hfusion-auto-vectorize-v2", rendered)
        self.assertIn("score=", rendered)
        self.assertIn("keyword deltas:", rendered)

    def test_performance_signals_renders_stage_and_transition_summaries(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            rendered = module.performance_signals_text(archive_dir, limit=2)

        self.assertIn("Performance signals:", rendered)
        self.assertIn("Vector-heavy stages:", rendered)
        self.assertIn("Transfer-heavy stages:", rendered)
        self.assertIn("Sync-heavy stages:", rendered)
        self.assertIn("Suspicious transitions:", rendered)
        self.assertIn("20_hfusion-auto-vectorize-v2", rendered)
        self.assertIn("30_hivm-inject-sync", rendered)

    def test_performance_signals_supports_json_output(self) -> None:
        module = _load_inspect_ir_module()

        with tempfile.TemporaryDirectory() as tmp:
            archive_dir = _make_archive(Path(tmp))
            rendered = module.performance_signals_text(
                archive_dir,
                limit=2,
                output_format="json",
            )

        payload = json.loads(rendered)
        self.assertIn("stage_summaries", payload)
        self.assertIn("vector_heavy_stages", payload)
        self.assertIn("transfer_heavy_stages", payload)
        self.assertIn("sync_heavy_stages", payload)
        self.assertIn("suspicious_transitions", payload)
        self.assertEqual(payload["vector_heavy_stages"][0]["stage"], "20_hfusion-auto-vectorize-v2")


if __name__ == "__main__":
    unittest.main()
