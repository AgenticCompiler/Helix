from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_skill_script(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PatternRoutingToolTests(unittest.TestCase):
    def test_build_index_requires_summary_and_use_when(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "broken.md").write_text(
                "# Broken Pattern\n\n## Summary\n\nMissing use-when.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Use When"):
                module.build_index_text(patterns_dir)

    def test_build_index_keeps_free_sections_but_ignores_them_for_summary(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "demo.md").write_text(
                """---
id: demo
title: Demo Pattern
---

## Summary

Short summary.

## Use When

- A stable trigger exists.

## Background

Extra prose that should stay in the source file but not become a first-line index field.
""",
                encoding="utf-8",
            )
            rendered = module.build_index_text(patterns_dir)
            self.assertIn("demo", rendered)
            self.assertIn("Short summary.", rendered)
            self.assertNotIn("Extra prose", rendered)

    def test_checked_in_pattern_index_matches_generator(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
        )
        patterns_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "patterns"
        )
        generated = module.build_index_text(patterns_dir)
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "pattern_index.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)

    def test_generated_index_links_to_pattern_subdirectory(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "demo.md").write_text(
                "# Demo Pattern\n\n## Summary\n\nShort summary.\n\n## Use When\n\n- Stable trigger.\n",
                encoding="utf-8",
            )
            rendered = module.build_index_text(patterns_dir)
            self.assertIn("[demo.md](patterns/demo.md)", rendered)

    def test_generated_index_omits_post_apply_verification_section(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "demo.md").write_text(
                """# Demo Pattern

## Summary

Short summary.

## Use When

- Stable trigger.

## What To Verify After Applying

- Verify something important after the rewrite.
""",
                encoding="utf-8",
            )
            rendered = module.build_index_text(patterns_dir)
            self.assertNotIn("What To Verify After Applying", rendered)
            self.assertNotIn("Verify something important", rendered)

    def test_generated_index_omits_related_patterns_section(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "demo.md").write_text(
                """# Demo Pattern

## Summary

Short summary.

## Use When

- Stable trigger.

## Related Patterns

- `other-pattern`: Read it after this one when the structure is already normalized.
""",
                encoding="utf-8",
            )
            rendered = module.build_index_text(patterns_dir)
            self.assertNotIn("Related Patterns", rendered)
            self.assertNotIn("other-pattern", rendered)

    def test_build_index_ignores_pattern_directory_readme(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize/scripts/build_pattern_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            patterns_dir = Path(tmp)
            (patterns_dir / "README.md").write_text(
                "# Pattern Docs\n\nThis file explains authoring rules.\n",
                encoding="utf-8",
            )
            (patterns_dir / "demo.md").write_text(
                "## Summary\n\nShort summary.\n\n## Use When\n\n- Stable trigger.\n",
                encoding="utf-8",
            )

            rendered = module.build_index_text(patterns_dir)

            self.assertIn("Short summary.", rendered)
            self.assertNotIn("Pattern Docs", rendered)

    def test_build_symptom_index_requires_summary_evidence_and_candidates(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize-knowledge/scripts/build_symptom_index.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            symptoms_dir = Path(tmp)
            (symptoms_dir / "broken.md").write_text(
                "# broken\n\n## Summary\n\nMissing evidence and pattern directions.\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "Evidence To Confirm, Candidate Pattern Directions"
            ):
                module.build_index_text(symptoms_dir)

    def test_checked_in_symptom_index_matches_generator(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize-knowledge/scripts/build_symptom_index.py"
        )
        symptoms_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "symptoms"
        )
        generated = module.build_index_text(symptoms_dir)
        checked_in = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "symptom_index.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(generated, checked_in)

    def test_extract_code_facts_reports_manual_reduction_and_index_load(self) -> None:
        module = _load_skill_script(
            "skills/triton-npu-optimize/scripts/extract_code_facts.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            operator = Path(tmp) / "kernel.py"
            operator.write_text(
                """
import triton.language as tl

def kernel(x_ptr, idx_ptr):
    acc = 0
    for k in range(0, 128, 32):
        idx = tl.load(idx_ptr + k)
        val = tl.load(x_ptr + idx)
        acc += val
""",
                encoding="utf-8",
            )
            payload = module.extract_code_facts(operator)
            self.assertIn("manual_k_reduction", payload["facts"])
            self.assertIn("index_based_load", payload["facts"])
            self.assertNotIn("weak_pipeline_overlap", payload["facts"])


if __name__ == "__main__":
    unittest.main()
