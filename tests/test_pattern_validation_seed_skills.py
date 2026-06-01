import tempfile
import unittest
from pathlib import Path

from triton_agent.pattern_validation_loop.seed_skills import seed_pattern_validation_skills_dir

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SeedPatternValidationSkillsTests(unittest.TestCase):
    def test_seed_creates_persistent_workdir_once(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            install = Path(tmp) / "install-skills"
            source = install / "triton-npu-optimize-knowledge"
            (source / "references" / "patterns").mkdir(parents=True)
            (source / "references" / "patterns" / "base.md").write_text("base\n", encoding="utf-8")

            repo = Path(tmp) / "repo"
            workdir = seed_pattern_validation_skills_dir(
                repo,
                "pattern-validation-skills",
                install_root=install,
            )
            seeded = workdir / "triton-npu-optimize-knowledge" / "references" / "patterns" / "base.md"
            self.assertTrue(seeded.is_file())

            (workdir / "triton-npu-optimize-knowledge" / "references" / "patterns" / "edited.md").write_text(
                "edited\n",
                encoding="utf-8",
            )
            seed_pattern_validation_skills_dir(repo, "pattern-validation-skills", install_root=install)
            self.assertTrue(
                (workdir / "triton-npu-optimize-knowledge" / "references" / "patterns" / "edited.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
