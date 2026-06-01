import tempfile
import unittest
from pathlib import Path

from triton_agent.skills import SkillLinkManager
from triton_agent.skills_source_dir import build_skills_source_overrides

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SkillsSourceDirTests(unittest.TestCase):
    def test_prepare_skills_overwrites_workspace_knowledge_from_persistent_workdir(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            skills_install = Path(tmp) / "install-skills"
            install_knowledge = skills_install / "triton-npu-optimize-knowledge"
            (install_knowledge / "references" / "patterns").mkdir(parents=True)
            (install_knowledge / "references" / "patterns" / "from-install.md").write_text(
                "install\n",
                encoding="utf-8",
            )

            repo = Path(tmp) / "repo"
            skills_workdir = repo / "pattern-validation-skills"
            repo_knowledge = skills_workdir / "triton-npu-optimize-knowledge"
            (repo_knowledge / "references" / "patterns").mkdir(parents=True)
            (repo_knowledge / "references" / "patterns" / "from-loop.md").write_text(
                "loop\n",
                encoding="utf-8",
            )

            workspace = repo / "pattern-validation-batch" / "chunk_o"
            workspace.mkdir(parents=True)
            ws_knowledge = workspace / ".codex" / "skills" / "triton-npu-optimize-knowledge"
            (ws_knowledge / "references" / "patterns").mkdir(parents=True)
            (ws_knowledge / "references" / "patterns" / "stale.md").write_text(
                "stale\n",
                encoding="utf-8",
            )

            manager = SkillLinkManager(skills_install)
            overrides = build_skills_source_overrides(
                workspace,
                "codex",
                skills_workdir,
                ("triton-npu-optimize-knowledge",),
            )
            self.assertIsNotNone(overrides)
            manager.prepare_skills(
                "codex",
                workspace,
                skill_names=("triton-npu-optimize-knowledge",),
                skill_dir_overrides=overrides,
            )

            pattern_file = ws_knowledge / "references" / "patterns" / "from-loop.md"
            self.assertTrue(pattern_file.is_file())
            self.assertFalse((ws_knowledge / "references" / "patterns" / "stale.md").exists())


if __name__ == "__main__":
    unittest.main()
