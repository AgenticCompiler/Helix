import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skills.staging import SkillLinkManager
from triton_agent.skills.catalog import list_catalog_skill_names


_BACKEND_SKILL_DIRS = {
    "codex": (".codex", "skills"),
    "opencode": (".opencode", "skills"),
    "pi": (".pi", "skills"),
    "claude": (".claude", "skills"),
    "openhands": (".openhands", "skills"),
    "traecli": (".traecli", "skills"),
}


class SkillLinkManagerTests(unittest.TestCase):
    def test_catalog_contains_distill_skill_and_omits_legacy_bench_logs_skill(self) -> None:
        names = list_catalog_skill_names()

        self.assertIn("ascend-npu-distill-patterns", names)
        self.assertNotIn("ascend-npu-kernel-bench-logs", names)

    def test_optimize_propagate_nan_guidance_is_workflow_visible(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        optimize_skill = repo_root / "skills" / "triton" / "triton-npu-optimize" / "SKILL.md"
        vec_cmp = (
            repo_root
            / "skills"
            / "triton"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "patterns"
            / "vec-cmp.md"
        )

        skill_text = optimize_skill.read_text(encoding="utf-8")
        vec_cmp_text = vec_cmp.read_text(encoding="utf-8")
        semantic_repairs = skill_text.split("## Kernel Semantic Repairs", maxsplit=1)[1].split(
            "## Stage 3",
            maxsplit=1,
        )[0]

        self.assertIn("propagate_nan=tl.PropagateNan.ALL", semantic_repairs)
        self.assertIn("tl.maximum()", semantic_repairs)
        self.assertIn("tl.minimum()", semantic_repairs)
        self.assertIn("semantic choice", semantic_repairs)
        self.assertIn("propagate_nan=tl.PropagateNan.ALL", vec_cmp_text)
        self.assertIn("NaN-input behavior", vec_cmp_text)

    def test_repo_skills_stage_unified_optimize_state_skill_for_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            source = Path(__file__).resolve().parents[1] / "skills"

            manager = SkillLinkManager(source)
            links = manager.prepare_skills(
                "codex",
                workspace,
                skill_names=(
                    "triton-npu-optimize",
                    "ascend-npu-prepare-optimize-baseline",
                    "ascend-npu-optimize-state",
                    "ascend-npu-analyze-round-performance",
                ),
            )

            target = self._skills_target(workspace, "codex")
            self.assertTrue((target / "triton-npu-optimize" / "SKILL.md").exists())
            self.assertTrue((target / "ascend-npu-prepare-optimize-baseline" / "SKILL.md").exists())
            self.assertTrue((target / "ascend-npu-optimize-state" / "SKILL.md").exists())
            self.assertTrue((target / "ascend-npu-analyze-round-performance" / "SKILL.md").exists())
            manager.cleanup(links)

    def test_copy_only_requested_skill_dirs_when_backend_skills_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            for name in (
                "ascend-npu-gen-eval-suite",
                "ascend-npu-gen-test",
                "ascend-npu-gen-bench",
                "ascend-npu-run-eval",
                "triton-npu-optimize",
                "ascend-npu-analyze-round-performance",
            ):
                if name.startswith("triton-"):
                    (source / "triton" / name).mkdir(parents=True)
                    (source / "triton" / name / "SKILL.md").write_text(f"{name}\n", encoding="utf-8")
                else:
                    (source / "common" / name).mkdir(parents=True)
                    (source / "common" / name / "SKILL.md").write_text(f"{name}\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_skills(
                "traecli",
                workspace,
                skill_names=(
                    "ascend-npu-gen-eval-suite",
                    "ascend-npu-gen-test",
                    "ascend-npu-gen-bench",
                    "ascend-npu-run-eval",
                ),
            )

            target = self._skills_target(workspace, "traecli")
            backend_root = workspace / ".traecli"
            self.assertTrue((target / "ascend-npu-gen-eval-suite").exists())
            self.assertTrue((target / "ascend-npu-gen-test").exists())
            self.assertTrue((target / "ascend-npu-gen-bench").exists())
            self.assertTrue((target / "ascend-npu-run-eval").exists())
            self.assertFalse((target / "triton-npu-optimize").exists())
            self.assertFalse((target / "ascend-npu-analyze-round-performance").exists())
            self.assertEqual(links.created_paths, [backend_root, target])
            manager.cleanup(links)
            self.assertFalse(target.exists())
            self.assertFalse(backend_root.exists())

    def test_reject_missing_requested_skill_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test\n", encoding="utf-8")

            manager = SkillLinkManager(source)

            with self.assertRaisesRegex(RuntimeError, "Requested skill does not exist"):
                manager.prepare_skills("codex", workspace, skill_names=("missing-skill",))

    def test_prepare_skills_can_stage_alternate_source_under_stable_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton" / "triton-npu-optimize-knowledge-v2").mkdir(parents=True)
            (source / "triton" / "triton-npu-optimize-knowledge-v2" / "SKILL.md").write_text(
                "v2 knowledge\n",
                encoding="utf-8",
            )

            manager = SkillLinkManager(source)
            links = manager.prepare_skills(
                "codex",
                workspace,
                skill_names=("triton-npu-optimize-knowledge",),
                skill_sources={
                    "triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2",
                },
            )

            target = self._skills_target(workspace, "codex")
            staged_dir = target / "triton-npu-optimize-knowledge"
            self.assertTrue(staged_dir.is_dir())
            self.assertEqual(
                (staged_dir / "SKILL.md").read_text(encoding="utf-8"),
                "v2 knowledge\n",
            )
            self.assertFalse((target / "triton-npu-optimize-knowledge-v2").exists())
            manager.cleanup(links)

    def test_prepare_skills_can_stage_v3_alternate_source_under_stable_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "triton" / "triton-npu-optimize-knowledge-v3").mkdir(parents=True)
            (source / "triton" / "triton-npu-optimize-knowledge-v3" / "SKILL.md").write_text(
                "v3 knowledge\n",
                encoding="utf-8",
            )

            manager = SkillLinkManager(source)
            links = manager.prepare_skills(
                "codex",
                workspace,
                skill_names=("triton-npu-optimize-knowledge",),
                skill_sources={
                    "triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v3",
                },
            )

            target = self._skills_target(workspace, "codex")
            staged_dir = target / "triton-npu-optimize-knowledge"
            self.assertTrue(staged_dir.is_dir())
            self.assertEqual(
                (staged_dir / "SKILL.md").read_text(encoding="utf-8"),
                "v3 knowledge\n",
            )
            self.assertFalse((target / "triton-npu-optimize-knowledge-v3").exists())
            manager.cleanup(links)

    def test_prepare_skills_preserves_temporary_git_repo_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_skills("codex", workspace, skill_names=("ascend-npu-gen-test",))

            temporary_git_dir = workspace / ".git"
            self.assertTrue(temporary_git_dir.is_dir())
            self.assertEqual(links.temporary_git_dir, temporary_git_dir)
            manager.cleanup(links)
            self.assertTrue(temporary_git_dir.exists())

    def test_prepare_skills_cleans_temporary_git_repo_when_reset_env_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_skills("codex", workspace, skill_names=("ascend-npu-gen-test",))

            temporary_git_dir = workspace / ".git"
            self.assertTrue(temporary_git_dir.is_dir())
            with mock.patch.dict("os.environ", {"TRITON_AGENT_RESET_GIT_REPO": "1"}, clear=False):
                manager.cleanup(links)
            self.assertFalse(temporary_git_dir.exists())

    def test_prepare_skills_preserves_existing_local_git_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (workspace / ".git").mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_skills("codex", workspace, skill_names=("ascend-npu-gen-test",))

            self.assertIsNone(links.temporary_git_dir)
            manager.cleanup(links)
            self.assertTrue((workspace / ".git").exists())

    def test_prepare_skills_creates_local_git_repo_even_under_parent_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo-root"
            workspace = repo_root / "workspace"
            source = Path(tmp) / "skills-source"
            repo_root.mkdir()
            workspace.mkdir()
            source.mkdir()
            (repo_root / ".git").mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            links = manager.prepare_skills("codex", workspace, skill_names=("ascend-npu-gen-test",))

            self.assertTrue((workspace / ".git").is_dir())
            self.assertEqual(links.temporary_git_dir, workspace / ".git")
            manager.cleanup(links)
            self.assertTrue((workspace / ".git").exists())
            self.assertTrue((repo_root / ".git").exists())

    def test_prepare_skills_rolls_back_temporary_git_repo_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            with mock.patch.object(
                manager,
                "_copy_selected_skill_dirs",
                side_effect=RuntimeError("copy failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    manager.prepare_skills("codex", workspace, skill_names=("ascend-npu-gen-test",))

            self.assertFalse((workspace / ".git").exists())

    def test_prepare_skills_skips_temporary_git_repo_when_git_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            workspace.mkdir()
            source.mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test skill\n", encoding="utf-8")

            manager = SkillLinkManager(source)
            with mock.patch("triton_agent.skills.staging.shutil.which", return_value=None):
                links = manager.prepare_skills("codex", workspace, skill_names=("ascend-npu-gen-test",))

            self.assertIsNone(links.temporary_git_dir)
            self.assertFalse((workspace / ".git").exists())
            staged_skill = self._skills_target(workspace, "codex") / "ascend-npu-gen-test" / "SKILL.md"
            self.assertTrue(staged_skill.exists())
            manager.cleanup(links)

    def test_copy_root_skills_directory_when_missing_for_supported_backends(self) -> None:
        for backend in _BACKEND_SKILL_DIRS:
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = Path(tmp) / "workspace"
                    source = Path(tmp) / "skills-source"
                    workspace.mkdir()
                    source.mkdir()
                    (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
                    (source / "common" / "ascend-npu-optimize-state").mkdir(parents=True)
                    (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text(
                        "test skill\n",
                        encoding="utf-8",
                    )
                    (source / "common" / "ascend-npu-optimize-state" / "SKILL.md").write_text(
                        "optimize-state skill\n",
                        encoding="utf-8",
                    )

                    manager = SkillLinkManager(source)
                    links = manager.prepare_skills(backend, workspace)

                    target = self._skills_target(workspace, backend)
                    copied_skill = target / "ascend-npu-gen-test" / "SKILL.md"
                    copied_optimize_state_skill = target / "ascend-npu-optimize-state" / "SKILL.md"
                    self.assertTrue(target.is_dir())
                    self.assertFalse(target.is_symlink())
                    self.assertEqual(copied_skill.read_text(encoding="utf-8"), "test skill\n")
                    self.assertEqual(
                        copied_optimize_state_skill.read_text(encoding="utf-8"),
                        "optimize-state skill\n",
                    )
                    if backend == "opencode":
                        self.assertEqual(
                            {path.name for path in links.created_paths},
                            {".opencode", "ascend-npu-gen-test", "ascend-npu-optimize-state"},
                        )
                    else:
                        backend_root = workspace / _BACKEND_SKILL_DIRS[backend][0]
                        self.assertEqual(links.created_paths, [backend_root, target])
                    self.assertTrue(
                        any("created skill copy" in message for message in manager.describe_prepare(links))
                    )
                    self.assertTrue(
                        any("removed skill copy" in message for message in manager.describe_cleanup(links))
                    )
                    manager.cleanup(links)
                    backend_root = workspace / _BACKEND_SKILL_DIRS[backend][0]
                    self.assertFalse(target.exists())
                    self.assertFalse(backend_root.exists())

    def test_copy_missing_per_skill_dirs_when_skills_dir_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            source = Path(tmp) / "skills-source"
            existing_skills = self._skills_target(workspace, "codex")
            workspace.mkdir()
            existing_skills.mkdir(parents=True)
            source.mkdir()
            (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-bench").mkdir(parents=True)
            (source / "common" / "ascend-npu-gen-test" / "SKILL.md").write_text("test\n", encoding="utf-8")
            (source / "common" / "ascend-npu-gen-bench" / "SKILL.md").write_text("bench\n", encoding="utf-8")
            (existing_skills / "user-skill").mkdir()

            manager = SkillLinkManager(source)
            links = manager.prepare_skills("codex", workspace)

            test_gen_dir = existing_skills / "ascend-npu-gen-test"
            bench_gen_dir = existing_skills / "ascend-npu-gen-bench"
            self.assertTrue(test_gen_dir.is_dir())
            self.assertTrue(bench_gen_dir.is_dir())
            self.assertFalse(test_gen_dir.is_symlink())
            self.assertFalse(bench_gen_dir.is_symlink())
            self.assertTrue((existing_skills / "user-skill").exists())
            self.assertEqual({path.name for path in links.created_paths}, {"ascend-npu-gen-test", "ascend-npu-gen-bench"})
            manager.cleanup(links)
            self.assertFalse(test_gen_dir.exists())
            self.assertFalse(bench_gen_dir.exists())
            self.assertTrue((existing_skills / "user-skill").exists())

    def test_reject_existing_root_skills_symlink_for_root_copy_backends(self) -> None:
        for backend in ("codex", "pi", "claude", "openhands", "traecli"):
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = Path(tmp) / "workspace"
                    source = Path(tmp) / "skills-source"
                    target = self._skills_target(workspace, backend)
                    workspace.mkdir()
                    source.mkdir()
                    (source / "common" / "ascend-npu-gen-test").mkdir(parents=True)
                    target.parent.mkdir(parents=True)
                    try:
                        target.symlink_to(source, target_is_directory=True)
                    except OSError as exc:
                        self.skipTest(f"directory symlinks are unavailable: {exc}")

                    manager = SkillLinkManager(source)

                    with self.assertRaises(RuntimeError):
                        manager.prepare_skills(backend, workspace)

    def test_reject_existing_per_skill_symlink_for_per_skill_copy_backends(self) -> None:
        for backend in ("opencode",):
            with self.subTest(backend=backend):
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = Path(tmp) / "workspace"
                    source = Path(tmp) / "skills-source"
                    link_path = self._skills_target(workspace, backend) / "ascend-npu-gen-test"
                    workspace.mkdir()
                    source.mkdir()
                    skill_dir = source / "common" / "ascend-npu-gen-test"
                    skill_dir.mkdir(parents=True)
                    link_path.parent.mkdir(parents=True)
                    try:
                        link_path.symlink_to(skill_dir, target_is_directory=True)
                    except OSError as exc:
                        self.skipTest(f"directory symlinks are unavailable: {exc}")

                    manager = SkillLinkManager(source)

                    with self.assertRaises(RuntimeError):
                        manager.prepare_skills(backend, workspace)

    def test_repo_skills_include_unified_optimize_state_skill_for_codex_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            source = Path(__file__).resolve().parents[1] / "skills"

            manager = SkillLinkManager(source)
            links = manager.prepare_skills(
                "codex",
                workspace,
                skill_names=(
                    "triton-npu-optimize",
                    "ascend-npu-optimize-state",
                ),
            )

            target = self._skills_target(workspace, "codex")
            self.assertTrue((target / "triton-npu-optimize" / "SKILL.md").exists())
            self.assertTrue((target / "ascend-npu-optimize-state" / "SKILL.md").exists())
            manager.cleanup(links)

    def _skills_target(self, workspace: Path, backend: str) -> Path:
        return workspace.joinpath(*_BACKEND_SKILL_DIRS[backend])


if __name__ == "__main__":
    unittest.main()
