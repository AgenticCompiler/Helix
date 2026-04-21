import tempfile
import unittest
from pathlib import Path
from typing import Optional

from triton_agent.optimize.compiler_source import (
    COMPILER_SOURCE_REPO_URL,
    CompilerSourceInfo,
    default_compiler_source_path,
    prepare_compiler_source,
)


class CompilerSourceTests(unittest.TestCase):
    def test_default_compiler_source_path_uses_triton_agent_home(self) -> None:
        root = Path("/tmp/fake-home")

        path = default_compiler_source_path(root)

        self.assertEqual(path, root / "compiler-sources" / "AscendNPU-IR")

    def test_prepare_returns_none_when_mode_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[list[str]] = []

            result = prepare_compiler_source(
                mode="off",
                triton_agent_home=Path(tmp),
                run_git=lambda args, cwd=None: calls.append(args) or "",
            )

            self.assertIsNone(result)
            self.assertEqual(calls, [])

    def test_prepare_clones_missing_default_checkout_depth_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / ".triton-agent"
            calls: list[list[str]] = []

            def fake_run(args: list[str], cwd: Optional[Path] = None) -> str:
                del cwd
                calls.append(args)
                if args[:2] == ["git", "clone"]:
                    target = Path(args[-1])
                    target.mkdir(parents=True)
                    (target / ".git").mkdir()
                    return ""
                if args == ["git", "rev-parse", "HEAD"]:
                    return "abc123\n"
                raise AssertionError(args)

            result = prepare_compiler_source(
                mode="auto",
                triton_agent_home=home,
                run_git=fake_run,
            )

            self.assertEqual(
                result,
                CompilerSourceInfo(
                    path=home / "compiler-sources" / "AscendNPU-IR",
                    commit="abc123",
                ),
            )
            self.assertIn(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    COMPILER_SOURCE_REPO_URL,
                    str(home / "compiler-sources" / "AscendNPU-IR"),
                ],
                calls,
            )

    def test_prepare_reuses_existing_checkout_without_fetch_or_pull(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            checkout = home / "compiler-sources" / "AscendNPU-IR"
            checkout.mkdir(parents=True)
            (checkout / ".git").mkdir()
            calls: list[list[str]] = []

            def fake_run(args: list[str], cwd: Optional[Path] = None) -> str:
                self.assertEqual(cwd, checkout)
                calls.append(args)
                if args == ["git", "rev-parse", "HEAD"]:
                    return "def456\n"
                raise AssertionError(args)

            result = prepare_compiler_source(
                mode="auto",
                triton_agent_home=home,
                run_git=fake_run,
            )

            self.assertEqual(
                result,
                CompilerSourceInfo(path=checkout, commit="def456"),
            )
            self.assertNotIn(["git", "fetch"], calls)
            self.assertNotIn(["git", "pull"], calls)
            self.assertNotIn(["git", "status", "--porcelain"], calls)
            self.assertTrue(all(call[:2] != ["git", "clone"] for call in calls))

    def test_prepare_rejects_file_checkout_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / "compiler-sources" / "AscendNPU-IR"
            path.parent.mkdir(parents=True)
            path.write_text("not a directory\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not a directory"):
                prepare_compiler_source(mode="auto", triton_agent_home=home)

    def test_prepare_cloned_checkout_must_be_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            calls: list[list[str]] = []

            def fake_run(args: list[str], cwd: Optional[Path] = None) -> str:
                del cwd
                calls.append(args)
                if args[:2] == ["git", "clone"]:
                    target = Path(args[-1])
                    target.mkdir(parents=True)
                    return ""
                raise AssertionError(args)

            with self.assertRaisesRegex(ValueError, "git checkout"):
                prepare_compiler_source(mode="auto", triton_agent_home=home, run_git=fake_run)

            self.assertTrue(calls)

    def test_prepare_rejects_non_git_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            path = home / "compiler-sources" / "AscendNPU-IR"
            path.parent.mkdir(parents=True)
            path.mkdir()

            with self.assertRaisesRegex(ValueError, "git checkout"):
                prepare_compiler_source(mode="auto", triton_agent_home=home)


if __name__ == "__main__":
    unittest.main()
