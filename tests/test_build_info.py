import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.build_info import (  # noqa: E402
    _find_git_root,
    _load_embedded_commit,
    _resolve_source_checkout_commit,
    get_build_commit,
    get_build_info_display,
)


class FindGitRootTests(unittest.TestCase):
    def test_finds_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".git").mkdir()
            found = _find_git_root(root / "src" / "helix")
            self.assertEqual(found, root)

    def test_finds_git_file_for_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / ".git").write_text("gitdir: /some/other/path")
            found = _find_git_root(root / "src" / "helix")
            self.assertEqual(found, root)

    def test_returns_none_when_no_git_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            found = _find_git_root(root / "src" / "helix")
            self.assertIsNone(found)


class ResolveSourceCheckoutCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        get_build_commit.cache_clear()

    def test_resolves_head_from_git_rev_parse(self) -> None:
        fake_commit = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = fake_commit + "\n"
                result = _resolve_source_checkout_commit()
                self.assertEqual(result, fake_commit)

    def test_returns_none_when_no_git_repo(self) -> None:
        with patch(
            "helix.build_info._find_git_root",
            return_value=None,
        ):
            result = _resolve_source_checkout_commit()
            self.assertIsNone(result)

    def test_returns_none_when_git_command_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            with patch("subprocess.run", side_effect=OSError("not found")):
                result = _resolve_source_checkout_commit()
                self.assertIsNone(result)

    def test_returns_none_when_git_rev_parse_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stdout = ""
                result = _resolve_source_checkout_commit()
                self.assertIsNone(result)

    def test_returns_none_when_git_output_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "\n"
                result = _resolve_source_checkout_commit()
                self.assertIsNone(result)


class LoadEmbeddedCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        get_build_commit.cache_clear()

    def test_loads_commit_from_valid_json(self) -> None:
        fake_commit = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "_build_meta.json"
            meta_path.write_text(json.dumps({"git_commit": fake_commit}))
            with patch.object(Path, "with_name", return_value=meta_path):
                result = _load_embedded_commit()
                self.assertEqual(result, fake_commit)

    def test_returns_none_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "_build_meta.json"
            with patch.object(Path, "with_name", return_value=meta_path):
                result = _load_embedded_commit()
                self.assertIsNone(result)

    def test_returns_none_when_json_corrupted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "_build_meta.json"
            meta_path.write_text("{not valid json")
            with patch.object(Path, "with_name", return_value=meta_path):
                result = _load_embedded_commit()
                self.assertIsNone(result)

    def test_returns_none_when_key_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "_build_meta.json"
            meta_path.write_text(json.dumps({"other": "value"}))
            with patch.object(Path, "with_name", return_value=meta_path):
                result = _load_embedded_commit()
                self.assertIsNone(result)

    def test_returns_none_when_commit_is_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "_build_meta.json"
            meta_path.write_text(json.dumps({"git_commit": ""}))
            with patch.object(Path, "with_name", return_value=meta_path):
                result = _load_embedded_commit()
                self.assertIsNone(result)

    def test_returns_none_when_value_is_not_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "_build_meta.json"
            meta_path.write_text(json.dumps({"git_commit": 12345}))
            with patch.object(Path, "with_name", return_value=meta_path):
                result = _load_embedded_commit()
                self.assertIsNone(result)


class GetBuildCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        get_build_commit.cache_clear()

    def test_source_checkout_wins_over_embedded(self) -> None:
        fake_source = "c" * 40
        fake_embedded = "d" * 40
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value=fake_source,
        ), patch(
            "helix.build_info._load_embedded_commit",
            return_value=fake_embedded,
        ), patch(
            "helix.build_info._is_installed_package",
            return_value=False,
        ):
            result = get_build_commit()
            self.assertEqual(result, fake_source)

    def test_installed_package_prefers_embedded_over_source(self) -> None:
        fake_source = "c" * 40
        fake_embedded = "d" * 40
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value=fake_source,
        ), patch(
            "helix.build_info._load_embedded_commit",
            return_value=fake_embedded,
        ), patch(
            "helix.build_info._is_installed_package",
            return_value=True,
        ):
            result = get_build_commit()
            self.assertEqual(result, fake_embedded)

    def test_installed_package_falls_back_to_unknown(self) -> None:
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value="c" * 40,
        ), patch(
            "helix.build_info._load_embedded_commit",
            return_value=None,
        ), patch(
            "helix.build_info._is_installed_package",
            return_value=True,
        ):
            result = get_build_commit()
            self.assertIsNone(result)

    def test_falls_back_to_embedded(self) -> None:
        fake_embedded = "e" * 40
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value=None,
        ), patch(
            "helix.build_info._load_embedded_commit",
            return_value=fake_embedded,
        ):
            result = get_build_commit()
            self.assertEqual(result, fake_embedded)

    def test_returns_none_when_both_unavailable(self) -> None:
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value=None,
        ), patch(
            "helix.build_info._load_embedded_commit",
            return_value=None,
        ):
            result = get_build_commit()
            self.assertIsNone(result)

    def test_memoizes_result(self) -> None:
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value="f" * 40,
        ) as mock_resolve:
            first = get_build_commit()
            second = get_build_commit()
            self.assertEqual(first, second)
            self.assertEqual(mock_resolve.call_count, 1)


class GetBuildInfoDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        get_build_commit.cache_clear()

    def test_shortens_40_char_to_12(self) -> None:
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value="a" * 40,
        ):
            result = get_build_info_display()
            self.assertEqual(result, "a" * 12)
            self.assertEqual(len(result), 12)

    def test_returns_unknown_when_no_commit(self) -> None:
        with patch(
            "helix.build_info._resolve_source_checkout_commit",
            return_value=None,
        ), patch(
            "helix.build_info._load_embedded_commit",
            return_value=None,
        ):
            result = get_build_info_display()
            self.assertEqual(result, "unknown")
