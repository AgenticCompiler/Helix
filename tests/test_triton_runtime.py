import tempfile
import unittest
from pathlib import Path

from helix.eval.triton_runtime import (
    REMOTE_TRITON_CACHE_ENV,
    TRITON_ALWAYS_COMPILE_ENV,
    TRITON_CACHE_DIR_ENV,
    cleanup_triton_runtime_session,
    prepare_triton_runtime_session,
    triton_runtime_env,
)


class TritonRuntimeTests(unittest.TestCase):
    def test_lease_environment_overrides_inherited_cache_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            session = prepare_triton_runtime_session(workspace, "run-one")

            env = triton_runtime_env(
                session,
                {
                    TRITON_CACHE_DIR_ENV: "/shared/cache",
                    TRITON_ALWAYS_COMPILE_ENV: "0",
                },
            )

            self.assertEqual(env[TRITON_CACHE_DIR_ENV], str(session.cache_dir))
            self.assertEqual(env[TRITON_ALWAYS_COMPILE_ENV], "1")
            self.assertEqual(env[REMOTE_TRITON_CACHE_ENV], "1")
            self.assertEqual(cleanup_triton_runtime_session(session), [])
            self.assertFalse(session.cache_dir.exists())
            self.assertFalse(session.cache_parent.exists())

    def test_cleanup_preserves_preexisting_parent_and_sibling_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            parent = workspace / ".helix-triton-cache"
            parent.mkdir()
            sibling = parent / "other-run"
            sibling.mkdir()
            session = prepare_triton_runtime_session(workspace, "run-one")

            self.assertEqual(cleanup_triton_runtime_session(session), [])
            self.assertTrue(parent.exists())
            self.assertTrue(sibling.exists())
            self.assertFalse(session.cache_dir.exists())

    def test_prepare_rejects_existing_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            first = prepare_triton_runtime_session(workspace, "run-one")

            with self.assertRaisesRegex(RuntimeError, "already exists"):
                prepare_triton_runtime_session(workspace, "run-one")

            self.assertEqual(cleanup_triton_runtime_session(first), [])


if __name__ == "__main__":
    unittest.main()
