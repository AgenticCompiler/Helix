import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize_upload.models import UploadIdentity
from triton_agent.optimize_upload.naming import slugify_workspace_name, build_upload_identity
from triton_agent.optimize_upload.collector import collect_workspace_upload_files
from triton_agent.optimize_upload.manifest import build_manifest


class OptimizeUploadNamingTests(unittest.TestCase):
    def test_slugify_workspace_name_preserves_safe_characters(self) -> None:
        self.assertEqual(slugify_workspace_name("matmul_case-01"), "matmul_case-01")

    def test_slugify_workspace_name_replaces_unsafe_characters(self) -> None:
        self.assertEqual(slugify_workspace_name("matmul case/01"), "matmul_case_01")

    def test_slugify_workspace_name_replaces_multiple_unsafe(self) -> None:
        self.assertEqual(slugify_workspace_name("hello world!!test"), "hello_world_test")

    def test_slugify_workspace_name_collapses_repeated_underscores(self) -> None:
        self.assertEqual(slugify_workspace_name("a___b"), "a_b")

    def test_slugify_workspace_name_falls_back_to_workspace(self) -> None:
        self.assertEqual(slugify_workspace_name("////"), "workspace")

    def test_build_upload_identity_uses_workspace_name_and_slug(self) -> None:
        identity = build_upload_identity(Path("/tmp/matmul case"))
        self.assertEqual(identity.workspace_name, "matmul case")
        self.assertEqual(identity.workspace_slug, "matmul_case")
        self.assertRegex(identity.upload_uid, r"^[0-9a-f]{32}$")
        self.assertRegex(identity.upload_timestamp, r"^\d{8}T\d{6}Z$")


class OptimizeUploadCollectorTests(unittest.TestCase):
    def _create_test_workspace(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        # Root files
        (tmp / "kernel.py").write_text("", encoding="utf-8")
        (tmp / "opt_kernel.py").write_text("", encoding="utf-8")
        (tmp / "test_kernel.py").write_text("", encoding="utf-8")
        (tmp / "differential_test_kernel.py").write_text("", encoding="utf-8")
        (tmp / "bench_kernel.py").write_text("", encoding="utf-8")
        (tmp / "opt-note.md").write_text("", encoding="utf-8")
        (tmp / "learned_lessons.md").write_text("", encoding="utf-8")
        # Baseline
        (tmp / "baseline").mkdir()
        baseline_state = '{"baseline_operator": "baseline/baseline_kernel.py", "perf_artifact": "baseline/kernel_perf.txt"}'
        (tmp / "baseline" / "state.json").write_text(baseline_state, encoding="utf-8")
        (tmp / "baseline" / "baseline_kernel.py").write_text("", encoding="utf-8")
        (tmp / "baseline" / "kernel_perf.txt").write_text("", encoding="utf-8")
        # Round
        (tmp / "opt-round-1").mkdir()
        (tmp / "opt-round-1" / "summary.md").write_text("", encoding="utf-8")
        (tmp / "opt-round-1" / "attempts.md").write_text("", encoding="utf-8")
        (tmp / "opt-round-1" / "round-state.json").write_text("{}", encoding="utf-8")
        (tmp / "opt-round-1" / "opt_kernel.py").write_text("", encoding="utf-8")
        (tmp / "opt-round-1" / "opt_kernel_perf.txt").write_text("", encoding="utf-8")
        (tmp / "opt-round-1" / "perf-analysis.md").write_text("", encoding="utf-8")
        (tmp / "opt-round-1" / "compiler-analysis.md").write_text("", encoding="utf-8")
        # triton-agent-logs
        (tmp / "triton-agent-logs").mkdir()
        (tmp / "triton-agent-logs" / "run-001").mkdir()
        (tmp / "triton-agent-logs" / "run-001" / "show-output.log").write_text("", encoding="utf-8")
        (tmp / "triton-agent-logs" / "run-001" / "agent-session-batch-1-5.json").write_text(
            '{"session_id":"abc"}\n',
            encoding="utf-8",
        )
        # Excluded paths
        (tmp / "opt-round-1" / "ir").mkdir()
        (tmp / "opt-round-1" / "ir" / "dummy.txt").write_text("", encoding="utf-8")
        (tmp / "opt-verify").mkdir()
        (tmp / "opt-verify" / "verify-1").mkdir()
        (tmp / "opt-verify" / "verify-1" / "verify-state.json").write_text("", encoding="utf-8")
        (tmp / "foo_result.pt").write_text("", encoding="utf-8")
        (tmp / "PROF_123").mkdir()
        (tmp / "PROF_123" / "data.txt").write_text("", encoding="utf-8")
        (tmp / "baseline" / "archive.tar.gz").write_text("", encoding="utf-8")
        return tmp

    def test_collect_includes_whitelist_files(self) -> None:
        workspace = self._create_test_workspace()
        collected = collect_workspace_upload_files(workspace)
        included_names = [p.name for p in collected.included_files]
        self.assertIn("kernel.py", included_names)
        self.assertIn("opt_kernel.py", included_names)  # root-level
        self.assertIn("test_kernel.py", included_names)
        self.assertIn("differential_test_kernel.py", included_names)
        self.assertIn("bench_kernel.py", included_names)
        self.assertIn("opt-note.md", included_names)
        self.assertIn("learned_lessons.md", included_names)
        self.assertIn("state.json", included_names)  # baseline
        self.assertIn("opt_kernel_perf.txt", included_names)  # round perf
        self.assertIn("show-output.log", included_names)

    def test_collect_excludes_forbidden_paths(self) -> None:
        workspace = self._create_test_workspace()
        collected = collect_workspace_upload_files(workspace)
        included_names = [str(p.relative_to(workspace)) for p in collected.included_files]
        # Forbidden paths must not be included
        for name in included_names:
            self.assertNotIn("ir/", name)
            self.assertNotIn("opt-verify/", name)
            self.assertNotIn(".pt", name)
            self.assertNotIn("PROF_", name)
            self.assertNotIn("archive.tar.gz", name)
            self.assertNotIn("agent-session-batch-1-5.json", name)

    def test_collect_validates_workspace_exists(self) -> None:
        with self.assertRaises(ValueError):
            collect_workspace_upload_files(Path("/nonexistent/path"))

    def test_collect_validates_workspace_is_directory(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        f = tmp / "file.txt"
        f.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            collect_workspace_upload_files(f)

    def test_collect_rejects_non_workspace(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        # Empty directory without baseline or opt-round-*
        with self.assertRaises(ValueError):
            collect_workspace_upload_files(tmp)


class OptimizeUploadPackagerTests(unittest.TestCase):
    def test_tarball_contains_expected_members(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        (workspace / "kernel.py").write_text("code", encoding="utf-8")
        (workspace / "baseline").mkdir()
        (workspace / "baseline" / "state.json").write_text("{}", encoding="utf-8")
        (workspace / "opt-round-1").mkdir()
        (workspace / "opt-round-1" / "summary.md").write_text("summary", encoding="utf-8")
        # Build identity and collect
        from triton_agent.optimize_upload.naming import build_upload_identity
        from triton_agent.optimize_upload.collector import collect_workspace_upload_files
        identity = build_upload_identity(workspace)
        collected = collect_workspace_upload_files(workspace)
        manifest = build_manifest(identity, collected)

        from triton_agent.optimize_upload.packager import build_upload_tarball
        with build_upload_tarball(collected, manifest) as tarball_path:
            self.assertTrue(tarball_path.exists())
            self.assertTrue(str(tarball_path).endswith(".tar.gz"))
            with tarfile.open(tarball_path, "r:gz") as tf:
                names = tf.getnames()
                self.assertIn("baseline/state.json", names)
                self.assertIn("opt-round-1/summary.md", names)
                self.assertIn("_upload/manifest.json", names)
                self.assertIn("kernel.py", names)
                # No top-level wrapper directory with all entries under it
                root_dirs = {n.split("/")[0] for n in names}
                self.assertNotEqual(root_dirs, set())

    def test_tarball_no_extra_top_level_wrapper(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        (workspace / "kernel.py").write_text("code", encoding="utf-8")
        (workspace / "baseline").mkdir()
        (workspace / "baseline" / "state.json").write_text("{}", encoding="utf-8")
        from triton_agent.optimize_upload.naming import build_upload_identity
        from triton_agent.optimize_upload.collector import collect_workspace_upload_files
        identity = build_upload_identity(workspace)
        collected = collect_workspace_upload_files(workspace)
        manifest = build_manifest(identity, collected)

        from triton_agent.optimize_upload.packager import build_upload_tarball
        with build_upload_tarball(collected, manifest) as tarball_path:
            with tarfile.open(tarball_path, "r:gz") as tf:
                names = tf.getnames()
                # All paths should be relative, not wrapped in a dir
                for name in names:
                    self.assertFalse(name.startswith("workspace/"))
                    self.assertFalse(name.startswith("./"))


class OptimizeUploadClientTests(unittest.TestCase):
    def test_load_upload_url_raises_when_unset(self) -> None:
        from triton_agent.optimize_upload.client import load_upload_url
        with self.assertRaises(ValueError):
            with patch.dict(os.environ, {}, clear=True):
                load_upload_url()

    def test_load_upload_url_returns_from_env(self) -> None:
        from triton_agent.optimize_upload.client import load_upload_url
        url = "http://example.com:8080/uploads"
        with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_UPLOAD_URL": url}):
            self.assertEqual(load_upload_url(), url)

    def test_upload_tarball_sends_correct_headers(self) -> None:
        """Verify request headers using a local test server."""
        import http.server
        import threading
        from triton_agent.optimize_upload.client import upload_tarball

        captured: dict = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                captured["method"] = self.command
                captured["path"] = self.path
                captured["headers"] = dict(self.headers)
                captured["body"] = body
                resp_body = (
                    b'{"ok": true, "upload_uid": "test123", '
                    b'"stored_path": "/store/test.tar.gz"}'
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.daemon = True
        t.start()

        identity = UploadIdentity(
            upload_uid="6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            upload_timestamp="20260526T141530Z",
            workspace_name="test_ws",
            workspace_slug="test_ws",
        )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(b"fake-tar-gz")
            tarball = Path(f.name)

        try:
            response = upload_tarball(identity, tarball, f"http://127.0.0.1:{port}/uploads")
            self.assertTrue(response.ok)
            self.assertEqual(response.upload_uid, "test123")
            self.assertEqual(
                response.stored_path, "/store/test.tar.gz"
            )
            self.assertEqual(captured["method"], "POST")
            self.assertEqual(captured["headers"].get("Content-Type"), "application/gzip")
            self.assertEqual(
                captured["headers"].get("X-Triton-Agent-Upload-Uid"),
                "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            )
        finally:
            tarball.unlink(missing_ok=True)

    def test_upload_tarball_fails_on_http_error(self) -> None:
        """Test HTTP error surfaces as exception."""
        import http.server
        import threading
        from triton_agent.optimize_upload.client import upload_tarball

        class ErrorHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                resp_body = b'{"error": "bad request"}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        server = http.server.HTTPServer(("127.0.0.1", 0), ErrorHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.daemon = True
        t.start()

        identity = UploadIdentity(
            upload_uid="6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            upload_timestamp="20260526T141530Z",
            workspace_name="test_ws",
            workspace_slug="test_ws",
        )

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(b"fake-tar-gz")
            tarball = Path(f.name)

        try:
            with self.assertRaises(RuntimeError):
                upload_tarball(identity, tarball, f"http://127.0.0.1:{port}/uploads")
        finally:
            tarball.unlink(missing_ok=True)

    def test_upload_tarball_rejects_ok_false(self) -> None:
        """Business failure: server returns ok=false -> RuntimeError."""
        import http.server
        import threading
        from triton_agent.optimize_upload.client import upload_tarball

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                resp_body = b'{"ok": false, "stored_path": "/store/t.tar.gz"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.daemon = True
        t.start()

        identity = self._make_test_identity()
        tarball = self._make_test_tarball()
        try:
            with self.assertRaises(RuntimeError):
                upload_tarball(identity, tarball, f"http://127.0.0.1:{port}/uploads")
        finally:
            tarball.unlink(missing_ok=True)

    def test_upload_tarball_rejects_ok_string(self) -> None:
        """Malformed response: ok=\"false\" string -> RuntimeError (not treated as truthy)."""
        import http.server
        import threading
        from triton_agent.optimize_upload.client import upload_tarball

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                resp_body = b'{"ok": "false", "stored_path": "/store/t.tar.gz"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.daemon = True
        t.start()

        identity = self._make_test_identity()
        tarball = self._make_test_tarball()
        try:
            with self.assertRaises(RuntimeError):
                upload_tarball(identity, tarball, f"http://127.0.0.1:{port}/uploads")
        finally:
            tarball.unlink(missing_ok=True)

    def test_upload_tarball_rejects_missing_stored_path(self) -> None:
        """Malformed response: stored_path missing -> RuntimeError."""
        import http.server
        import threading
        from triton_agent.optimize_upload.client import upload_tarball

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                resp_body = b'{"ok": true, "upload_uid": "abc"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.daemon = True
        t.start()

        identity = self._make_test_identity()
        tarball = self._make_test_tarball()
        try:
            with self.assertRaises(RuntimeError):
                upload_tarball(identity, tarball, f"http://127.0.0.1:{port}/uploads")
        finally:
            tarball.unlink(missing_ok=True)

    @staticmethod
    def _make_test_identity() -> UploadIdentity:
        return UploadIdentity(
            upload_uid="6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            upload_timestamp="20260526T141530Z",
            workspace_name="test_ws",
            workspace_slug="test_ws",
        )

    @staticmethod
    def _make_test_tarball() -> Path:
        f = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        f.write(b"fake-tar-gz")
        f.close()
        return Path(f.name)


class OptimizeUploadWorkflowTests(unittest.TestCase):
    def test_workflow_missing_url_raises(self) -> None:
        workspace = Path(tempfile.mkdtemp())
        (workspace / "kernel.py").write_text("code", encoding="utf-8")
        (workspace / "baseline").mkdir()
        (workspace / "baseline" / "state.json").write_text("{}", encoding="utf-8")
        from triton_agent.optimize_upload.workflow import upload_optimize_workspace
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                upload_optimize_workspace(workspace, verbose=True)

    def test_workflow_invalid_workspace_raises(self) -> None:
        from triton_agent.optimize_upload.workflow import upload_optimize_workspace
        with self.assertRaises(ValueError):
            upload_optimize_workspace(Path("/nonexistent"), verbose=True)

    def test_workflow_successful_upload(self) -> None:
        """Integration test: full workflow hits a real HTTP server."""
        import http.server
        import threading
        from triton_agent.optimize_upload.workflow import upload_optimize_workspace

        received_data: dict = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                received_data["headers"] = dict(self.headers)
                received_data["body"] = body
                resp_body = (
                    b'{"ok": true, "upload_uid": "abc123", "upload_timestamp": "20260526T141530Z", '
                    b'"workspace_name": "test_ws", "workspace_slug": "test_ws", '
                    b'"stored_path": "/store/test.tar.gz"}'
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request)
        t.daemon = True
        t.start()

        workspace = Path(tempfile.mkdtemp())
        (workspace / "kernel.py").write_text("code", encoding="utf-8")
        (workspace / "baseline").mkdir()
        (workspace / "baseline" / "state.json").write_text("{}", encoding="utf-8")

        upload_url = f"http://127.0.0.1:{port}/uploads"
        with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_UPLOAD_URL": upload_url}):
            response = upload_optimize_workspace(workspace, verbose=False)
            self.assertTrue(response.ok)
            self.assertEqual(response.upload_uid, "abc123")
