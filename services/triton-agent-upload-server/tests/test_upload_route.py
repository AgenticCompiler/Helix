import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from triton_agent_upload_server.app import create_app


class TestUploadRoute:
    @staticmethod
    def _make_headers(overrides=None):
        headers = {
            "Content-Type": "application/gzip",
            "X-Triton-Agent-Upload-Uid": "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
            "X-Triton-Agent-Upload-Timestamp": "20260526T141530Z",
            "X-Triton-Agent-Workspace-Name": "test_workspace",
            "X-Triton-Agent-Workspace-Slug": "test_workspace",
            "X-Triton-Agent-Manifest-Version": "1",
        }
        if overrides:
            headers.update(overrides)
        return headers

    @staticmethod
    def _upload(client, content=b"fake-tar", headers_overrides=None):
        headers = TestUploadRoute._make_headers(headers_overrides or {})
        if headers.get("Content-Length") is None:
            headers["Content-Length"] = str(len(content))
        return client.post("/uploads", content=content, headers=headers)

    @staticmethod
    def _fresh_app(tmp_path):
        return create_app(
            storage_root=Path(tmp_path) / "store",
            temp_root=Path(tmp_path) / "tmp",
            min_upload_bytes=0,
        )

    def test_healthz(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
            client = TestClient(app)
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_valid_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)
            data = b"fake-tar-gz-content"
            resp = client.post(
                "/uploads",
                content=data,
                headers=self._make_headers({"Content-Length": str(len(data))}),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["ok"] is True
            assert body["upload_uid"] == "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12"

    def test_missing_headers_returns_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)
            resp = client.post("/uploads", content=b"data", headers={"Content-Type": "application/gzip"})
            assert resp.status_code == 400

    def test_invalid_content_length_returns_411(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)
            resp = client.post(
                "/uploads",
                content=b"data",
                headers=self._make_headers({"Content-Length": "not_an_int"}),
            )
            assert resp.status_code == 411

    def test_wrong_content_type_returns_415(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)
            resp = client.post(
                "/uploads",
                content=b"data",
                headers=self._make_headers({"Content-Type": "application/json"}),
            )
            assert resp.status_code == 415

    # -- Dedup & rate limit tests --

    def test_same_slug_replaces_old_upload(self):
        """Same IP + same slug within dedup window replaces the old file."""
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)

            resp1 = self._upload(client)
            assert resp1.status_code == 200
            stored1 = resp1.json()["stored_path"]

            # Change UID so the archive name differs
            resp2 = self._upload(
                client,
                headers_overrides={
                    "X-Triton-Agent-Upload-Uid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "X-Triton-Agent-Upload-Timestamp": "20260526T141600Z",
                },
            )
            assert resp2.status_code == 200
            stored2 = resp2.json()["stored_path"]
            assert stored2 != stored1

            # Old archive should be deleted
            store_root = Path(tmp) / "store"
            assert not (store_root / stored1).exists()
            assert (store_root / stored2).exists()

    def test_different_slugs_dont_trigger_dedup(self):
        """Different slugs from same IP are stored independently."""
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)

            resp1 = self._upload(client)
            assert resp1.status_code == 200

            resp2 = self._upload(
                client,
                headers_overrides={
                    "X-Triton-Agent-Workspace-Slug": "other_workspace",
                    "X-Triton-Agent-Workspace-Name": "other_workspace",
                    "X-Triton-Agent-Upload-Uid": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                },
            )
            assert resp2.status_code == 200

            # Both should exist
            store_root = Path(tmp) / "store"
            assert len(list(store_root.glob("*.tar.gz"))) == 2

    def test_burst_rate_limit_rejects_after_threshold(self):
        """Same IP + ≥3 distinct slugs within 60s triggers 429 ban."""
        with tempfile.TemporaryDirectory() as tmp:
            app = self._fresh_app(tmp)
            client = TestClient(app)

            # 1st slug — accepted
            resp = self._upload(
                client,
                headers_overrides={
                    "X-Triton-Agent-Workspace-Slug": "slug_a",
                    "X-Triton-Agent-Workspace-Name": "slug_a",
                    "X-Triton-Agent-Upload-Uid": "a1" + "0" * 30,
                },
            )
            assert resp.status_code == 200

            # 2nd slug — accepted
            resp = self._upload(
                client,
                headers_overrides={
                    "X-Triton-Agent-Workspace-Slug": "slug_b",
                    "X-Triton-Agent-Workspace-Name": "slug_b",
                    "X-Triton-Agent-Upload-Uid": "b2" + "0" * 30,
                },
            )
            assert resp.status_code == 200

            # 3rd slug — triggers burst ban
            resp = self._upload(
                client,
                headers_overrides={
                    "X-Triton-Agent-Workspace-Slug": "slug_c",
                    "X-Triton-Agent-Workspace-Name": "slug_c",
                    "X-Triton-Agent-Upload-Uid": "c3" + "0" * 30,
                },
            )
            assert resp.status_code == 429
            assert "429" in str(resp.status_code)

            # 4th slug — still banned
            resp = self._upload(
                client,
                headers_overrides={
                    "X-Triton-Agent-Workspace-Slug": "slug_d",
                    "X-Triton-Agent-Workspace-Name": "slug_d",
                    "X-Triton-Agent-Upload-Uid": "d4" + "0" * 30,
                },
            )
            assert resp.status_code == 429

    def test_upload_smaller_than_min_bytes_returns_400(self):
        """Content-Length below min_upload_bytes is rejected."""
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                storage_root=Path(tmp) / "store",
                temp_root=Path(tmp) / "tmp",
                min_upload_bytes=100000,
            )
            client = TestClient(app)
            data = b"tiny"
            resp = self._upload(client, content=data)
            assert resp.status_code == 400
