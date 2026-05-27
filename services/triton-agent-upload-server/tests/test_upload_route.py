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

    def test_healthz(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
            client = TestClient(app)
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_valid_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
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
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
            client = TestClient(app)
            resp = client.post("/uploads", content=b"data", headers={"Content-Type": "application/gzip"})
            assert resp.status_code == 400

    def test_invalid_content_length_returns_411(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
            client = TestClient(app)
            resp = client.post(
                "/uploads",
                content=b"data",
                headers=self._make_headers({"Content-Length": "not_an_int"}),
            )
            assert resp.status_code == 411

    def test_wrong_content_type_returns_415(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
            client = TestClient(app)
            resp = client.post(
                "/uploads",
                content=b"data",
                headers=self._make_headers({"Content-Type": "application/json"}),
            )
            assert resp.status_code == 415

    def test_duplicate_upload_returns_409(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
            client = TestClient(app)
            headers = self._make_headers({"Content-Length": "4"})
            resp1 = client.post("/uploads", content=b"data", headers=headers)
            assert resp1.status_code == 200
            resp2 = client.post("/uploads", content=b"data", headers=headers)
            assert resp2.status_code == 409
