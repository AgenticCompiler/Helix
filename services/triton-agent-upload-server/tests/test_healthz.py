import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from triton_agent_upload_server.app import create_app


def test_healthz_returns_ok():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(storage_root=Path(tmp) / "store", temp_root=Path(tmp) / "tmp")
        client = TestClient(app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
