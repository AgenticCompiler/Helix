import json
import tempfile
from pathlib import Path

import pytest
from helix_upload_server.storage import UploadStorage


async def _stream(data: bytes):
    yield data


class TestStorage:
    async def test_temp_write_then_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = UploadStorage(
                storage_root=Path(tmp) / "store",
                temp_root=Path(tmp) / "tmp",
            )
            archive_path = await storage.save_upload(
                archive_name="test.tar.gz",
                stream=_stream(b"hello world"),
                receipt={"test": True},
                content_length=11,
            )
            assert archive_path.exists()
            assert archive_path.name == "test.tar.gz"
            receipt_path = archive_path.parent / "test.receipt.json"
            assert receipt_path.exists()
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert receipt["test"] is True

    async def test_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = UploadStorage(
                storage_root=Path(tmp) / "store",
                temp_root=Path(tmp) / "tmp",
            )
            await storage.save_upload("test.tar.gz", _stream(b"data"), {}, 4)
            with pytest.raises(FileExistsError):
                await storage.save_upload("test.tar.gz", _stream(b"data2"), {}, 5)

    async def test_content_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = UploadStorage(
                storage_root=Path(tmp) / "store",
                temp_root=Path(tmp) / "tmp",
            )
            with pytest.raises(ValueError, match="Content-Length"):
                await storage.save_upload("test.tar.gz", _stream(b"abc"), {}, 999)
