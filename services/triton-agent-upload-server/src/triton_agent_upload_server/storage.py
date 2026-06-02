from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator


def _receipt_name(archive_name: str) -> str:
    """Derive receipt filename from the archive name.

    ``foo.tar.gz`` → ``foo.receipt.json``
    """
    suffix = ".tar.gz"
    if archive_name.endswith(suffix):
        stem = archive_name[: -len(suffix)]
    else:
        stem = archive_name.rsplit(".", 1)[0]
    return stem + ".receipt.json"


class UploadStorage:
    def __init__(
        self,
        storage_root: Path,
        temp_root: Path,
    ) -> None:
        self._storage_root = storage_root
        self._temp_root = temp_root

    async def save_upload(
        self,
        archive_name: str,
        stream: AsyncIterator[bytes],
        receipt: dict[str, object],
        content_length: int,
    ) -> Path:
        archive_path = self._storage_root / archive_name
        receipt_path = self._storage_root / _receipt_name(archive_name)

        if archive_path.exists() or receipt_path.exists():
            raise FileExistsError(f"Upload target already exists: {archive_name}")

        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._temp_root.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            suffix=".partial",
            dir=str(self._temp_root),
        )
        total = 0
        try:
            with open(fd, "wb") as f:
                async for chunk in stream:
                    total += len(chunk)
                    f.write(chunk)
            if total != content_length:
                Path(tmp_path).unlink(missing_ok=True)
                raise ValueError(
                    f"Content-Length mismatch: expected {content_length} bytes, "
                    f"received {total} bytes"
                )
            shutil.move(tmp_path, str(archive_path))
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        receipt_payload: dict[str, object] = {
            **receipt,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "stored_path": str(archive_path),
        }
        receipt_path.write_text(
            json.dumps(receipt_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return archive_path
