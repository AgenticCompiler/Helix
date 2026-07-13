from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from helix.optimize_upload.models import CollectedUpload


@contextmanager
def build_upload_tarball(
    collected: CollectedUpload,
    manifest: dict[str, object],
) -> Iterator[Path]:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(tmp_fd)
    try:
        with tarfile.open(tmp_path, "w:gz") as tf:
            for file_path in collected.included_files:
                try:
                    arcname = str(file_path.relative_to(collected.workspace))
                except ValueError:
                    continue
                tf.add(str(file_path), arcname=arcname)
            manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(name="_upload/manifest.json")
            info.size = len(manifest_bytes)
            tf.addfile(info, io.BytesIO(manifest_bytes))
        yield Path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
