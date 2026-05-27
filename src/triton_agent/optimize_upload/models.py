from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class UploadIdentity:
    upload_uid: str
    upload_timestamp: str
    workspace_name: str
    workspace_slug: str


@dataclass(frozen=True)
class CollectedUpload:
    workspace: Path
    operator_file: Optional[Path]
    included_files: tuple[Path, ...]
    excluded_entries: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class UploadResponse:
    ok: bool
    upload_uid: str
    upload_timestamp: str
    workspace_name: str
    workspace_slug: str
    stored_path: str
    raw_body: str = ""
