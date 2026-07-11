from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UploadReceipt:
    upload_uid: str
    upload_timestamp: str
    workspace_name: str
    workspace_slug: str
    received_at: str
    stored_path: str
    content_length: int
    content_type: str = "application/gzip"
    manifest_version: str = ""


@dataclass
class UploadResponse:
    ok: bool = True
    upload_uid: str = ""
    upload_timestamp: str = ""
    workspace_name: str = ""
    workspace_slug: str = ""
    stored_path: str = ""
    message: str = ""


@dataclass
class ErrorResponse:
    ok: bool = False
    error: str = ""
    detail: str = ""
