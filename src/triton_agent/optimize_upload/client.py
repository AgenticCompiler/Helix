from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from triton_agent.optimize_upload.models import UploadIdentity, UploadResponse


class UploadUrlMissingError(ValueError):
    """Raised when TRITON_AGENT_OPTIMIZE_UPLOAD_URL is not set."""

    def __init__(self) -> None:
        super().__init__(
            "TRITON_AGENT_OPTIMIZE_UPLOAD_URL is not set. "
            "Set this environment variable to the upload server endpoint."
        )


def load_upload_url() -> str:
    url = os.environ.get("TRITON_AGENT_OPTIMIZE_UPLOAD_URL")
    if not url:
        raise UploadUrlMissingError()
    return url


def upload_tarball(
    identity: UploadIdentity,
    tarball: Path,
    url: str,
) -> UploadResponse:
    data = tarball.read_bytes()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/gzip",
            "Content-Length": str(len(data)),
            "X-Triton-Agent-Upload-Uid": identity.upload_uid,
            "X-Triton-Agent-Upload-Timestamp": identity.upload_timestamp,
            "X-Triton-Agent-Workspace-Name": identity.workspace_name,
            "X-Triton-Agent-Workspace-Slug": identity.workspace_slug,
            "X-Triton-Agent-Manifest-Version": "1",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # nosec: socket may already be closed
            detail = "(unable to read response body)"
        raise RuntimeError(
            f"Upload failed (HTTP {exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Upload failed: {exc.reason}") from exc

    if payload.get("ok") is not True:
        raise RuntimeError(
            f"Upload rejected by server: ok=false. Response: {body}"
        )
    stored_path = payload.get("stored_path")
    if not stored_path:
        raise RuntimeError(
            f"Upload response missing stored_path. Response: {body}"
        )

    return UploadResponse(
        ok=True,
        upload_uid=payload.get("upload_uid", ""),
        upload_timestamp=payload.get("upload_timestamp", ""),
        workspace_name=payload.get("workspace_name", ""),
        workspace_slug=payload.get("workspace_slug", ""),
        stored_path=stored_path,
        raw_body=body,
    )
