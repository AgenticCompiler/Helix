from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from triton_agent.optimize_upload.models import UploadIdentity


def slugify_workspace_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")
    return slug if slug else "workspace"


def build_upload_identity(workspace: Path) -> UploadIdentity:
    workspace_name = workspace.name
    workspace_slug = slugify_workspace_name(workspace_name)
    upload_uid = uuid.uuid4().hex
    upload_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return UploadIdentity(
        upload_uid=upload_uid,
        upload_timestamp=upload_timestamp,
        workspace_name=workspace_name,
        workspace_slug=workspace_slug,
    )
