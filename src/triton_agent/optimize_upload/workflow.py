from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from triton_agent.optimize_upload.client import load_upload_url, upload_tarball
from triton_agent.optimize_upload.collector import collect_workspace_upload_files
from triton_agent.optimize_upload.manifest import build_manifest
from triton_agent.optimize_upload.models import UploadResponse
from triton_agent.optimize_upload.naming import build_upload_identity
from triton_agent.optimize_upload.packager import build_upload_tarball


def upload_optimize_workspace(
    workspace: Path,
    *,
    url: Optional[str] = None,
    verbose: bool = False,
) -> UploadResponse:
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError(f"Workspace path does not exist or is not a directory: {workspace}")

    identity = build_upload_identity(workspace)
    collected = collect_workspace_upload_files(workspace)
    manifest = build_manifest(identity, collected)

    upload_url = url if url else load_upload_url()

    if verbose:
        print(
            f"Packaging {len(collected.included_files)} files "
            f"({sum(p.stat().st_size for p in collected.included_files if p.exists())} bytes)...",
            file=sys.stderr,
        )

    with build_upload_tarball(collected, manifest) as tarball_path:
        if verbose:
            tarball_size = tarball_path.stat().st_size
            print(f"Uploading {tarball_size} bytes to {upload_url}...", file=sys.stderr)
        response = upload_tarball(identity, tarball_path, upload_url)

    return response
