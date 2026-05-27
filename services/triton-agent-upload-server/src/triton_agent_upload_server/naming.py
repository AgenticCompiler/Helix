from __future__ import annotations

import re


def slugify_workspace_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    slug = re.sub(r"_+", "_", slug)
    slug = slug.strip("_")
    return slug if slug else "workspace"


def build_archive_name(timestamp: str, slug: str, uid: str) -> str:
    return f"{timestamp}-{slug}-{uid}.tar.gz"


def build_receipt_name(timestamp: str, slug: str, uid: str) -> str:
    return f"{timestamp}-{slug}-{uid}.receipt.json"


def validate_upload_headers(headers: dict[str, str]) -> dict[str, str]:
    normalized = {k.lower(): v for k, v in headers.items()}

    required = {
        "x-triton-agent-upload-uid": "upload_uid",
        "x-triton-agent-upload-timestamp": "upload_timestamp",
        "x-triton-agent-workspace-name": "workspace_name",
        "x-triton-agent-workspace-slug": "workspace_slug",
        "x-triton-agent-manifest-version": "manifest_version",
    }
    result: dict[str, str] = {}
    for header_key, field_name in required.items():
        value = normalized.get(header_key, "")
        if not value:
            raise ValueError(f"Missing required header: {header_key}")
        result[field_name] = value

    uid = result["upload_uid"]
    if not re.match(r"^[0-9a-f]{32}$", uid):
        raise ValueError(f"Invalid upload_uid format: {uid}")

    return result
