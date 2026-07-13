from __future__ import annotations

from helix.optimize_upload.models import CollectedUpload, UploadIdentity

_MANIFEST_VERSION = 1


def build_manifest(
    identity: UploadIdentity,
    collected: CollectedUpload,
) -> dict[str, object]:
    included_files = sorted(
        str(p.relative_to(collected.workspace))
        for p in collected.included_files
    )
    total_bytes = sum(
        p.stat().st_size
        for p in collected.included_files
        if p.exists()
    )
    return {
        "manifest_version": _MANIFEST_VERSION,
        "upload_uid": identity.upload_uid,
        "upload_timestamp": identity.upload_timestamp,
        "workspace_name": identity.workspace_name,
        "workspace_slug": identity.workspace_slug,
        "operator_file": str(collected.operator_file.relative_to(collected.workspace))
            if collected.operator_file else None,
        "included_files": included_files,
        "excluded_entries": [
            {"path": path, "reason": reason}
            for path, reason in collected.excluded_entries
        ],
        "file_count": len(included_files),
        "total_bytes": total_bytes,
    }
