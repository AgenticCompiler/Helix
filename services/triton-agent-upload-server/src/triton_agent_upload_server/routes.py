from __future__ import annotations

import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Request

from triton_agent_upload_server.auth import authorize_request
from triton_agent_upload_server.dedup import UploadGuard
from triton_agent_upload_server.models import UploadReceipt
from triton_agent_upload_server.naming import (
    build_archive_name,
    slugify_workspace_name,
    validate_upload_headers,
)
from triton_agent_upload_server.responses import error_response, success_response
from triton_agent_upload_server.storage import UploadStorage

logger = logging.getLogger(__name__)


def create_router(storage: UploadStorage, max_upload_bytes: int, guard: UploadGuard | None = None, min_upload_bytes: int = 102400) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": "0.1.0"}

    @router.post("/uploads")
    async def upload(request: Request):
        await authorize_request(request)

        content_type = request.headers.get("content-type", "")
        if content_type != "application/gzip":
            return error_response(415, "Unsupported Media Type", f"Expected application/gzip, got {content_type}")

        content_length_str = request.headers.get("content-length")
        if not content_length_str:
            return error_response(411, "Length Required", "Content-Length header is required")
        try:
            content_length = int(content_length_str)
        except ValueError:
            return error_response(411, "Length Required", "Content-Length must be an integer")
        if content_length > max_upload_bytes:
            return error_response(
                413, "Payload Too Large",
                f"Upload exceeds maximum size of {max_upload_bytes} bytes",
            )
        if content_length < min_upload_bytes:
            return error_response(
                400, "Bad Request",
                f"Upload smaller than minimum size of {min_upload_bytes} bytes",
            )

        try:
            header_data = validate_upload_headers(dict(request.headers))
        except ValueError as exc:
            return error_response(400, "Bad Request", str(exc))

        expected_slug = slugify_workspace_name(header_data["workspace_name"])
        if header_data["workspace_slug"] != expected_slug:
            return error_response(
                400, "Bad Request",
                f"workspace_slug '{header_data['workspace_slug']}' does not match "
                f"normalized form '{expected_slug}'",
            )

        archive_name = build_archive_name(
            header_data["upload_timestamp"],
            header_data["workspace_slug"],
            header_data["upload_uid"],
        )

        client_ip = _client_ip(request)
        workspace_slug = header_data["workspace_slug"]

        if guard is not None:
            result = guard.check(client_ip, workspace_slug)
            if result.rejected:
                logger.warning(
                    "Burst upload rejected: ip=%s slug=%s uid=%s",
                    client_ip,
                    workspace_slug,
                    header_data["upload_uid"],
                )
                return error_response(
                    429, "Too Many Requests",
                    "Too many distinct uploads from this IP; temporarily banned",
                )
            if result.replace and result.old_archive_name is not None:
                logger.info(
                    "Dedup replacing old upload: ip=%s slug=%s old=%s new=%s",
                    client_ip,
                    workspace_slug,
                    result.old_archive_name,
                    archive_name,
                )
                storage.delete_upload(result.old_archive_name)

        receipt = UploadReceipt(
            upload_uid=header_data["upload_uid"],
            upload_timestamp=header_data["upload_timestamp"],
            workspace_name=header_data["workspace_name"],
            workspace_slug=header_data["workspace_slug"],
            received_at=datetime.now(timezone.utc).isoformat(),
            stored_path="",
            content_length=content_length,
            content_type=content_type,
            manifest_version=header_data.get("manifest_version", ""),
        )

        try:
            archive_path = await storage.save_upload(
                archive_name=archive_name,
                stream=request.stream(),
                receipt=vars(receipt),
                content_length=content_length,
            )
        except FileExistsError:
            return error_response(409, "Conflict", f"Upload target already exists: {archive_name}")
        except ValueError as exc:
            return error_response(400, "Bad Request", str(exc))

        logger.info(
            "Upload accepted: uid=%s slug=%s size=%d path=%s",
            header_data["upload_uid"],
            header_data["workspace_slug"],
            content_length,
            archive_path,
        )

        if guard is not None:
            guard.record(client_ip, workspace_slug, archive_name)

        return success_response({
            "upload_uid": header_data["upload_uid"],
            "upload_timestamp": header_data["upload_timestamp"],
            "workspace_name": header_data["workspace_name"],
            "workspace_slug": header_data["workspace_slug"],
            "stored_path": archive_name,
        })

    return router


def _client_ip(request: Request) -> str:
    if request.client is not None:
        return request.client.host
    return "unknown"
