from __future__ import annotations

import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from triton_agent_upload_server.config import parse_args
from triton_agent_upload_server.routes import create_router
from triton_agent_upload_server.storage import UploadStorage


def create_app(
    storage_root: Path | None = None,
    temp_root: Path | None = None,
    max_upload_bytes: int = 536870912,
) -> FastAPI:
    app = FastAPI(title="triton-agent-upload-server")

    _storage_root = storage_root or Path("/data/triton-agent/uploads")
    _temp_root = temp_root or Path("/data/triton-agent/uploads/.tmp")

    storage = UploadStorage(
        storage_root=_storage_root,
        temp_root=_temp_root,
    )
    router = create_router(storage=storage, max_upload_bytes=max_upload_bytes)
    app.include_router(router)

    return app


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(levelname)s\t%(name)s\t%(message)s",
    )

    app = create_app(
        storage_root=config.storage_root,
        temp_root=config.temp_root,
        max_upload_bytes=config.max_upload_bytes,
    )

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
