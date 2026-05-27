from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    storage_root: Path = Path("/data/triton-agent/uploads")
    temp_root: Path = Path("/data/triton-agent/uploads/.tmp")
    max_upload_bytes: int = 536870912  # 512 MB
    log_level: str = "info"


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(prog="triton-agent-upload-server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-upload-bytes", type=int, default=536870912)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    return AppConfig(
        host=args.host,
        port=args.port,
        storage_root=args.storage_root,
        temp_root=args.temp_root,
        max_upload_bytes=args.max_upload_bytes,
        log_level=args.log_level,
    )
