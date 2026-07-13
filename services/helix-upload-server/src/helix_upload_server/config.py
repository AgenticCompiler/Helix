from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    storage_root: Path = Path("/data/helix/uploads")
    temp_root: Path = Path("/data/helix/uploads/.tmp")
    max_upload_bytes: int = 536870912  # 512 MB
    min_upload_bytes: int = 102400  # 100 KB
    log_level: str = "info"
    dedup_window_seconds: int = 900
    rate_limit_max_slugs: int = 3
    rate_limit_window_seconds: int = 30
    rate_limit_cooldown_seconds: int = 600


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(prog="helix-upload-server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-upload-bytes", type=int, default=536870912)
    parser.add_argument("--min-upload-bytes", type=int, default=102400)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--dedup-window-seconds", type=int, default=900)
    parser.add_argument("--rate-limit-max-slugs", type=int, default=3)
    parser.add_argument("--rate-limit-window-seconds", type=int, default=30)
    parser.add_argument("--rate-limit-cooldown-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    return AppConfig(
        host=args.host,
        port=args.port,
        storage_root=args.storage_root,
        temp_root=args.temp_root,
        max_upload_bytes=args.max_upload_bytes,
        min_upload_bytes=args.min_upload_bytes,
        log_level=args.log_level,
        dedup_window_seconds=args.dedup_window_seconds,
        rate_limit_max_slugs=args.rate_limit_max_slugs,
        rate_limit_window_seconds=args.rate_limit_window_seconds,
        rate_limit_cooldown_seconds=args.rate_limit_cooldown_seconds,
    )
