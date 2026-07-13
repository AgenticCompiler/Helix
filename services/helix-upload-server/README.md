# helix-upload-server

Standalone upload server for Triton NPU optimize workspace analysis archives.

## Quick Start

```bash
# Install dependencies
cd services/helix-upload-server
uv sync

# Start the server
uv run helix-upload-server \
  --storage-root /data/helix/uploads \
  --temp-root /data/helix/uploads/.tmp
```

## Endpoints

- `GET /healthz` — health check
- `POST /uploads` — receive optimize workspace archive

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | 0.0.0.0 | Bind address |
| `--port` | 8080 | Bind port |
| `--storage-root` | (required) | Directory for uploaded archives |
| `--temp-root` | (required) | Directory for temporary upload files |
| `--max-upload-bytes` | 536870912 (512 MB) | Maximum upload size |
| `--log-level` | info | Log level |

## Storage Format

Uploads are stored as:
```
<timestamp>-<workspace_slug>-<uid>.tar.gz
<timestamp>-<workspace_slug>-<uid>.receipt.json
```
