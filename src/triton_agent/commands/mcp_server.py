from __future__ import annotations

import argparse

from triton_agent.eval.mcp_server import serve_http_server_forever


def handle_run_eval_mcp_server(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    del parser
    return serve_http_server_forever(
        port=int(args.port),
        npu_devices=getattr(args, "npu_devices", None),
        workers_per_npu=getattr(args, "workers_per_npu", None),
    )
