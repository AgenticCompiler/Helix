from __future__ import annotations

import argparse

from triton_agent.run_eval_mcp_server import serve_http_server_forever


def handle_run_eval_mcp_server(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    del parser
    return serve_http_server_forever(port=int(args.port))

