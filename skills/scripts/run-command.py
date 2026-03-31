from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    # Resolve the real script location so this still works when `skills/` is reached
    # through a workspace symlink such as `.codex/skills` or `.opencode/skills`.
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "src"))

    from triton_agent.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
