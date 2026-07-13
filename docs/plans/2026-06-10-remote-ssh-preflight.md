# Remote SSH Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit-`--remote` SSH key preflight that fails early with an `ssh-copy-id` hint when public-key access is not ready, without changing environment-only remote fallback behavior.

**Architecture:** Add one small `src/helix/remote_ssh_preflight.py` helper that reuses the existing `user@host[:port]` parser from the staged run-eval runtime, builds a non-interactive SSH probe command, and classifies authentication failures separately from transport failures. Wire the helper into `src/helix/cli.py` before remote env injection and handler dispatch, then lock the behavior with new helper tests, CLI integration tests, and a short README note.

**Tech Stack:** Python 3, `subprocess`, `argparse`, `unittest`, README docs

---

## File Structure

- Create: `src/helix/remote_ssh_preflight.py`
  Responsibility: Parse the existing remote target syntax through the staged runtime parser, build the non-interactive SSH probe command, and raise short actionable failures.
- Create: `tests/test_remote_ssh_preflight.py`
  Responsibility: Lock helper behavior for command construction, auth-failure mapping, and non-auth failure passthrough.
- Modify: `src/helix/cli.py:640-653`
  Responsibility: Run the explicit `--remote` preflight before remote env injection and handler dispatch, and convert helper failures into a clean exit code `1`.
- Modify: `tests/test_cli.py:2589-2668,3791-4047`
  Responsibility: Lock top-level CLI behavior for explicit remote preflight, preflight failure short-circuiting, and local-command skip behavior.
- Modify: `README.md:736-742`
  Responsibility: Document that explicit `--remote` now checks key-based SSH access first and suggests `ssh-copy-id` when the target still needs password-based setup.

### Task 1: Add failing helper tests for the SSH preflight contract

**Files:**
- Create: `tests/test_remote_ssh_preflight.py`
- Test: `tests/test_remote_ssh_preflight.py`

- [ ] **Step 1: Write the failing helper tests**

```python
import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import helix.remote_ssh_preflight as module


class RemoteSshPreflightTests(unittest.TestCase):
    def test_build_remote_ssh_preflight_command_without_port(self) -> None:
        command = module.build_remote_ssh_preflight_command("alice@example.com")

        self.assertEqual(
            command,
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "PreferredAuthentications=publickey",
                "-o",
                "NumberOfPasswordPrompts=0",
                "-o",
                "ConnectTimeout=5",
                "alice@example.com",
                "true",
            ],
        )

    def test_build_remote_ssh_preflight_command_with_port(self) -> None:
        command = module.build_remote_ssh_preflight_command("alice@example.com:2200")

        self.assertIn("-p", command)
        self.assertEqual(command[-2:], ["alice@example.com", "true"])

    def test_ensure_remote_ssh_ready_auth_failure_suggests_ssh_copy_id(self) -> None:
        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="Permission denied (publickey,password).",
        )

        with patch.object(module.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, r"ssh-copy-id -p 2200 alice@example.com"):
                module.ensure_remote_ssh_ready("alice@example.com:2200")

    def test_ensure_remote_ssh_ready_preserves_non_auth_failure(self) -> None:
        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="ssh: Could not resolve hostname missing.example.com: Name or service not known",
        )

        with patch.object(module.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, r"Could not resolve hostname missing\\.example\\.com"):
                module.ensure_remote_ssh_ready("alice@missing.example.com")
```

- [ ] **Step 2: Run the helper tests to verify they fail because the helper module does not exist yet**

Run: `uv run python -m unittest tests.test_remote_ssh_preflight -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'helix.remote_ssh_preflight'`.

- [ ] **Step 3: Commit the failing helper tests**

```bash
git add tests/test_remote_ssh_preflight.py
git commit -m "test: add remote ssh preflight helper coverage"
```

### Task 2: Implement the shared SSH preflight helper

**Files:**
- Create: `src/helix/remote_ssh_preflight.py`
- Test: `tests/test_remote_ssh_preflight.py`

- [ ] **Step 1: Write the minimal helper implementation**

```python
from __future__ import annotations

import subprocess
from typing import Protocol, TypedDict, cast

from helix.skill_loader import load_operator_eval_script_module


class RemoteSpec(TypedDict):
    user_host: str
    port: int | None


class _RunRuntimeModule(Protocol):
    def parse_remote_spec(self, raw: str) -> RemoteSpec: ...


_AUTH_FAILURE_MARKERS = (
    "permission denied",
    "publickey",
    "too many authentication failures",
    "no supported authentication methods available",
)
_CONNECT_TIMEOUT_SECONDS = 5
_PRECHECK_TIMEOUT_SECONDS = 10


def _parse_remote_spec(remote: str) -> RemoteSpec:
    module = cast(_RunRuntimeModule, load_operator_eval_script_module("run_runtime"))
    return module.parse_remote_spec(remote)


def build_remote_ssh_preflight_command(remote: str) -> list[str]:
    spec = _parse_remote_spec(remote)
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        f"ConnectTimeout={_CONNECT_TIMEOUT_SECONDS}",
    ]
    if spec["port"] is not None:
        command.extend(["-p", str(spec["port"])])
    command.extend([spec["user_host"], "true"])
    return command


def format_ssh_copy_id_command(remote: str) -> str:
    spec = _parse_remote_spec(remote)
    if spec["port"] is None:
        return f"ssh-copy-id {spec['user_host']}"
    return f"ssh-copy-id -p {spec['port']} {spec['user_host']}"


def ensure_remote_ssh_ready(remote: str) -> None:
    command = build_remote_ssh_preflight_command(remote)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=_PRECHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH preflight timed out while connecting to {remote}.") from exc

    if result.returncode == 0:
        return

    detail = (result.stderr or result.stdout).strip() or f"SSH preflight failed for {remote}."
    if _is_auth_failure(detail):
        raise RuntimeError(
            f"Remote target {remote} is not ready for key-based SSH access.\n"
            f"Run `{format_ssh_copy_id_command(remote)}` and enter the remote login password to copy your public key."
        )
    raise RuntimeError(detail)


def _is_auth_failure(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _AUTH_FAILURE_MARKERS)
```

- [ ] **Step 2: Run the helper tests to verify they pass**

Run: `uv run python -m unittest tests.test_remote_ssh_preflight -v`

Expected: PASS with 4 passing tests.

- [ ] **Step 3: Commit the helper implementation**

```bash
git add src/helix/remote_ssh_preflight.py tests/test_remote_ssh_preflight.py
git commit -m "feat: add remote ssh preflight helper"
```

### Task 3: Add failing CLI tests for explicit remote preflight behavior

**Files:**
- Modify: `tests/test_cli.py:2589-2668,3791-4047`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI integration tests around `main()`**

```python
    def test_main_explicit_remote_runs_ssh_preflight_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("helix.cli.ensure_remote_ssh_ready") as preflight, patch(
                "helix.commands.execution.run_remote_test",
                return_value=(fake_result, None, "/tmp/helix-abc"),
            ) as mocked:
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                        "--remote",
                        "alice@example.com:2200",
                    ]
                )

        self.assertEqual(exit_code, 0)
        preflight.assert_called_once_with("alice@example.com:2200")
        mocked.assert_called_once()

    def test_main_explicit_remote_preflight_failure_returns_1_and_skips_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            stderr = StringIO()
            with patch(
                "helix.cli.ensure_remote_ssh_ready",
                side_effect=RuntimeError("Run `ssh-copy-id alice@example.com` and enter the remote login password."),
            ), patch("helix.commands.execution.run_remote_test") as mocked:
                with redirect_stderr(stderr):
                    exit_code = main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--remote",
                            "alice@example.com",
                        ]
                    )

        self.assertEqual(exit_code, 1)
        mocked.assert_not_called()
        self.assertIn("ssh-copy-id alice@example.com", stderr.getvalue())

    def test_main_local_run_test_skips_ssh_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "test_kernel.py"
            operator.write_text("print('x')", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\nprint('test')\n", encoding="utf-8")

            fake_result = AgentResult(return_code=0, stdout="", stderr="")
            with patch("helix.cli.ensure_remote_ssh_ready") as preflight, patch(
                "helix.commands.execution.run_local_test",
                return_value=(fake_result, None),
            ) as mocked:
                exit_code = main(
                    [
                        "run-test",
                        "--test-file",
                        str(test_file),
                        "--operator-file",
                        str(operator),
                    ]
                )

        self.assertEqual(exit_code, 0)
        preflight.assert_not_called()
        mocked.assert_called_once()

```

- [ ] **Step 2: Run the targeted CLI tests to verify they fail before the CLI is wired**

Run: `uv run python -m unittest tests.test_cli.PathResolutionTests.test_main_explicit_remote_runs_ssh_preflight_before_dispatch tests.test_cli.PathResolutionTests.test_main_explicit_remote_preflight_failure_returns_1_and_skips_handler tests.test_cli.PathResolutionTests.test_main_local_run_test_skips_ssh_preflight -v`

Expected: FAIL because `main()` does not call `ensure_remote_ssh_ready(...)` yet.

- [ ] **Step 3: Commit the failing CLI tests**

```bash
git add tests/test_cli.py
git commit -m "test: lock explicit remote ssh preflight behavior"
```

### Task 4: Wire the CLI preflight, update docs, and verify the full change

**Files:**
- Modify: `src/helix/cli.py:1-30,640-653`
- Modify: `README.md:736-742`
- Test: `tests/test_remote_ssh_preflight.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Update `src/helix/cli.py` to run the explicit remote preflight before env injection**

```python
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from helix.remote_execution_env import (
    apply_remote_execution_env,
    remote_target_env_name,
    remote_workdir_env_name,
)
from helix.remote_ssh_preflight import ensure_remote_ssh_ready


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))
    try:
        _ensure_explicit_remote_ssh_ready_from_args(args)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _apply_remote_execution_env_from_args(args)
    command_kind = args.command_kind
    return _COMMAND_SPECS[command_kind].handler(parser, args)


def _ensure_explicit_remote_ssh_ready_from_args(args: argparse.Namespace) -> None:
    remote = getattr(args, "remote", None) if hasattr(args, "remote") else None
    if not remote:
        return
    ensure_remote_ssh_ready(remote)
```

- [ ] **Step 2: Update the shared README remote option note**

```markdown
- `--remote`: run execution and comparison commands through SSH, and pass remote context to generation and optimize workflows. When passed explicitly, the CLI first checks non-interactive key-based SSH access and suggests `ssh-copy-id` if the target still needs password-based setup.
```

- [ ] **Step 3: Re-run the targeted tests to verify the helper and CLI behavior pass**

Run: `uv run python -m unittest tests.test_remote_ssh_preflight tests.test_cli.PathResolutionTests.test_main_explicit_remote_runs_ssh_preflight_before_dispatch tests.test_cli.PathResolutionTests.test_main_explicit_remote_preflight_failure_returns_1_and_skips_handler tests.test_cli.PathResolutionTests.test_main_local_run_test_skips_ssh_preflight -v`

Expected: PASS with all helper and CLI preflight tests green.

- [ ] **Step 4: Run repository verification**

Run: `uv run --group dev ruff check`
Expected: PASS with no lint errors.

Run: `uv run pyright`
Expected: PASS with no type errors.

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
Expected: PASS with the full test suite green.

- [ ] **Step 5: Commit the integrated change**

```bash
git add src/helix/remote_ssh_preflight.py src/helix/cli.py tests/test_remote_ssh_preflight.py tests/test_cli.py README.md
git commit -m "feat: add remote ssh preflight"
```
