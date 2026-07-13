import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import helix.remote.ssh_preflight as module


class RemoteSshPreflightTests(unittest.TestCase):
    def test_build_remote_ssh_preflight_command_without_port(self) -> None:
        with patch.object(module, "_parse_remote_spec", return_value={"user_host": "alice@example.com", "port": None}):
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

    def test_build_remote_ssh_preflight_command_accepts_ssh_alias(self) -> None:
        with patch.object(module, "_parse_remote_spec", return_value={"user_host": "R154_cdj", "port": None}):
            command = module.build_remote_ssh_preflight_command("R154_cdj")

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
                "R154_cdj",
                "true",
            ],
        )

    def test_build_remote_ssh_preflight_command_with_port(self) -> None:
        with patch.object(
            module,
            "_parse_remote_spec",
            return_value={"user_host": "alice@example.com", "port": 2200},
        ):
            command = module.build_remote_ssh_preflight_command("alice@example.com:2200")

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
                "-p",
                "2200",
                "alice@example.com",
                "true",
            ],
        )

    def test_format_ssh_copy_id_command_without_port(self) -> None:
        with patch.object(module, "_parse_remote_spec", return_value={"user_host": "alice@example.com", "port": None}):
            command = module.format_ssh_copy_id_command("alice@example.com")

        self.assertEqual(command, "ssh-copy-id alice@example.com")

    def test_ensure_remote_ssh_ready_auth_failure_suggests_ssh_copy_id(self) -> None:
        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="Permission denied (publickey,password).",
        )

        with patch.object(
            module,
            "build_remote_ssh_preflight_command",
            return_value=["ssh", "alice@example.com", "true"],
        ), patch.object(module, "format_ssh_copy_id_command", return_value="ssh-copy-id -p 2200 alice@example.com"), patch.object(
            module.subprocess,
            "run",
            return_value=result,
        ):
            with self.assertRaisesRegex(RuntimeError, r"ssh-copy-id -p 2200 alice@example.com"):
                module.ensure_remote_ssh_ready("alice@example.com:2200")

    def test_ensure_remote_ssh_ready_preserves_non_auth_failure(self) -> None:
        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout="",
            stderr="ssh: Could not resolve hostname missing.example.com: Name or service not known",
        )

        with patch.object(
            module,
            "build_remote_ssh_preflight_command",
            return_value=["ssh", "alice@example.com", "true"],
        ), patch.object(module.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, r"Could not resolve hostname missing\.example\.com"):
                module.ensure_remote_ssh_ready("alice@missing.example.com")

    def test_ensure_remote_ssh_ready_returns_when_probe_succeeds(self) -> None:
        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=0,
            stdout="",
            stderr="",
        )

        with patch.object(
            module,
            "build_remote_ssh_preflight_command",
            return_value=["ssh", "alice@example.com", "true"],
        ), patch.object(module.subprocess, "run", return_value=result):
            self.assertIsNone(module.ensure_remote_ssh_ready("alice@example.com"))

    def test_ensure_remote_ssh_ready_timeout_raises_runtime_error(self) -> None:
        with patch.object(
            module,
            "build_remote_ssh_preflight_command",
            return_value=["ssh", "alice@example.com", "true"],
        ), patch.object(
            module.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["ssh", "alice@example.com", "true"], timeout=10),
        ):
            with self.assertRaisesRegex(RuntimeError, r"SSH preflight timed out while connecting to alice@example.com"):
                module.ensure_remote_ssh_ready("alice@example.com")

    def test_ensure_remote_ssh_ready_decodes_utf8_failure_output_from_bytes(self) -> None:
        result = subprocess.CompletedProcess(
            args=["ssh"],
            returncode=255,
            stdout=b"",
            stderr="无法解析主机名".encode("utf-8"),
        )

        with patch.object(
            module,
            "build_remote_ssh_preflight_command",
            return_value=["ssh", "alice@example.com", "true"],
        ), patch.object(module.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "无法解析主机名"):
                module.ensure_remote_ssh_ready("alice@example.com")
