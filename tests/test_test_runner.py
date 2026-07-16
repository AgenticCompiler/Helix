import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

from helix.skills.loader import load_operator_eval_script_module

from tests.run_skill_test_utils import (
    load_compare_result_module,
    load_local_test_api_module,
    load_local_test_worker_module,
    load_remote_api_module,
    load_test_contract_module,
)

_WARNING_LINE = "[WARNING] Please DO NOT tune args ['num_warps']!\n"
_ANOTHER_WARNING_LINE = "[WARNING] autotune fallback was used\n"


class _FakeTensor:
    def __init__(self, value: str, device: str) -> None:
        self.value = value
        self.device = device

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return _FakeTensor(self.value, "cpu")


@contextmanager
def _without_preloaded_modules(*names: str):
    saved: dict[str, ModuleType] = {}
    missing: set[str] = set()
    for name in names:
        module = sys.modules.pop(name, None)
        if module is None:
            missing.add(name)
        else:
            saved[name] = module
    try:
        yield
    finally:
        for name in missing:
            sys.modules.pop(name, None)
        for name, module in saved.items():
            sys.modules[name] = module


class LocalTestRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._remote_runner = load_remote_api_module()
        self._batch_copy_patcher = patch.object(self._remote_runner, "copy_files_to_remote")
        self._batch_copy = self._batch_copy_patcher.start()

    def tearDown(self) -> None:
        self._batch_copy_patcher.stop()
        super().tearDown()

    def test_load_module_registers_temporary_module_in_sys_modules_during_exec(self) -> None:
        module = load_test_contract_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "temp_module.py"
            script.write_text(
                "import sys\n"
                "SEEN_DURING_EXEC = __name__ in sys.modules\n",
                encoding="utf-8",
            )

            loaded = module.load_module(script, "temp_runtime")

        self.assertTrue(loaded.SEEN_DURING_EXEC)
        self.assertNotIn(loaded.__name__, sys.modules)

    def test_load_differential_test_cases_bootstraps_torch_before_user_module_exec(self) -> None:
        module = load_test_contract_module()
        import_events: list[str] = []

        def fake_import(name: str, package: Optional[str] = None):
            import_events.append(name)
            if name == "torch":
                return SimpleNamespace(npu=SimpleNamespace())
            if name == "torch_npu":
                return SimpleNamespace()
            return original_import(name, package)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                "TRACE = []\n"
                "TRACE.append('module-imported')\n\n"
                "def build_operator_api(operator_module):\n"
                "    return operator_module.build_api()\n\n"
                "def build_differential_test_cases(operator_api):\n"
                "    return [\n"
                "        {'id': 'case-a', 'inputs': ('a',), 'fn': lambda: operator_api('a')}\n"
                "    ]\n",
                encoding="utf-8",
            )
            original_import = module.importlib.import_module
            with _without_preloaded_modules("torch", "torch_npu"):
                with patch.object(module.importlib, "import_module", side_effect=fake_import):
                    cases = module.load_differential_test_cases(test_file, operator)

        self.assertEqual([case.case_id for case in cases], ["case-a"])
        self.assertGreaterEqual(import_events[:2], ["torch", "torch_npu"])

    def test_load_differential_test_cases_selects_requested_case_id(self) -> None:
        module = load_test_contract_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                "def build_operator_api(operator_module):\n"
                "    return operator_module.build_api()\n\n"
                "def build_differential_test_cases(operator_api):\n"
                "    return [\n"
                "        {'id': 'case-a', 'inputs': ('a',), 'fn': lambda: operator_api('a')},\n"
                "        {'id': 'case-b', 'inputs': ('b',), 'fn': lambda: operator_api('b')},\n"
                "    ]\n",
                encoding="utf-8",
            )

            with patch.object(module, "bootstrap_torch_npu"):
                cases = module.load_differential_test_cases(test_file, operator, case_id="case-b")

        self.assertEqual([case.case_id for case in cases], ["case-b"])

    def test_load_differential_test_cases_rejects_unknown_case_id(self) -> None:
        module = load_test_contract_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                "def build_operator_api(operator_module):\n"
                "    return operator_module.build_api()\n\n"
                "def build_differential_test_cases(operator_api):\n"
                "    return [\n"
                "        {'id': 'case-a', 'inputs': ('a',), 'fn': lambda: operator_api('a')},\n"
                "        {'id': 'case-b', 'inputs': ('b',), 'fn': lambda: operator_api('b')},\n"
                "    ]\n",
                encoding="utf-8",
            )

            with patch.object(module, "bootstrap_torch_npu"):
                with self.assertRaisesRegex(
                    ValueError,
                    "Unknown differential test case id 'case-z'. Available case ids: case-a, case-b",
                ):
                    module.load_differential_test_cases(test_file, operator, case_id="case-z")

    def test_run_import_only_standalone_test_bootstraps_torch_before_user_module_exec(self) -> None:
        module = load_local_test_worker_module()
        import_events: list[str] = []

        def fake_import(name: str, package: Optional[str] = None):
            import_events.append(name)
            if name == "torch":
                return SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None))
            if name == "torch_npu":
                return SimpleNamespace()
            return original_import(name, package)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

MARKER = "loaded"

def main(operator_api):
    print(operator_api())
""",
                encoding="utf-8",
            )
            original_import = module.importlib.import_module
            with _without_preloaded_modules("torch", "torch_npu"):
                with patch.object(module.importlib, "import_module", side_effect=fake_import):
                    result = module._run_import_only_standalone_test(test_file, operator, verbose=False)

        self.assertEqual(result["return_code"], 0)
        self.assertGreaterEqual(import_events[:2], ["torch", "torch_npu"])

    def test_run_declarative_differential_test_bootstraps_before_importing_torch(self) -> None:
        module = load_local_test_worker_module()
        events: list[str] = []

        def fake_bootstrap(*_args: object) -> None:
            events.append("bootstrap")

        def fake_import(name: str, package: Optional[str] = None):
            events.append(f"import:{name}")
            if name == "torch":
                return SimpleNamespace(
                    save=lambda *_args, **_kwargs: None,
                    npu=SimpleNamespace(synchronize=lambda: None),
                )
            return original_import(name, package)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            archive_path = root / "abs_result.pt"
            operator.write_text("def build_api():\n    return lambda value: value.upper()\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    return [{"id": "case-a", "inputs": ("a",), "fn": lambda: operator_api("a")}]
""",
                encoding="utf-8",
            )
            original_import = module.importlib.import_module
            with patch.object(module, "_bootstrap_torch_npu", side_effect=fake_bootstrap), patch.object(
                module.importlib,
                "import_module",
                side_effect=fake_import,
            ):
                result = module._run_declarative_differential_test(
                    test_file,
                    operator,
                    archive_path,
                    verbose=False,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(events[0:2], ["bootstrap", "import:torch"])

    def test_run_local_test_prints_visible_devices_when_debug_enabled(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

def main(operator_api):
    print(operator_api())
""",
                encoding="utf-8",
            )
            stdout = StringIO()

            with patch.dict(
                os.environ,
                {
                    "HELIX_DEBUG": "1",
                    "ASCEND_RT_VISIBLE_DEVICES": "3",
                },
                clear=False,
            ):
                with patch("sys.stdout", stdout):
                    result, archived = module.run_local_test(
                        test_file,
                        operator,
                        "standalone",
                    )

        self.assertEqual(result["return_code"], 0)
        self.assertIsNone(archived)
        self.assertIn("[HELIX_DEBUG] ASCEND_RT_VISIBLE_DEVICES=3", stdout.getvalue())

    def test_run_local_test_launches_worker_subprocess_and_reads_result_file(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test_abs.py"
            operator = root / "abs.py"
            test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")

            result_file_holder: dict[str, Path] = {}

            def fake_buffered(
                command,
                workdir,
                stall_timeout_seconds,
                extra_env=None,
                *,
                timeout_seconds=None,
            ):
                del extra_env
                self.assertEqual(workdir, str(root.resolve()))
                self.assertEqual(stall_timeout_seconds, 0)
                self.assertEqual(timeout_seconds, 300)
                self.assertTrue(command[1].endswith("run_test_local_worker.py"))
                self.assertIn("local-test-worker", command)
                result_file = Path(command[command.index("--result-file") + 1])
                result_file_holder["path"] = result_file
                result_file.write_text(
                    json.dumps(
                        {
                            "result": {
                                "return_code": 0,
                                "stdout": "worker stdout\n",
                                "stderr": "",
                                "stalled": False,
                                "session_id": None,
                            },
                            "archived_result": None,
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                }

            with patch.object(module, "run_buffered_process", side_effect=fake_buffered, create=True):
                result, archived = module.run_local_test(test_file, operator, "standalone")

        self.assertEqual(result["stdout"], "worker stdout\n")
        self.assertIsNone(archived)
        self.assertIn("path", result_file_holder)
        self.assertEqual(result_file_holder["path"].name, "local-test-result.json")

    def test_local_worker_script_executes_standalone_test_and_writes_result_file(self) -> None:
        worker = load_local_test_worker_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test_abs.py"
            operator = root / "abs.py"
            result_file = root / "result.json"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                "# test-mode: standalone\n"
                "# compute-kind: compute\n"
                "# api-name: abs_entry\n"
                "# api-kind: torch-function\n\n"
                "def main(operator_api):\n"
                "    assert operator_api() == 1\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(worker.__file__ or "")),
                    "local-test-worker",
                    "--test-file",
                    str(test_file),
                    "--operator-file",
                    str(operator),
                    "--test-mode",
                    "standalone",
                    "--result-file",
                    str(result_file),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(result_file.read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(payload["result"]["return_code"], 0)
        self.assertIsNone(payload["archived_result"])

    def test_run_local_test_reads_archived_result_path_from_worker_payload(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archived_path = root / "abs_result.pt"
            test_file = root / "differential_test_abs.py"
            operator = root / "abs.py"
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            operator.write_text("def build_api():\n    return lambda value: value\n", encoding="utf-8")

            def fake_buffered(
                command,
                workdir,
                stall_timeout_seconds,
                extra_env=None,
                *,
                timeout_seconds=None,
            ):
                del workdir, stall_timeout_seconds, extra_env
                self.assertEqual(timeout_seconds, 300)
                result_file = Path(command[command.index("--result-file") + 1])
                archived_path.write_bytes(b"pt")
                result_file.write_text(
                    json.dumps(
                        {
                            "result": {
                                "return_code": 0,
                                "stdout": "",
                                "stderr": "",
                                "stalled": False,
                                "session_id": None,
                            },
                            "archived_result": str(archived_path),
                        }
                    ),
                    encoding="utf-8",
                )
                return {
                    "return_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                }

            with patch.object(module, "run_buffered_process", side_effect=fake_buffered, create=True):
                result, archived = module.run_local_test(test_file, operator, "differential")

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(archived, archived_path.resolve())

    def test_run_local_test_executes_declarative_differential_cases(self) -> None:
        module = load_local_test_worker_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    inputs_b = ("b",)
    return [
        {"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)},
        {"id": "case-b", "inputs": inputs_b, "fn": lambda: operator_api(*inputs_b)},
    ]
                """,
                encoding="utf-8",
            )

            def fake_save(obj, path) -> None:
                Path(path).write_bytes(pickle.dumps(obj))

            with patch.dict(sys.modules, {"torch": SimpleNamespace(save=fake_save)}, clear=False):
                result = module._run_declarative_differential_test(
                    test_file,
                    operator,
                    root / "abs_result.pt",
                )

            payload = pickle.loads((root / "abs_result.pt").read_bytes())

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [
                    {"id": "case-a", "inputs": ("a",), "result": "A"},
                    {"id": "case-b", "inputs": ("b",), "result": "B"},
                ],
            },
        )
        self.assertFalse((root / "TEST_RESULT.pt").exists())

    def test_run_local_test_case_payload_returns_selected_case_without_archive(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            (root / "torch.py").write_text(
                "class _Npu:\n"
                "    def synchronize(self):\n"
                "        pass\n\n"
                "npu = _Npu()\n",
                encoding="utf-8",
            )
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    return [
        {"id": "case-a", "inputs": ("a",), "fn": lambda: operator_api("a")},
        {"id": "case-b", "inputs": ("b",), "fn": lambda: operator_api("b")},
    ]
""",
                encoding="utf-8",
            )

            result, payload = module.run_local_test_case_payload(
                test_file,
                operator,
                case_id="case-b",
            )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
            },
        )
        self.assertFalse((root / "abs_result.pt").exists())

    def test_serialize_payload_object_normalizes_tensor_payloads_to_cpu(self) -> None:
        module = load_test_contract_module()
        tensor = _FakeTensor("B", "npu:0")

        def fake_import(name: str, package: Optional[str] = None):
            del package
            if name == "torch":
                return SimpleNamespace(Tensor=_FakeTensor)
            raise AssertionError(f"unexpected import: {name}")

        with patch.object(module.importlib, "import_module", side_effect=fake_import):
            serialized_payload = load_test_contract_module().serialize_payload_object(
                {
                    "compute": True,
                    "cases": [{"id": "case-b", "inputs": ("b",), "result": tensor}],
                }
            )

        payload = module.deserialize_payload_object(serialized_payload)
        result_tensor = payload["cases"][0]["result"]
        self.assertIsInstance(result_tensor, _FakeTensor)
        self.assertEqual(result_tensor.value, "B")
        self.assertEqual(result_tensor.device, "cpu")
        self.assertEqual(tensor.device, "npu:0")

    def test_run_local_test_imports_standalone_module_and_calls_main_with_operator_api(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text(
                "import torch\n\n"
                "def abs_entry():\n"
                "    return torch.tensor([1, 2], dtype=torch.int32)\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: non-compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

from npu_compare import compare_case_result

def main(operator_api):
    expected = operator_api()
    actual = operator_api()
    result = compare_case_result(
        case_id="case-a",
        actual=actual,
        golden=expected,
        inputs=(),
        compute=False,
    )
    if not result.passed:
        raise AssertionError(result.message)
    print(f"path={result.comparison_path} compute={result.compute}")
""",
                encoding="utf-8",
            )
            result, archived = module.run_local_test(test_file, operator, "standalone")

        self.assertEqual(result["return_code"], 0)
        self.assertIn("path=non-compute compute=False", result["stdout"])
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(archived)

    def test_run_local_test_passes_accuracy_mode_to_worker_env(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
            observed_env: Optional[dict[str, str]] = None
            fake_result = {
                "return_code": 1,
                "stdout": "",
                "stderr": "worker stopped before payload",
                "stalled": False,
                "session_id": None,
            }

            def fake_run_buffered_process(
                _command: list[str],
                _workdir: str,
                stall_timeout_seconds: int,
                extra_env: Optional[dict[str, str]] = None,
                *,
                timeout_seconds: Optional[float] = None,
            ) -> dict[str, object]:
                self.assertEqual(stall_timeout_seconds, 0)
                self.assertEqual(timeout_seconds, 300)
                nonlocal observed_env
                observed_env = extra_env
                return fake_result

            with patch.object(module, "run_buffered_process", side_effect=fake_run_buffered_process):
                result, archived = module.run_local_test(
                    test_file,
                    operator,
                    "standalone",
                    accuracy_mode="dtype-close",
                )

        self.assertEqual(result, fake_result)
        self.assertIsNone(archived)
        self.assertEqual(
            observed_env,
            {"HELIX_RUN_TEST_ACCURACY_MODE": "dtype-close"},
        )

    def test_run_local_test_reports_missing_differential_hooks_without_legacy_fallback(self) -> None:
        module = load_local_test_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text("def main():\n    return None\n", encoding="utf-8")
            result, archived = module.run_local_test(test_file, operator, "differential")

        self.assertEqual(result["return_code"], 1)
        self.assertIn("missing required hook 'build_operator_api'", result["stderr"])
        self.assertIsNone(archived)

    def test_run_local_test_filters_warning_prefix_lines_by_default(self) -> None:
        module = load_operator_eval_script_module("run_test_result")
        fake_result = {
            "return_code": 0,
            "stdout": _WARNING_LINE + "useful stdout\n" + _ANOTHER_WARNING_LINE,
            "stderr": _ANOTHER_WARNING_LINE + "useful stderr\n",
            "stalled": False,
            "session_id": None,
        }
        result = module.filter_result_payload(fake_result, verbose=False)

        self.assertEqual(result["stdout"], "useful stdout\n")
        self.assertEqual(result["stderr"], "useful stderr\n")

    def test_run_local_test_preserves_warning_prefix_lines_in_verbose_mode(self) -> None:
        module = load_operator_eval_script_module("run_test_result")
        fake_result = {
            "return_code": 0,
            "stdout": _WARNING_LINE + "useful stdout\n",
            "stderr": _ANOTHER_WARNING_LINE,
            "stalled": False,
            "session_id": None,
        }
        result = module.filter_result_payload(fake_result, verbose=True)

        self.assertEqual(result, fake_result)

    def test_run_remote_test_executes_declarative_differential_cases(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    return [{"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)}]
""",
                encoding="utf-8",
            )

            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }
            recorded_command: list[str] = []

            with patch.object(module, "create_remote_workspace", return_value=({"user_host": "user@host", "port": None}, "/tmp/remote")):
                with patch.object(module, "copy_file_to_remote"):
                    with patch.object(module, "run_remote_command_streaming", side_effect=lambda _spec, _workspace, command, **_kwargs: recorded_command.extend(command) or fake_result):
                        with patch.object(module, "_copy_remote_differential_archive", return_value=root / "abs_result.pt"):
                            with patch.object(module, "cleanup_remote_workspace"):
                                result, archived, remote_workspace = module.run_remote_test(
                                    test_file,
                                    operator,
                                    "differential",
                                    "user@host",
                                    None,
                                    keep_remote_workdir=False,
                                    verbose=False,
                                    stderr=None,
                                )

        self.assertEqual(result, fake_result)
        self.assertEqual(archived, root / "abs_result.pt")
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertEqual(
            recorded_command,
            [
                "python3", "run_test_remote_worker.py", "--test-file", test_file.name,
                "--operator-file", operator.name, "--test-mode", "differential",
            ],
        )

    def test_run_remote_test_case_payload_parses_serialized_payload_without_archive(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def build_api():\n    return lambda value: value.upper()\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            serialized_payload = load_test_contract_module().serialize_payload_object(
                {
                    "compute": True,
                    "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
                }
            )
            fake_result = {
                "return_code": 0,
                "stdout": (
                    f"{module._SERIALIZED_PAYLOAD_BEGIN}\n"
                    f"{serialized_payload}\n"
                    f"{module._SERIALIZED_PAYLOAD_END}\n"
                ),
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }
            with patch.object(
                module,
                "create_remote_workspace",
                return_value=({"user_host": "user@host", "port": None}, "/tmp/remote"),
            ), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=fake_result,
            ), patch.object(
                module,
                "_copy_remote_differential_archive",
                side_effect=AssertionError("remote archive copy should not run in case payload mode"),
            ), patch.object(module, "cleanup_remote_workspace"):
                result, payload, remote_workspace = module.run_remote_test_case_payload(
                    test_file,
                    operator,
                    "user@host",
                    None,
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
            },
        )
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertEqual(
            [path.name for path in self._batch_copy.call_args.args[1]],
            [
                "differential_test_abs.py",
                "abs.py",
                "npu_compare.py",
                "dtype_close_compare.py",
                "npu_compare_common.py",
                "npu_contract_compare.py",
                "env_registry.py",
                "run_test_remote_worker.py",
                "test_contract.py",
                "torch_npu_warnings.py",
            ],
        )

    def test_run_remote_test_case_payload_parses_crlf_serialized_payload_without_archive(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def build_api():\n    return lambda value: value.upper()\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            serialized_payload = load_test_contract_module().serialize_payload_object(
                {
                    "compute": True,
                    "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
                }
            )
            fake_result = {
                "return_code": 0,
                "stdout": (
                    f"{module._SERIALIZED_PAYLOAD_BEGIN}\r\n"
                    f"{serialized_payload}\r\n"
                    f"{module._SERIALIZED_PAYLOAD_END}\r\n"
                ),
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=({"user_host": "user@host", "port": None}, "/tmp/remote"),
            ), patch.object(module, "run_remote_command_streaming", return_value=fake_result), patch.object(
                module,
                "_copy_remote_differential_archive",
                side_effect=AssertionError("remote archive copy should not run in case payload mode"),
            ), patch.object(module, "cleanup_remote_workspace"):
                result, payload, remote_workspace = module.run_remote_test_case_payload(
                    test_file,
                    operator,
                    "user@host",
                    None,
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
            },
        )
        self.assertEqual(remote_workspace, "/tmp/remote")

    def test_run_remote_test_case_payload_ignores_warning_lines_inside_payload_markers(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def build_api():\n    return lambda value: value.upper()\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            serialized_payload = load_test_contract_module().serialize_payload_object(
                {
                    "compute": True,
                    "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
                }
            )
            fake_result = {
                "return_code": 0,
                "stdout": (
                    f"{module._SERIALIZED_PAYLOAD_BEGIN}\n"
                    "Warning: torch.save with legacy tensor serialization emitted a notice\n"
                    f"{serialized_payload}\n"
                    f"{module._SERIALIZED_PAYLOAD_END}\n"
                ),
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=({"user_host": "user@host", "port": None}, "/tmp/remote"),
            ), patch.object(module, "run_remote_command_streaming", return_value=fake_result), patch.object(
                module,
                "_copy_remote_differential_archive",
                side_effect=AssertionError("remote archive copy should not run in case payload mode"),
            ), patch.object(module, "cleanup_remote_workspace"):
                result, payload, remote_workspace = module.run_remote_test_case_payload(
                    test_file,
                    operator,
                    "user@host",
                    None,
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(result["stdout"], "")
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-b", "inputs": ("b",), "result": "B"}],
            },
        )
        self.assertEqual(remote_workspace, "/tmp/remote")

    def test_run_remote_test_uses_shared_eval_timeout(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

def main(operator_api):
    print(operator_api())
""",
                encoding="utf-8",
            )
            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }

            contract = load_test_contract_module()
            with patch.dict(
                contract.os.environ,
                {
                    "HELIX_EVAL_TIMEOUT_SECONDS": "300",
                    "HELIX_TEST_TIMEOUT_SECONDS": "900",
                },
                clear=False,
            ):
                with patch.object(
                    module,
                    "create_remote_workspace",
                    return_value=({"user_host": "user@host", "port": None}, "/tmp/remote"),
                ), patch.object(module, "copy_file_to_remote"), patch.object(
                    module,
                    "run_remote_command_streaming",
                    return_value=fake_result,
                ) as remote_run, patch.object(module, "cleanup_remote_workspace"):
                    result, archived, remote_workspace = module.run_remote_test(
                        test_file,
                        operator,
                        "standalone",
                        "user@host",
                        None,
                    )

        self.assertEqual(result, fake_result)
        self.assertIsNone(archived)
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertEqual(remote_run.call_args.kwargs["stall_timeout_seconds"], 300)

    def test_run_remote_test_forwards_accuracy_env_to_remote_command(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }

            contract = load_test_contract_module()
            with patch.dict(
                contract.os.environ,
                {
                    "HELIX_RUN_TEST_ACCURACY_MODE": "dtype-close",
                    "HELIX_RUN_TEST_ATOL": "0",
                    "HELIX_RUN_TEST_RTOL": "0.01",
                },
                clear=False,
            ), patch.object(
                module,
                "create_remote_workspace",
                return_value=({"user_host": "user@host", "port": None}, "/tmp/remote"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=fake_result,
            ) as remote_run, patch.object(module, "cleanup_remote_workspace"):
                result, archived, remote_workspace = module.run_remote_test(
                    test_file,
                    operator,
                    "standalone",
                    "user@host",
                    None,
                )

        self.assertEqual(result, fake_result)
        self.assertIsNone(archived)
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertEqual(
            remote_run.call_args.kwargs["extra_env"],
            {
                "TRITON_ALWAYS_COMPILE": "1",
                "HELIX_RUN_TEST_ACCURACY_MODE": "dtype-close",
                "HELIX_RUN_TEST_ATOL": "0",
                "HELIX_RUN_TEST_RTOL": "0.01",
            },
        )

    def test_remote_differential_generated_script_executes_successfully(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            (root / "torch.py").write_text(
                "import pickle\n\n"
                "def save(obj, path):\n"
                "    with open(path, 'wb') as handle:\n"
                "        pickle.dump(obj, handle)\n\n"
                "class _Npu:\n"
                "    def synchronize(self):\n"
                "        pass\n\n"
                "npu = _Npu()\n",
                encoding="utf-8",
            )
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    return [{"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)}]
""",
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(Path(module.__file__ or "").with_name("run_test_remote_worker.py")),
                "--test-file", test_file.name,
                "--operator-file", operator.name,
                "--test-mode", "differential",
            ]
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = pickle.loads((root / "abs_result.pt").read_bytes()) if (root / "abs_result.pt").exists() else None

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-a", "inputs": ("a",), "result": "A"}],
            },
        )

    def test_remote_differential_generated_script_accepts_mapping_cases(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            (root / "torch.py").write_text(
                "import pickle\n\n"
                "def save(obj, path):\n"
                "    with open(path, 'wb') as handle:\n"
                "        pickle.dump(obj, handle)\n\n"
                "class _Npu:\n"
                "    def synchronize(self):\n"
                "        pass\n\n"
                "npu = _Npu()\n",
                encoding="utf-8",
            )
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

from collections import UserDict

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    return [UserDict({"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)})]
""",
                encoding="utf-8",
            )

            command = [
                sys.executable,
                str(Path(module.__file__ or "").with_name("run_test_remote_worker.py")),
                "--test-file", test_file.name,
                "--operator-file", operator.name,
                "--test-mode", "differential",
            ]
            completed = subprocess.run(
                command,
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = pickle.loads((root / "abs_result.pt").read_bytes()) if (root / "abs_result.pt").exists() else None

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-a", "inputs": ("a",), "result": "A"}],
            },
        )

    def test_run_remote_test_executes_import_only_standalone_main(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

from npu_compare import compare_case_result

def main(operator_api):
    result = compare_case_result(
        case_id="case-a",
        actual=operator_api(),
        golden=operator_api(),
        inputs=(),
        compute=True,
    )
    if not result.passed:
        raise AssertionError(result.message)
""",
                encoding="utf-8",
            )

            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }
            recorded_command: list[str] = []

            with patch.object(module, "create_remote_workspace", return_value=({"user_host": "user@host", "port": None}, "/tmp/remote")):
                with patch.object(module, "run_remote_command_streaming", side_effect=lambda _spec, _workspace, command, **_kwargs: recorded_command.extend(command) or fake_result):
                    with patch.object(module, "cleanup_remote_workspace"):
                        result, archived, remote_workspace = module.run_remote_test(
                            test_file,
                            operator,
                            "standalone",
                            "user@host",
                            None,
                            keep_remote_workdir=False,
                            verbose=False,
                            stderr=None,
                        )

        self.assertEqual(result, fake_result)
        self.assertIsNone(archived)
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertEqual(
            [path.name for path in self._batch_copy.call_args.args[1]],
            [
                "test_abs.py", "abs.py", "npu_compare.py", "dtype_close_compare.py",
                "npu_compare_common.py", "npu_contract_compare.py", "env_registry.py",
                "run_test_remote_worker.py", "test_contract.py", "torch_npu_warnings.py",
            ],
        )
        self.assertEqual(
            recorded_command,
            [
                "python3", "run_test_remote_worker.py", "--test-file", test_file.name,
                "--operator-file", operator.name, "--test-mode", "standalone",
            ],
        )

    def test_run_remote_test_filters_warning_prefix_lines_by_default(self) -> None:
        module = load_remote_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")

            fake_result = {
                "return_code": 0,
                "stdout": _WARNING_LINE + "remote stdout\n",
                "stderr": _ANOTHER_WARNING_LINE + "remote stderr\n",
                "stalled": False,
                "session_id": None,
            }

            with patch.object(module, "create_remote_workspace", return_value=({"user_host": "user@host", "port": None}, "/tmp/remote")):
                with patch.object(module, "copy_file_to_remote"):
                    with patch.object(module, "run_remote_command_streaming", return_value=fake_result):
                        with patch.object(module, "cleanup_remote_workspace"):
                            result, archived, remote_workspace = module.run_remote_test(
                                test_file,
                                operator,
                                "standalone",
                                "user@host",
                                None,
                                keep_remote_workdir=False,
                                verbose=False,
                                stderr=None,
                            )

        self.assertEqual(result["stdout"], "remote stdout\n")
        self.assertEqual(result["stderr"], "remote stderr\n")
        self.assertIsNone(archived)
        self.assertEqual(remote_workspace, "/tmp/remote")

    def test_compare_result_files_compares_payloads_locally(self) -> None:
        module = load_compare_result_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("placeholder", encoding="utf-8")
            new.write_text("placeholder", encoding="utf-8")

            with patch.object(module, "compare_result_files", return_value=0) as compare_mock:
                return_code = module.compare_result_files(oracle, new)

            self.assertEqual(return_code, 0)
            compare_mock.assert_called_once_with(oracle, new)


if __name__ == "__main__":
    unittest.main()
