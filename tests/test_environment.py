"""Dependency repair boundaries, without changing the test machine's packages."""
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from io import StringIO
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

from semantic_toolkit import environment as env


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.pins = env.read_runtime_pins()

    def inspect(self, missing=(), alternate=None):
        def importer(name):
            if name in missing:
                raise ImportError("synthetic missing dependency")
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])

        def version(name):
            if name == alternate:
                return "1.24.4"
            if name == "onnxruntime" and alternate:
                raise PackageNotFoundError(name)
            if name in self.pins:
                return self.pins[name]
            raise PackageNotFoundError(name)

        return env.inspect_environment(module_importer=importer, version_reader=version)

    def test_complete_and_missing_runtime(self):
        complete = self.inspect()
        self.assertTrue(complete.dependencies_ready)
        self.assertEqual(complete.onnx_providers, ("CPUExecutionProvider",))
        missing = self.inspect(missing={"numpy"})
        self.assertEqual([item.distribution for item in missing.missing_dependencies], ["numpy"])
        self.assertIn("synthetic missing", env.format_environment_report(missing))

    def test_install_uses_current_interpreter_and_only_declared_pins(self):
        report = self.inspect(missing={"numpy"})
        command = env.build_install_command(report.missing_dependencies)
        self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn("numpy==" + self.pins["numpy"], command)
        injected = replace(report.missing_dependencies[0], distribution="untrusted-package")
        self.assertEqual(env.build_install_command([injected]), [])
        self.assertEqual(env.build_install_command(self.inspect().dependencies), [])

    def test_broken_installed_package_is_reinstalled(self):
        broken = self.inspect(missing={"numpy"}).missing_dependencies
        self.assertIn("--force-reinstall", env.build_install_command(broken))

    def test_gpu_distribution_is_preserved(self):
        for alternate in ("onnxruntime-gpu", "onnxruntime-directml"):
            with self.subTest(alternate=alternate):
                report = self.inspect(alternate=alternate)
                self.assertTrue(report.dependencies_ready)
                self.assertEqual(env.build_install_command(report.dependencies), [])

    def test_frozen_runtime_never_launches_pip(self):
        report = self.inspect(missing={"numpy"})
        with self.assertRaises(env.FrozenRuntimeError):
            env.build_install_command(report.missing_dependencies, frozen=True)

    def test_install_failure_returns_output_and_exit_status(self):
        output = []
        process = SimpleNamespace(stdout=StringIO("synthetic pip failure\n"), returncode=7, poll=lambda: 7)
        with patch.object(env.subprocess, "Popen", return_value=process) as popen:
            code, cancelled = env.run_install_command(
                env.build_install_command(self.inspect(missing={"numpy"}).missing_dependencies),
                on_output=output.append,
            )
        self.assertEqual((code, cancelled), (7, False))
        self.assertEqual(output, ["synthetic pip failure"])
        self.assertFalse(popen.call_args.kwargs["shell"])

    def test_install_can_cancel_its_owned_process(self):
        cancelled = Event()
        cancelled.set()
        process = SimpleNamespace(stdout=StringIO(""), returncode=None)
        process.poll = lambda: process.returncode
        process.terminate = lambda: setattr(process, "returncode", -15)
        process.wait = lambda timeout=None: process.returncode
        with patch.object(env.subprocess, "Popen", return_value=process):
            code, was_cancelled = env.run_install_command(
                env.build_install_command(self.inspect(missing={"numpy"}).missing_dependencies),
                cancel_event=cancelled,
            )
        self.assertTrue(was_cancelled)
        self.assertEqual(code, -15)


if __name__ == "__main__":
    unittest.main()
