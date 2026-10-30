"""Capture the compute-acceleration controls and live environment status card."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import subprocess
import json
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import ImageGrab
from dataclasses import asdict

from semantic_toolkit.analysis import AnalysisConfig
from semantic_toolkit.settings_io import validate_settings
from semantic_toolkit.ui import SemanticApp


def run(output: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("This capture script uses the Windows app-window capture API.")
    output.mkdir(parents=True, exist_ok=True)
    app = SemanticApp(runtime_root=output / "profile")
    errors: list[str] = []
    app.report_callback_exception = lambda kind, value, tb: errors.append(
        "".join(traceback.format_exception(kind, value, tb))
    )
    user32 = ctypes.windll.user32

    def pump(seconds: float = .25) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.update()
            time.sleep(.01)
        if errors:
            raise AssertionError("\n".join(errors))

    def capture(filename: str) -> None:
        app.lift()
        app.update_idletasks()
        app.update()
        child = int(app.winfo_id())
        hwnd = int(user32.GetAncestor(child, 2) or child)
        ImageGrab.grab(window=hwnd).save(output / filename)

    def capture_recovery_window() -> None:
        # Windows venv python.exe is a launcher; its actual Python child owns the window.
        pid_marker = output / "recovery.pid"
        pid_marker.unlink(missing_ok=True)
        launcher = ("import os,sys,runpy; from pathlib import Path; "
                    "Path(sys.argv[1]).write_text(str(os.getpid()),encoding='ascii'); "
                    "entry=sys.argv[2]; sys.argv=[entry,'--repair-environment']; "
                    "runpy.run_path(entry,run_name='__main__')")
        command = [sys.executable, "-c", launcher, str(pid_marker), str(ROOT / "main.py")]
        process = subprocess.Popen(command, cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + 45
        recovery_hwnd = 0
        while time.monotonic() < deadline and process.poll() is None:
            found: list[int] = []
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def inspect_window(hwnd, _lparam):
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, len(title))
                owner_pid = int(pid_marker.read_text(encoding="ascii")) if pid_marker.exists() else process.pid
                if pid.value == owner_pid and title.value.startswith("PYsemantic · 环境检查与修复"):
                    found.append(int(hwnd))
                return True

            user32.EnumWindows(callback_type(inspect_window), 0)
            if found:
                recovery_hwnd = found[0]
                break
            time.sleep(.1)
        try:
            if not recovery_hwnd:
                raise AssertionError("Recovery window did not open within 45 seconds.")
            deadline = time.monotonic() + 30
            report_ready = False
            while time.monotonic() < deadline and not report_ready:
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(recovery_hwnd, title, len(title))
                report_ready = "检查完成" in title.value
                if not report_ready:
                    time.sleep(.2)
            if not report_ready:
                raise AssertionError("Recovery environment report did not finish within 30 seconds.")
            ImageGrab.grab(window=recovery_hwnd).save(output / "04-recovery-window.png")
            user32.PostMessageW(recovery_hwnd, 0x0010, 0, 0)  # WM_CLOSE
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    try:
        app.show_page("methods")
        pump()
        baseline = app._config_from_ui()
        assert baseline == AnalysisConfig()
        capture("01-methods-acceleration-default.png")

        app.parallel_enabled_check.toggle()
        app.cpu_workers_var.set("2")
        app.bge_device_var.set("自动使用可用 GPU")
        assert app.cpu_workers_entry.cget("state") == "normal"
        app.busy = True
        app._set_busy_controls(True)
        assert app.parallel_enabled_check.cget("state") == "disabled"
        assert app.bge_device_combo.cget("state") == "disabled"
        app.busy = False
        app._set_busy_controls(False)
        configured = app._config_from_ui()
        snapshot = validate_settings(app._settings_snapshot())
        assert configured.parallel_enabled and configured.cpu_workers == 2
        assert configured.bge_device == "auto"
        assert snapshot["cpu_workers"] == 2 and snapshot["bge_device"] == "auto"
        app._apply_settings(asdict(AnalysisConfig()))
        assert app._config_from_ui() == AnalysisConfig()
        app._apply_settings(snapshot)
        assert app._config_from_ui() == configured
        capture("02-methods-acceleration-configured.png")

        app.show_page("tasks")
        app._start_environment_check()
        deadline = time.monotonic() + 60
        while app._environment_checking and time.monotonic() < deadline:
            pump(.1)
        if app.environment_report is None:
            raise AssertionError("Environment check did not return a report within 60 seconds.")
        capture("03-tasks-environment.png")
        # No pip process is started: only test the close/deferred-terminal UI path.
        for terminal in (("environment-install-finished", (-15, True, "")),
                         ("environment-error", "synthetic install failure")):
            app._environment_installing = True
            app._environment_close_pending = False
            app._environment_cancel_event.clear()
            with patch.object(app, "destroy") as destroy:
                app._on_close()
                assert app._environment_cancel_event.is_set()
                assert app._environment_close_pending
                destroy.assert_not_called()
                app.environment_queue.put(terminal)
                app._poll_environment_events()
                destroy.assert_called_once()
            app._environment_close_pending = False
        (output / "checks.json").write_text(json.dumps({
            "status": "PASS",
            "config_defaults": asdict(baseline),
            "configured_acceleration": asdict(configured),
            "settings_round_trip": True,
            "install_close_waits_for_terminal_mocked": True,
            "environment_dependencies_ready": app.environment_report.dependencies_ready,
            "onnx_providers": app.environment_report.onnx_providers,
            "model_available": app.environment_report.model_available,
            "geometry": app.winfo_geometry(),
            "callback_errors": errors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        app.destroy()
    capture_recovery_window()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
