"""Capture the workbench pages, shared dropdown states, and pending indicator."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import json
import os
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import ImageGrab

from semantic_toolkit.ui import SemanticApp


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    app = SemanticApp(runtime_root=output / "profile")
    errors: list[str] = []
    app.report_callback_exception = lambda kind, value, tb: errors.append(
        "".join(traceback.format_exception(kind, value, tb))
    )
    user32 = ctypes.windll.user32

    def main_hwnd() -> int:
        child = int(app.winfo_id())
        return int(user32.GetAncestor(child, 2) or child)

    def capture(filename: str) -> None:
        app.lift()
        app.update_idletasks()
        app.update()
        ImageGrab.grab(window=main_hwnd()).save(output / filename)

    def capture_native_dropdown(filename: str) -> None:
        found: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def inspect_window(hwnd: int, _lparam: int) -> bool:
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
            class_name = ctypes.create_unicode_buffer(200)
            user32.GetClassNameW(hwnd, class_name, len(class_name))
            if (process_id.value == os.getpid() and user32.IsWindowVisible(hwnd)
                    and class_name.value == "#32768"):
                ImageGrab.grab(window=hwnd).save(output / filename)
                found.append(int(hwnd))
            return True

        callback = callback_type(inspect_window)
        user32.EnumWindows(callback, 0)
        if not found:
            raise AssertionError("系统下拉菜单窗口未出现，无法保存打开状态截图。")

    try:
        for key, filename in (("data", "01-data.png"), ("methods", "02-methods.png"),
                              ("results", "03-results.png"), ("tasks", "04-tasks.png"),
                              ("help", "05-help.png")):
            app.show_page(key)
            capture(filename)

        app.show_page("methods")
        combo = app.method_combo
        combo.configure(state="normal")
        combo._open_dropdown_menu()
        app.update_idletasks()
        capture_native_dropdown("07-dropdown-open.png")
        combo._dropdown_menu.close()
        combo.configure(state="disabled")
        capture("08-dropdown-disabled.png")
        combo.configure(state="normal")

        if errors:
            raise AssertionError("\n".join(errors))
        (output / "checks.json").write_text(json.dumps({
            "status": "PASS", "pages": ["data", "methods", "results", "tasks", "help"],
            "dropdown_menu_capture": True, "dropdown_disabled": True,
            "window_geometry": app.winfo_geometry(), "callback_errors": errors,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        app.destroy()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
