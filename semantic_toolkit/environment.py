"""Standard-library environment inspection and source-only dependency repair."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata, util
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable
import os
import re
import subprocess
import sys
import time


# Only these application runtime dependencies may be installed by the in-app
# repair action. Versions are read from the repository's pinned lock file.
RUNTIME_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("customtkinter", "customtkinter"),
    ("jieba", "jieba"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("threadpoolctl", "threadpoolctl"),
    ("tokenizers", "tokenizers"),
    ("onnxruntime", "onnxruntime"),
    ("openpyxl", "openpyxl"),
)
_ALLOWED_DISTRIBUTIONS = {name for name, _module in RUNTIME_DISTRIBUTIONS}
BOOT_IMPORT_DISTRIBUTIONS = {
    "customtkinter", "jieba", "numpy", "pandas", "scipy", "scikit-learn",
}
_LOCK_LINE = re.compile(r"^\s*([A-Za-z0-9_.-]+)==([^\s;]+)\s*(?:#.*)?$")
class FrozenRuntimeError(RuntimeError):
    """Raised when a packaged runtime is asked to mutate its embedded Python."""


@dataclass(frozen=True)
class DependencyStatus:
    distribution: str
    module: str
    required_version: str | None
    installed_version: str | None
    importable: bool
    error: str = ""
    alternate_onnxruntime: bool = False

    @property
    def ready(self) -> bool:
        # onnxruntime-gpu and onnxruntime-directml expose the same
        # `onnxruntime` module. Preserve either runtime even if the CPU wheel's
        # distribution metadata is absent or its version differs from the lock.
        if self.alternate_onnxruntime and self.importable:
            return True
        if self.distribution == "onnxruntime" and self.importable:
            # Treat any successfully importable ONNX Runtime as usable. The
            # provider wheel may intentionally differ from the CPU lock pin.
            return True
        return self.importable and (
            self.required_version is None or self.installed_version == self.required_version
        )


@dataclass(frozen=True)
class EnvironmentReport:
    executable: str
    frozen: bool
    cpu_cores: int | None
    model_available: bool
    model_directory: str
    dependencies: tuple[DependencyStatus, ...]
    onnx_providers: tuple[str, ...]
    onnx_error: str = ""

    @property
    def missing_dependencies(self) -> tuple[DependencyStatus, ...]:
        return tuple(item for item in self.dependencies if not item.ready)

    @property
    def dependencies_ready(self) -> bool:
        return not self.missing_dependencies

    @property
    def repairable_dependencies(self) -> tuple[DependencyStatus, ...]:
        return tuple(
            item for item in self.missing_dependencies
            if not (item.distribution == "onnxruntime" and item.alternate_onnxruntime)
        )


def is_frozen_runtime() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Return the source root, or the one-file extraction root when frozen."""
    if is_frozen_runtime():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def requirements_lock_path() -> Path:
    return project_root() / "requirements-lock.txt"


def _canonical_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_runtime_pins(lock_path: str | Path | None = None) -> dict[str, str]:
    """Read exact versions for the fixed runtime allowlist from the lock file."""
    path = Path(lock_path) if lock_path is not None else requirements_lock_path()
    if not path.is_file():
        raise FileNotFoundError(f"找不到固定依赖清单：{path}")
    wanted = {_canonical_distribution(name): name for name in _ALLOWED_DISTRIBUTIONS}
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _LOCK_LINE.match(line)
        if not match:
            continue
        canonical = _canonical_distribution(match.group(1))
        if canonical in wanted:
            pins[wanted[canonical]] = match.group(2)
    missing = sorted(_ALLOWED_DISTRIBUTIONS - pins.keys())
    if missing:
        raise ValueError(f"固定依赖清单缺少运行时版本：{', '.join(missing)}")
    return pins


def _model_directory() -> Path:
    return project_root() / "assets" / "bge-small-zh-v1.5"


def inspect_environment(
    *,
    lock_path: str | Path | None = None,
    module_importer: Callable[[str], object] = import_module,
    version_reader: Callable[[str], str] = metadata.version,
    frozen: bool | None = None,
) -> EnvironmentReport:
    """Check runtime imports and versions without importing app or UI modules.

    Call from a background thread in the main app. The recovery window itself
    remains usable with only Python's standard library and tkinter.
    """
    pins: dict[str, str] = {}
    pin_error = ""
    if frozen is not True and not (frozen is None and is_frozen_runtime()):
        try:
            pins = read_runtime_pins(lock_path)
        except Exception as exc:
            pin_error = f"固定依赖清单不可读取：{type(exc).__name__}: {exc}"

    statuses: list[DependencyStatus] = []
    for distribution, module_name in RUNTIME_DISTRIBUTIONS:
        module_value: object | None = None
        try:
            module_value = module_importer(module_name)
            importable = True
            error = ""
        except Exception as exc:  # includes binary import / DLL load errors
            importable = False
            error = f"{type(exc).__name__}: {exc}"
        try:
            version = version_reader(distribution)
        except Exception:
            version = None
        if version is None and module_value is not None:
            module_version = getattr(module_value, "__version__", None)
            version = str(module_version) if module_version is not None else None
        if distribution == "onnxruntime":
            # If a CUDA/DirectML wheel provides the module, its distribution
            # name may differ from the CPU wheel. Never "repair" it by adding
            # onnxruntime from the lock file.
            try:
                alternate = version_reader("onnxruntime-gpu")
                alternate_onnxruntime = bool(alternate)
            except Exception:
                try:
                    alternate = version_reader("onnxruntime-directml")
                    alternate_onnxruntime = bool(alternate)
                except Exception:
                    alternate_onnxruntime = False
            if alternate_onnxruntime:
                version = str(alternate)
        else:
            alternate_onnxruntime = False
        statuses.append(DependencyStatus(
            distribution=distribution,
            module=module_name,
            required_version=pins.get(distribution),
            installed_version=version,
            importable=importable,
            error=error,
            alternate_onnxruntime=alternate_onnxruntime,
        ))

    providers: tuple[str, ...] = ()
    onnx_error = pin_error
    try:
        ort = module_importer("onnxruntime")
        providers = tuple(str(name) for name in ort.get_available_providers())
    except Exception as exc:
        onnx_error = "; ".join(filter(None, (onnx_error, f"{type(exc).__name__}: {exc}")))

    model_dir = _model_directory()
    model_available = (model_dir / "onnx" / "model.onnx").is_file() and (model_dir / "tokenizer.json").is_file()
    return EnvironmentReport(
        executable=str(Path(sys.executable).resolve()),
        frozen=is_frozen_runtime() if frozen is None else frozen,
        cpu_cores=os.cpu_count(),
        model_available=model_available,
        model_directory=str(model_dir),
        dependencies=tuple(statuses),
        onnx_providers=providers,
        onnx_error=onnx_error,
    )


def build_install_command(
    dependencies: tuple[DependencyStatus, ...] | list[DependencyStatus],
    *,
    lock_path: str | Path | None = None,
    executable: str | None = None,
    frozen: bool | None = None,
) -> list[str]:
    """Build a shell-free pip command containing only missing allowlisted pins."""
    if is_frozen_runtime() if frozen is None else frozen:
        raise FrozenRuntimeError("打包程序的 Python 运行环境不可由应用内安装器修改；请重新安装包含依赖的完整版本。")
    pins = read_runtime_pins(lock_path)
    candidates: list[str] = []
    seen: set[str] = set()
    force_reinstall = False
    for item in dependencies:
        name = item.distribution
        if name not in _ALLOWED_DISTRIBUTIONS or name in seen or item.ready:
            continue
        if name == "onnxruntime" and (item.importable or item.alternate_onnxruntime):
            # Any importable ORT distribution is an intentional CPU/CUDA/DML
            # choice. Do not replace it with the CPU wheel from the lock.
            continue
        if item.installed_version and not item.importable:
            # `pip install pinned==version` otherwise says the broken wheel is
            # already satisfied and leaves a DLL/import failure unchanged.
            force_reinstall = True
        candidates.append(f"{name}=={pins[name]}")
        seen.add(name)
    if not candidates:
        return []
    command = [executable or sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    if force_reinstall:
        command.append("--force-reinstall")
    command.extend(candidates)
    return command


def run_install_command(
    command: list[str],
    *,
    cancel_event: Event | None = None,
    on_output: Callable[[str], None] | None = None,
) -> tuple[int, bool]:
    """Run pip without a shell, stream output, and allow cooperative cancellation."""
    if not command:
        return 0, False
    if len(command) < 4 or command[1:3] != ["-m", "pip"] or command[3] != "install":
        raise ValueError("仅允许执行固定的 Python pip install 命令。")
    for argument in command[4:]:
        if argument.startswith("-"):
            continue
        name = argument.split("==", 1)[0]
        if name not in _ALLOWED_DISTRIBUTIONS or "==" not in argument:
            raise ValueError("安装项不在固定运行时依赖白名单中。")
    process = subprocess.Popen(
        command,
        cwd=str(project_root()),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_queue: Queue[str | None] = Queue()

    def read_output() -> None:
        stream = process.stdout
        if stream is None:
            output_queue.put(None)
            return
        try:
            for line in stream:
                output_queue.put(line.rstrip())
        finally:
            output_queue.put(None)

    reader = Thread(target=read_output, name="environment-pip-output", daemon=True)
    reader.start()
    ended_output = False
    while process.poll() is None or not ended_output:
        try:
            while True:
                line = output_queue.get_nowait()
                if line is None:
                    ended_output = True
                elif line and on_output:
                    on_output(line)
        except Empty:
            pass
        if cancel_event is not None and cancel_event.is_set() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            reader.join(timeout=2)
            return process.returncode or 1, True
        if process.poll() is None or not ended_output:
            time.sleep(0.05)
    reader.join(timeout=2)
    return int(process.returncode or 0), False


def format_environment_report(report: EnvironmentReport) -> str:
    runtime_label = "打包 EXE（内置运行时，只读）" if report.frozen else "源码运行（当前 Python）"
    lines = [
        f"运行方式：{runtime_label}",
        f"Python：{report.executable}",
        f"CPU 逻辑核心：{report.cpu_cores if report.cpu_cores is not None else '未知'}",
        f"BGE 模型：{'齐全' if report.model_available else '缺少模型或分词器'}（{report.model_directory}）",
        f"ONNX Runtime providers：{', '.join(report.onnx_providers) if report.onnx_providers else '不可读取'}",
        "依赖：",
    ]
    for item in report.dependencies:
        if item.ready:
            detail = f"可用，版本 {item.installed_version or '未知'}"
            if item.required_version and item.installed_version != item.required_version and not item.alternate_onnxruntime:
                detail += f"（锁定版本 {item.required_version}）"
            if item.alternate_onnxruntime:
                detail += "（保留当前 CUDA/DirectML 运行时）"
        elif not item.importable:
            detail = "缺失或无法导入" + (f"：{item.error}" if item.error else "")
            if item.alternate_onnxruntime:
                detail += "（检测到所选 CUDA/DirectML 运行时；安装器不会替换它，请按官方说明修复）"
        else:
            detail = f"版本 {item.installed_version or '未知'}，锁定版本 {item.required_version or '未知'}"
        lines.append(f"  • {item.distribution}: {detail}")
    if report.onnx_error:
        lines.append(f"  ONNX Runtime 检查：{report.onnx_error}")
    if report.frozen:
        lines.append("打包程序不能修改内置 Python；缺少运行依赖时请重新安装完整版本。")
    elif report.missing_dependencies:
        lines.append("可点击“一键安装缺失依赖”；安装结束后请重启程序。")
    else:
        lines.append("运行依赖已就绪。")
    if not report.model_available:
        lines.append("模型资源不通过 pip 修复；源码运行请按 README 准备本地模型，EXE 请重新安装完整版本。")
    return "\n".join(lines)


def _set_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try:
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        except Exception:
            pass
        try:
            if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
                return
        except Exception:
            pass
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def launch_recovery_window(*, initial_error: str = "", auto_install: bool = False) -> int:
    """Open a stdlib tkinter recovery window that works without app packages."""
    _set_windows_dpi_awareness()
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("PYsemantic · 环境检查与修复")
    root.geometry("860x620")
    root.minsize(680, 500)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    header = ttk.Label(root, text="PYsemantic 环境检查与修复", font=("Segoe UI", 15, "bold"))
    header.grid(row=0, column=0, sticky="w", padx=16, pady=(15, 6))
    text = tk.Text(root, wrap="word", height=24, padx=10, pady=8)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)
    text.grid(row=1, column=0, sticky="nsew", padx=(16, 0), pady=8)
    scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 16), pady=8)
    status = ttk.Label(root, text="正在检查环境…")
    status.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 6))
    buttons = ttk.Frame(root)
    buttons.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 14))
    events: Queue[tuple[str, object]] = Queue()
    cancel_event = Event()
    state = {"checking": False, "installing": False, "report": None,
             "close_after": False, "auto_install_started": False, "restart_required": False}

    def set_text(value: str, *, append: bool = False) -> None:
        text.configure(state="normal")
        if not append:
            text.delete("1.0", "end")
        text.insert("end" if append else "1.0", value + ("\n" if append else ""))
        text.see("end" if append else "1.0")
        text.configure(state="disabled")

    def update_buttons() -> None:
        report = state["report"]
        install_enabled = bool(report and report.repairable_dependencies and not report.frozen and not state["installing"])
        install_button.configure(state="normal" if install_enabled else "disabled")
        check_button.configure(state="disabled" if state["checking"] or state["installing"] else "normal")
        cancel_button.configure(state="normal" if state["installing"] else "disabled")

    def check_environment() -> None:
        if state["checking"] or state["installing"]:
            return
        state["checking"] = True
        status.configure(text="正在检查包导入、版本、模型与 ONNX Runtime providers…")
        update_buttons()

        def work() -> None:
            try:
                events.put(("check", inspect_environment()))
            except Exception as exc:
                events.put(("failure", f"检查失败：{type(exc).__name__}: {exc}"))

        Thread(target=work, name="environment-check", daemon=True).start()

    def install_missing() -> None:
        report = state["report"]
        if not report or report.frozen or not report.repairable_dependencies or state["installing"]:
            return
        state["installing"] = True
        state["close_after"] = False
        cancel_event.clear()
        status.configure(text="正在安装锁定版本；可取消当前 pip 子进程。")
        set_text(format_environment_report(report) + "\n\n安装输出：")
        update_buttons()

        def work() -> None:
            try:
                command = build_install_command(report.repairable_dependencies, frozen=report.frozen)
                if not command:
                    events.put(("install", (0, False, "没有需要安装的包。")))
                    return
                code, cancelled = run_install_command(
                    command, cancel_event=cancel_event,
                    on_output=lambda line: events.put(("output", line)),
                )
                events.put(("install", (code, cancelled, "")))
            except Exception as exc:
                events.put(("failure", f"安装失败：{type(exc).__name__}: {exc}"))

        Thread(target=work, name="environment-install", daemon=True).start()

    def cancel_install() -> None:
        cancel_event.set()
        status.configure(text="正在取消安装…")

    check_button = ttk.Button(buttons, text="检查环境", command=check_environment)
    check_button.pack(side="left", padx=(0, 8))
    install_button = ttk.Button(buttons, text="一键安装缺失依赖", command=install_missing)
    install_button.pack(side="left", padx=(0, 8))
    cancel_button = ttk.Button(buttons, text="取消安装", command=cancel_install, state="disabled")
    cancel_button.pack(side="left", padx=(0, 8))
    close_button = ttk.Button(buttons, text="关闭")
    close_button.pack(side="right")

    def close_window() -> None:
        if state["installing"]:
            state["close_after"] = True
            cancel_event.set()
            status.configure(text="正在取消 pip；结束后关闭窗口。")
            close_button.configure(state="disabled")
            update_buttons()
            return
        root.destroy()

    close_button.configure(command=close_window)
    root.protocol("WM_DELETE_WINDOW", close_window)
    if initial_error:
        set_text(f"主界面未能加载：{initial_error}\n正在检查当前 Python 中的依赖。")
    else:
        set_text("正在检查当前运行环境。")

    def poll() -> None:
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "output":
                    set_text(str(payload), append=True)
                elif kind == "check":
                    state["checking"] = False
                    state["report"] = payload
                    detail = format_environment_report(payload)
                    if state["restart_required"]:
                        detail += "\n\n依赖安装已完成；请关闭并重新启动程序后再使用新依赖。"
                    set_text(detail)
                    status.configure(text="检查完成；安装后请重启程序。" if state["restart_required"] else "检查完成。")
                    root.title("PYsemantic · 环境检查与修复 · 检查完成")
                    update_buttons()
                    if (auto_install and not state["auto_install_started"]
                            and payload.repairable_dependencies and not payload.frozen):
                        state["auto_install_started"] = True
                        root.after(100, install_missing)
                elif kind == "install":
                    state["installing"] = False
                    code, cancelled, message = payload
                    if message:
                        set_text(message, append=True)
                    elif cancelled:
                        set_text("\n安装已取消。", append=True)
                    elif code == 0:
                        state["restart_required"] = True
                        set_text("\n安装完成，正在重新检查环境…", append=True)
                        status.configure(text="安装完成；请在检查结束后重启程序再使用新依赖。")
                        root.after(150, check_environment)
                    else:
                        set_text(f"\n安装失败，pip 退出码：{code}。请检查上方输出。", append=True)
                        status.configure(text=f"安装失败（退出码 {code}）。")
                    update_buttons()
                    if state["close_after"]:
                        root.destroy()
                        return
                elif kind == "failure":
                    state["checking"] = False
                    state["installing"] = False
                    set_text(str(payload), append=True)
                    status.configure(text="操作未完成；请查看错误信息。")
                    update_buttons()
                    if state["close_after"]:
                        root.destroy()
                        return
        except Empty:
            pass
        if root.winfo_exists():
            root.after(100, poll)

    root.after(100, poll)
    root.after(150, check_environment)
    root.mainloop()
    return 0


def missing_import_specs() -> tuple[str, ...]:
    """Fast probe used before importing the heavy GUI/scientific modules."""
    missing: list[str] = []
    for distribution, module_name in RUNTIME_DISTRIBUTIONS:
        if distribution not in BOOT_IMPORT_DISTRIBUTIONS:
            continue
        if util.find_spec(module_name) is None:
            missing.append(distribution)
    return tuple(missing)


def should_open_recovery(exc: BaseException) -> bool:
    """Return whether a UI import failure can plausibly be repaired via pip."""
    return isinstance(exc, (ImportError, OSError))
