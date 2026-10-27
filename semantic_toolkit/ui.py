"""Chinese four-page desktop interface for the offline semantic toolkit."""
from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from dataclasses import asdict, replace
from typing import Any, Callable
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import webbrowser

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from .analysis import AnalysisConfig, AnalysisResult, METHODS, run_analysis
from .data_io import Corpus, Document, load_corpus, table_columns
from .eta import EtaTracker, STAGE_LABELS, StageEstimate, estimate_stage_plan
from .resources import resource_path
from .result_views import ResultsExplorer
from .runtime import (RuntimeStore, close_owned_process_tree, corpus_to_dict, load_project_full,
                      owned_process_tree_finished, popen_owned_process, result_cache_load,
                      terminate_owned_process)


INK = "#27324A"
CANVAS = "#FCF8F2"
GREEN = "#20C46A"
MINT = "#E9FBF1"
SKY = "#E6F7FB"
CORAL = "#FFB4AA"
LEMON = "#F1F9D8"
LAVENDER = "#E6E6FA"
WHITE = "#FFFFFF"
BODY_FONT = ("Microsoft YaHei UI", 13)
SMALL_FONT = ("Microsoft YaHei UI", 11)
TITLE_FONT = ("Microsoft YaHei UI", 23, "bold")
SECTION_FONT = ("Microsoft YaHei UI", 16, "bold")


def set_windows_dpi_awareness() -> None:
    """Request per-monitor DPI before constructing Tk; fall back for older Windows."""
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
            if ctypes.windll.shcore.SetProcessDpiAwareness(1) == 0:
                return
        except Exception:
            user32.SetProcessDPIAware()
    except Exception:
        pass


def set_app_user_model_id() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("OpenAI.ChineseSemanticToolkit.1")
        except Exception:
            pass


class SemanticApp(ctk.CTk):
    PAGE_COLORS = {"data": SKY, "methods": MINT, "results": CORAL, "help": LAVENDER, "tasks": LEMON}
    PAGE_TITLES = {"data": ("01  数据准备", "导入文本，检查来源与清理统计"),
                   "methods": ("02  方法设置", "选择语义表示、分词和计算参数"),
                   "results": ("03  结果查看", "浏览指标、主题、近邻与相似度矩阵"),
                   "help": ("04  帮助与方法", "离线使用说明、来源文献和许可信息"),
                   "tasks": ("05  任务、日志与项目", "阶段进度、历史记录、已保存项目与方法预设")}

    def __init__(self, ui_smoke_dir: str | None = None, runtime_root: str | Path | None = None):
        set_windows_dpi_awareness()
        set_app_user_model_id()
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("中文语义分析工具包")
        self.configure(fg_color=CANVAS)
        self._center_window()
        self._set_icon()
        self.option_add("*Font", BODY_FONT)
        self.corpus: Corpus | None = None
        self.result: AnalysisResult | None = None
        self.results_by_method: dict[str, AnalysisResult] = {}
        self.result_sources: dict[str, Path] = {}
        self.result_corpora: dict[str, Corpus] = {}
        self.result_elapsed: dict[str, float] = {}
        self.result_runtime_metadata: dict[str, Any] = {}
        self.runtime = RuntimeStore(runtime_root)
        self.current_page = "data"
        self.worker_queue: Queue = Queue()
        self.cancel_event: Event | None = None
        self.worker: Thread | None = None
        self.current_process: subprocess.Popen | None = None
        self._process_lock = __import__("threading").Lock()
        self.busy = False
        self.active_run_id: str | None = None
        self.eta_tracker: EtaTracker | None = None
        self._close_after_task = False
        self._last_status_payload = ""
        self._result_selection_restore = False
        self.ui_smoke_dir = Path(ui_smoke_dir) if ui_smoke_dir else None
        self.smoke_capture_count = 0
        self.smoke_required_captures = 11
        self.smoke_errors: list[str] = []
        self.page_widgets: dict[str, ctk.CTkFrame] = {}
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self._all_controls: list[Any] = []
        self._busy_widget_states: dict[Any, str] = {}
        self._build_shell()
        self._build_data_page()
        self._build_method_page()
        self._build_results_page()
        self._build_help_page()
        self._build_tasks_page()
        self._restore_last_settings()
        self.show_page("data")
        self.after(100, self._poll_worker)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.ui_smoke_dir:
            self.after(900, lambda: self._smoke_step(0))

    def _center_window(self) -> None:
        """Size and center the scaled CTk window inside the active monitor work area."""
        self.update_idletasks()
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            class MonitorInfo(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                            ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

            child_hwnd = int(self.winfo_id())
            root_hwnd = user32.GetAncestor(child_hwnd, 2) or child_hwnd
            monitor = user32.MonitorFromWindow(root_hwnd, 2)
            info = MonitorInfo()
            info.cbSize = ctypes.sizeof(MonitorInfo)
            if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                work = info.rcWork
                work_width = max(1, work.right - work.left)
                work_height = max(1, work.bottom - work.top)
            else:
                work_width = max(1, user32.GetSystemMetrics(0))
                work_height = max(1, user32.GetSystemMetrics(1) - 48)
                work = wintypes.RECT(0, 0, work_width, work_height)
            try:
                dpi_scale = float(user32.GetDpiForWindow(root_hwnd)) / 96.0
            except Exception:
                dpi_scale = 1.0
            try:
                ctk_scale = float(ctk.ScalingTracker.get_window_scaling(self))
            except Exception:
                ctk_scale = dpi_scale
            scale = ctk_scale if ctk_scale > 0 else dpi_scale
            # Map the outer root window first to measure native title-bar and border space.
            width = max(1, min(1220, int(work_width / scale)))
            height = max(1, min(640, int(work_height / scale)))
            self.geometry(f"{width}x{height}+0+0")
            self.update_idletasks()
            outer = wintypes.RECT()
            client = wintypes.RECT()
            user32.GetWindowRect(root_hwnd, ctypes.byref(outer))
            user32.GetClientRect(root_hwnd, ctypes.byref(client))
            frame_width = max(0, (outer.right - outer.left) - (client.right - client.left))
            frame_height = max(0, (outer.bottom - outer.top) - (client.bottom - client.top))

            margin_px = 16
            logical_width = max(1, int(max(1, work_width - frame_width - margin_px) / scale))
            logical_height = max(1, int(max(1, work_height - frame_height - margin_px) / scale))
            width = min(1220, logical_width)
            height = min(840, logical_height)
            min_width = min(1080, width)
            min_height = min(560, height)
            self.minsize(min_width, min_height)
            self.geometry(f"{width}x{height}+0+0")
            self.update_idletasks()
            user32.GetWindowRect(root_hwnd, ctypes.byref(outer))
            actual_width = outer.right - outer.left
            actual_height = outer.bottom - outer.top
            x = work.left + max(0, (work_width - actual_width) // 2)
            y = work.top + max(0, (work_height - actual_height) // 2)
            self.geometry(f"{width}x{height}+{x}+{y}")
            self.update_idletasks()
            return

        width, height = 1220, 840
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(width, screen_width)
        height = min(height, max(1, screen_height - 48))
        self.minsize(min(1080, width), min(560, height))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _set_icon(self) -> None:
        icon_path = resource_path("assets/app.ico")
        if sys.platform == "win32" and icon_path.is_file():
            try:
                self.iconbitmap(default=str(icon_path))
            except Exception:
                pass

    def _build_shell(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.rail = ctk.CTkFrame(self, width=216, corner_radius=0, fg_color=CANVAS)
        self.rail.grid(row=0, column=0, sticky="nsw", padx=(20, 10), pady=(20, 12))
        self.rail.grid_propagate(False)
        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color=CANVAS)
        self.content.grid(row=0, column=1, sticky="nsew", padx=(8, 20), pady=(20, 12))
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        logo_shell = ctk.CTkFrame(self.rail, fg_color=INK, corner_radius=12, height=72)
        logo_shell.pack(fill="x", pady=(0, 18))
        logo = ctk.CTkFrame(logo_shell, fg_color=GREEN, border_color=INK, border_width=2, corner_radius=10)
        logo.pack(fill="both", expand=True, padx=(0, 5), pady=(0, 5))
        ctk.CTkLabel(logo, text="语义分析", font=("Microsoft YaHei UI", 21, "bold"), text_color=INK).pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(logo, text="离线桌面工具包", font=SMALL_FONT, text_color=INK).pack(anchor="w", padx=13, pady=(0, 8))
        self.nav_group = ctk.CTkFrame(self.rail, fg_color="transparent")
        self.nav_group.pack(fill="x")
        for key, label in [("data", "数据 · 输入与清理"), ("methods", "方法 · 参数设置"),
                           ("results", "结果 · 矩阵与指标"), ("tasks", "任务 · 日志与项目"),
                           ("help", "帮助 · 口径与来源")]:
            holder = ctk.CTkFrame(self.nav_group, fg_color=INK, corner_radius=11, height=70)
            holder.pack(fill="x", pady=5)
            holder.pack_propagate(False)
            button = ctk.CTkButton(holder, text=label, command=lambda p=key: self.show_page(p),
                                   anchor="w", height=64, corner_radius=10, border_width=2,
                                   border_color=INK, font=("Microsoft YaHei UI", 14, "bold"),
                                   fg_color=WHITE, hover_color=self.PAGE_COLORS[key], text_color=INK)
            button.pack(fill="both", expand=True, padx=(0, 5), pady=(0, 5))
            self.nav_buttons[key] = button
            self._all_controls.append(button)
        ctk.CTkLabel(self.rail, text="数据保留在本机\n启动和分析无需联网", font=SMALL_FONT,
                     text_color=INK, justify="left").pack(anchor="w", side="bottom", pady=(0, 10))
        self.title_shell = ctk.CTkFrame(self.content, fg_color=INK, corner_radius=14, height=93)
        self.title_shell.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.title_shell.grid_propagate(False)
        self.title_card = ctk.CTkFrame(self.title_shell, fg_color=SKY, border_color=INK, border_width=2, corner_radius=12)
        self.title_card.pack(fill="both", expand=True, padx=(0, 6), pady=(0, 6))
        self.page_title = ctk.CTkLabel(self.title_card, text="", font=TITLE_FONT, text_color=INK, anchor="w")
        self.page_title.pack(anchor="w", padx=17, pady=(12, 0))
        self.page_subtitle = ctk.CTkLabel(self.title_card, text="", font=SMALL_FONT, text_color=INK, anchor="w")
        self.page_subtitle.pack(anchor="w", padx=19, pady=(1, 10))
        self.page_host = ctk.CTkFrame(self.content, fg_color="transparent")
        self.page_host.grid(row=1, column=0, sticky="nsew")
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)
        self.status_shell = ctk.CTkFrame(self, fg_color=INK, corner_radius=10, height=72)
        self.status_shell.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 14))
        self.status_shell.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(self.status_shell, text="就绪。可先导入文本，也可在帮助页查看口径。",
                                         text_color=WHITE, anchor="w", justify="left", wraplength=900, font=SMALL_FONT)
        self.status_label.grid(row=0, column=0, sticky="ew", padx=13, pady=7)
        self.progress = ctk.CTkProgressBar(self.status_shell, width=150, height=12, progress_color=GREEN)
        self.progress.set(0)
        self.progress.grid(row=0, column=1, padx=10, pady=10)
        self.cancel_button = ctk.CTkButton(self.status_shell, text="取消", width=70, height=30, fg_color=CORAL,
                                           hover_color="#f29b8e", text_color=INK, border_color=INK, border_width=1,
                                           command=self.cancel_work, state="disabled")
        self.cancel_button.grid(row=0, column=2, padx=(4, 9), pady=7)
        self._all_controls.append(self.cancel_button)

    def _scroll_page(self, key: str, color: str) -> ctk.CTkScrollableFrame:
        page = ctk.CTkScrollableFrame(self.page_host, fg_color=color, corner_radius=12,
                                     border_color=INK, border_width=2, scrollbar_button_color=INK,
                                     scrollbar_button_hover_color="#46516a")
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=1)
        self.page_widgets[key] = page
        return page

    def _card(self, parent: Any, fill: str = WHITE, *, padx: int = 14, pady: int = 10) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color=fill, border_color=INK, border_width=2, corner_radius=12)
        frame.pack(fill="x", padx=padx, pady=pady)
        return frame

    def _label(self, parent: Any, text: str, **kwargs: Any) -> ctk.CTkLabel:
        return ctk.CTkLabel(parent, text=text, font=kwargs.pop("font", BODY_FONT), text_color=INK, **kwargs)

    def _register_control(self, widget: Any) -> Any:
        self._all_controls.append(widget)
        return widget

    def _entry(self, parent: Any, var: Any, *, width: int = 130, placeholder: str = "") -> ctk.CTkEntry:
        return self._register_control(ctk.CTkEntry(parent, textvariable=var, width=width, height=36, fg_color=WHITE,
                                           border_color=INK, border_width=2, corner_radius=8,
                                           placeholder_text=placeholder, text_color=INK, font=BODY_FONT))

    def _combo(self, parent: Any, var: Any, values: list[str], *, width: int = 190) -> ctk.CTkComboBox:
        return self._register_control(ctk.CTkComboBox(parent, variable=var, values=values, width=width, height=36,
                                             fg_color=WHITE, border_color=INK, border_width=2,
                                             button_color=LAVENDER, button_hover_color=SKY,
                                             text_color=INK, dropdown_fg_color=WHITE,
                                             dropdown_text_color=INK, font=BODY_FONT))

    def _button(self, parent: Any, text: str, command: Callable[[], None], *, primary: bool = False,
                width: int = 150) -> ctk.CTkButton:
        color = GREEN if primary else LAVENDER
        holder = ctk.CTkFrame(parent, fg_color=INK, corner_radius=9, width=width, height=44)
        holder.pack_propagate(False)
        button = ctk.CTkButton(holder, text=text, command=command, height=39, corner_radius=8,
                               border_width=2, border_color=INK, fg_color=color, hover_color=SKY,
                               text_color=INK, font=("Microsoft YaHei UI", 13, "bold"))
        button.pack(fill="both", expand=True, padx=(0, 5), pady=(0, 5))
        self._register_control(button)
        return holder

    def _add_page_title(self, key: str) -> None:
        title, subtitle = self.PAGE_TITLES[key]
        self.title_card.configure(fg_color=self.PAGE_COLORS[key])
        self.page_title.configure(text=title)
        self.page_subtitle.configure(text=subtitle)

    def show_page(self, key: str) -> None:
        if key not in self.page_widgets:
            return
        for page in self.page_widgets.values():
            page.grid_remove()
        self.page_widgets[key].grid()
        self.current_page = key
        self._add_page_title(key)
        for name, button in self.nav_buttons.items():
            button.configure(fg_color=self.PAGE_COLORS[name] if name == key else WHITE,
                             border_width=3 if name == key else 2)

    def _build_data_page(self) -> None:
        page = self._scroll_page("data", self.PAGE_COLORS["data"])
        card = self._card(page, SKY, pady=8)
        self._label(card, "粘贴文本", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(10, 5))
        self.paste_box = self._register_control(ctk.CTkTextbox(card, height=115, fg_color=WHITE, text_color=INK,
                                                       border_color=INK, border_width=2, corner_radius=8,
                                                       font=BODY_FONT, wrap="word"))
        self.paste_box.pack(fill="x", padx=12, pady=(0, 8))
        paste_row = ctk.CTkFrame(card, fg_color="transparent")
        paste_row.pack(fill="x", padx=12, pady=(0, 10))
        self._label(paste_row, "粘贴文本切分").pack(side="left", padx=(0, 8))
        self.split_var = tk.StringVar(value="每段")
        self._combo(paste_row, self.split_var, ["整篇/每文件", "每行", "每段"], width=180).pack(side="left")
        self._label(paste_row, "（文件按同一方式切分）", font=SMALL_FONT).pack(side="left", padx=8)

        file_card = self._card(page, WHITE, pady=8)
        self._label(file_card, "文件与表格", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(10, 5))
        toolbar = ctk.CTkFrame(file_card, fg_color="transparent")
        toolbar.pack(fill="x", padx=12, pady=(0, 7))
        self._button(toolbar, "选择文件", self._browse_files, width=130).pack(side="left", padx=(0, 9))
        self._button(toolbar, "选择 TXT 目录", self._browse_directory, width=150).pack(side="left", padx=0)
        self.files_label = self._label(file_card, "尚未选择文件。支持 TXT、CSV、TSV、XLSX、JSONL。", font=SMALL_FONT, justify="left", anchor="w")
        self.files_label.pack(fill="x", padx=13, pady=(0, 8))
        self.selected_paths: list[str] = []

        columns = ctk.CTkFrame(file_card, fg_color="transparent")
        columns.pack(fill="x", padx=12, pady=(0, 6))
        self._label(columns, "文本列").grid(row=0, column=0, padx=(0, 7), pady=5, sticky="w")
        self.text_column_var = tk.StringVar(value="自动")
        self.text_column_combo = self._combo(columns, self.text_column_var, ["自动"], width=220)
        self.text_column_combo.grid(row=0, column=1, padx=(0, 18), pady=5, sticky="w")
        self._label(columns, "ID 列").grid(row=0, column=2, padx=(0, 7), pady=5, sticky="w")
        self.id_column_var = tk.StringVar(value="无")
        self.id_column_combo = self._combo(columns, self.id_column_var, ["无"], width=200)
        self.id_column_combo.grid(row=0, column=3, padx=(0, 18), pady=5, sticky="w")
        self.dedupe_var = tk.BooleanVar(value=False)
        self.dedupe_check = self._register_control(ctk.CTkCheckBox(columns, text="按空白归一后的完全相同文本去重", variable=self.dedupe_var,
                                                           text_color=INK, fg_color=GREEN, hover_color="#13a955",
                                                           border_color=INK, font=SMALL_FONT))
        self.dedupe_check.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        read_row = ctk.CTkFrame(file_card, fg_color="transparent")
        read_row.pack(fill="x", padx=12, pady=(5, 10))
        self.load_button = self._button(read_row, "读取并清理", self._load_clicked, primary=True, width=160)
        self.load_button.pack(side="left")
        self.input_summary = self._label(read_row, "尚未读取数据", font=SMALL_FONT, anchor="w", justify="left")
        self.input_summary.pack(side="left", padx=15, fill="x", expand=True)
        note = self._card(page, LEMON, pady=8)
        self._label(note, "导入规则", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(9, 4))
        self._label(note, "原文件只读；空文本自动排除。默认 ID 来自文件名与行/段号，表格可指定 ID 列。重复 ID 会加 #2、#3 保证唯一；去重仅在勾选时执行，并保留首条。目录读取其内 TXT 文件。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 10))

    def _build_method_page(self) -> None:
        page = self._scroll_page("methods", self.PAGE_COLORS["methods"])
        card = self._card(page, MINT, pady=8)
        self._label(card, "文档表示方法", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(10, 6))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 4))
        self.method_var = tk.StringVar(value="BGE-small-zh-v1.5（本地 ONNX）")
        self._label(row, "方法").pack(side="left", padx=(0, 8))
        self.method_combo = self._combo(row, self.method_var, list(METHODS.values()), width=340)
        self.method_combo.pack(side="left", padx=(0, 18))
        self.method_combo.configure(command=lambda _value: self._update_method_hint())
        self._label(row, "分词").pack(side="left", padx=(0, 8))
        self.segmentation_var = tk.StringVar(value="jieba精确模式")
        self._combo(row, self.segmentation_var, ["jieba精确模式", "jieba搜索模式", "字符"], width=180).pack(side="left", padx=(0, 18))
        self._label(row, "停词").pack(side="left", padx=(0, 8))
        self.stopwords_var = tk.StringVar(value="保守内置")
        self.stopwords_combo = self._combo(row, self.stopwords_var, ["无", "保守内置", "标准内置", "自定义文件"], width=160)
        self.stopwords_combo.pack(side="left")

        resource_row = ctk.CTkFrame(card, fg_color="transparent")
        resource_row.pack(fill="x", padx=12, pady=(8, 2))
        self.user_dictionary_var = tk.StringVar(value="")
        self.custom_stopwords_var = tk.StringVar(value="")
        self.word2vec_path_var = tk.StringVar(value="")
        dictionary_row = ctk.CTkFrame(resource_row, fg_color="transparent")
        dictionary_row.pack(fill="x", pady=(0, 4))
        self._label(dictionary_row, "自定义词典").pack(side="left", padx=(0, 8))
        self.user_dictionary_entry = self._entry(dictionary_row, self.user_dictionary_var, width=360, placeholder="可选 UTF-8 词典路径")
        self.user_dictionary_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.user_dict_button = self._register_control(ctk.CTkButton(dictionary_row, text="浏览…", width=72, height=34, command=self._browse_user_dictionary,
                                                               fg_color=LAVENDER, hover_color=SKY, border_color=INK, border_width=1, text_color=INK))
        self.user_dict_button.pack(side="left")
        stopword_row = ctk.CTkFrame(resource_row, fg_color="transparent")
        stopword_row.pack(fill="x", pady=(0, 4))
        self._label(stopword_row, "自定义停词").pack(side="left", padx=(0, 8))
        self.custom_stopwords_entry = self._entry(stopword_row, self.custom_stopwords_var, width=320, placeholder="每行一个词")
        self.custom_stopwords_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.custom_stopwords_button = self._register_control(ctk.CTkButton(stopword_row, text="浏览…", width=72, height=34, command=self._browse_stopwords,
                                                                     fg_color=LAVENDER, hover_color=SKY, border_color=INK, border_width=1, text_color=INK))
        self.custom_stopwords_button.pack(side="left")

        params = ctk.CTkFrame(card, fg_color="transparent")
        params.pack(fill="x", padx=12, pady=(4, 7))
        self.max_features_var = tk.StringVar(value="4000")
        self.min_frequency_var = tk.StringVar(value="1")
        self.topic_count_var = tk.StringVar(value="8")
        self.cooc_window_var = tk.StringVar(value="5")
        self.cooc_vocab_var = tk.StringVar(value="400")
        self.clusters_var = tk.StringVar(value="5")
        fields = [("词表上限（1–20000）", self.max_features_var), ("最小文档频次", self.min_frequency_var),
                  ("主题/LSA 维数", self.topic_count_var), ("共现窗口", self.cooc_window_var),
                  ("PPMI 独立词表上限（≤1000）", self.cooc_vocab_var), ("KMeans 簇数", self.clusters_var)]
        for i, (label, var) in enumerate(fields):
            row_no, col_no = divmod(i, 3)
            group = ctk.CTkFrame(params, fg_color="transparent")
            group.grid(row=row_no, column=col_no, sticky="ew", padx=(0, 16), pady=4)
            self._label(group, label, font=SMALL_FONT).pack(anchor="w", pady=(0, 3))
            self._entry(group, var, width=130).pack(anchor="w")
        for col in range(3):
            params.grid_columnconfigure(col, weight=1)

        self.word2vec_row = ctk.CTkFrame(card, fg_color="transparent")
        self.word2vec_row.pack(fill="x", padx=12, pady=(4, 2))
        self._label(self.word2vec_row, "外部 word2vec 文本文件", font=SMALL_FONT).pack(side="left", padx=(0, 8))
        self.word2vec_entry = self._entry(self.word2vec_row, self.word2vec_path_var, width=590, placeholder="仅在选择 word2vec 方法时需要；二进制 .bin 不支持")
        self.word2vec_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.word2vec_button = self._register_control(ctk.CTkButton(self.word2vec_row, text="浏览…", width=72, height=34, command=self._browse_word2vec,
                                                            fg_color=LAVENDER, hover_color=SKY, border_color=INK, border_width=1, text_color=INK))
        self.word2vec_button.pack(side="left")
        self.word2vec_tfidf_var = tk.BooleanVar(value=True)
        self.word2vec_check = self._register_control(ctk.CTkCheckBox(card, text="word2vec 词向量按文档 TF-IDF 权重平均", variable=self.word2vec_tfidf_var,
                                                             text_color=INK, fg_color=GREEN, hover_color="#13a955", border_color=INK,
                                                             font=SMALL_FONT))
        self.word2vec_check.pack(anchor="w", padx=14, pady=(5, 8))
        self.similarity_var = tk.BooleanVar(value=True)
        self.similarity_check = self._register_control(ctk.CTkCheckBox(card, text=f"计算完整相似度矩阵（最多 {1500:,} 篇文档；超出时仍计算分块近邻）",
                                                               variable=self.similarity_var, text_color=INK, fg_color=GREEN,
                                                               hover_color="#13a955", border_color=INK, font=SMALL_FONT))
        self.similarity_check.pack(anchor="w", padx=14, pady=(0, 8))
        self.compute_clusters_var = tk.BooleanVar(value=True)
        self.clustering_check = self._register_control(ctk.CTkCheckBox(
            card, text="计算 KMeans 聚类与轮廓系数（关闭后仍计算主题与近邻）", variable=self.compute_clusters_var,
            text_color=INK, fg_color=GREEN, hover_color="#13a955", border_color=INK, font=SMALL_FONT))
        self.clustering_check.pack(anchor="w", padx=14, pady=(0, 8))
        self.extra_bool_vars: dict[str, tk.BooleanVar] = {}
        for name, label in [("compute_neighbors", "计算近邻（关闭后跳过所有近邻文档对计算）"),
                            ("compute_silhouette", "计算轮廓系数（须同时开启聚类；可单独关闭以保留聚类标签）")]:
            variable = tk.BooleanVar(value=True)
            self.extra_bool_vars[name] = variable
            checkbox = self._register_control(ctk.CTkCheckBox(card, text=label, variable=variable,
                text_color=INK, fg_color=GREEN, hover_color="#13a955", border_color=INK, font=SMALL_FONT))
            checkbox.pack(anchor="w", padx=14, pady=(0, 8))
        self.method_hint = self._label(card, "BGE 使用内置本地模型，原文编码、CLS 池化、L2 归一化；启动无需联网。", font=SMALL_FONT,
                                       anchor="w", justify="left", wraplength=860)
        self.method_hint.pack(fill="x", padx=14, pady=(0, 8))
        run_row = ctk.CTkFrame(card, fg_color="transparent")
        run_row.pack(fill="x", padx=12, pady=(4, 11))
        self.run_button = self._button(run_row, "开始分析", self._run_clicked, primary=True, width=160)
        self.run_button.pack(side="left")
        self._label(run_row, "向量与词项不代表概率或准确率；计算口径和来源见帮助页。", font=SMALL_FONT).pack(side="left", padx=15)
        batch_card = self._card(page, WHITE, pady=7)
        self._label(batch_card, "同一语料依次运行多种方法（方法失败会记录并继续；取消会停止后续方法）",
                    font=SMALL_FONT, anchor="w", justify="left").pack(fill="x", padx=12, pady=(8, 3))
        batch_grid = ctk.CTkFrame(batch_card, fg_color="transparent")
        batch_grid.pack(fill="x", padx=10, pady=(0, 8))
        self.batch_vars: dict[str, tk.BooleanVar] = {}
        for index, (method, label) in enumerate(METHODS.items()):
            var = tk.BooleanVar(value=False)
            self.batch_vars[method] = var
            checkbox = self._register_control(ctk.CTkCheckBox(batch_grid, text=label, variable=var,
                                                               text_color=INK, fg_color=GREEN,
                                                               hover_color="#13a955", border_color=INK,
                                                               font=SMALL_FONT))
            checkbox.grid(row=index // 4, column=index % 4, sticky="w", padx=8, pady=4)
        for col in range(4):
            batch_grid.grid_columnconfigure(col, weight=1)
        advanced = self._card(page, WHITE, pady=7)
        self._label(advanced, "词项与计算参数", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(9, 4))
        advanced_grid = ctk.CTkFrame(advanced, fg_color="transparent")
        advanced_grid.pack(fill="x", padx=12, pady=4)
        defaults = AnalysisConfig()
        self.extra_parameter_vars: dict[str, tk.StringVar] = {}
        extra_fields = [("max_df", "最大文档比例（0 < 比例 ≤ 1）"),
                        ("neighbors_k", "近邻 Top-K（1–100）"), ("random_state", "随机种子（0–4294967295）"),
                        ("word_ngram_min", "词级 n-gram 最小值（1–3）"), ("word_ngram_max", "词级 n-gram 最大值（1–3）"),
                        ("nmf_max_iter", "NMF 最大迭代次数（1–1000）"),
                        ("char_ngram_min", "字符 n-gram 最小值（1–5）"), ("char_ngram_max", "字符 n-gram 最大值（1–5）"),
                        ("lda_max_iter", "LDA 最大迭代次数（1–1000）")]
        for index, (name, label) in enumerate(extra_fields):
            variable = tk.StringVar(value=str(getattr(defaults, name)))
            self.extra_parameter_vars[name] = variable
            group = ctk.CTkFrame(advanced_grid, fg_color="transparent")
            group.grid(row=index // 3, column=index % 3, sticky="ew", padx=(0, 16), pady=4)
            self._label(group, label, font=SMALL_FONT).pack(anchor="w", pady=(0, 3))
            self._entry(group, variable, width=150).pack(anchor="w")
        for col in range(3):
            advanced_grid.grid_columnconfigure(col, weight=1)
        self._label(advanced, "最大文档比例按出现文档数筛词，1 表示不过滤；n-gram 最小值须不大于最大值。词级范围用于词 TF-IDF / LSA / NMF / LDA；字符范围用于字符 TF-IDF。近邻不足 K 条时返回全部有效近邻。",
                    font=SMALL_FONT, justify="left", anchor="w", wraplength=850).pack(fill="x", padx=12, pady=(0, 10))
        self.parameter_sections = {"词项与计算参数": advanced}

        def parameter_card(title: str, fields: list[tuple[str, str]], note: str) -> Any:
            host = self._card(page, WHITE, pady=7)
            self.parameter_sections[title] = host
            self._label(host, title, font=SECTION_FONT).pack(anchor="w", padx=12, pady=(9, 4))
            grid = ctk.CTkFrame(host, fg_color="transparent")
            if fields:
                grid.pack(fill="x", padx=12, pady=4)
            for index, (name, label) in enumerate(fields):
                value = getattr(defaults, name)
                variable = tk.StringVar(value="" if value is None else str(value))
                self.extra_parameter_vars[name] = variable
                group = ctk.CTkFrame(grid, fg_color="transparent")
                group.grid(row=index // 3, column=index % 3, sticky="ew", padx=(0, 16), pady=4)
                self._label(group, label, font=SMALL_FONT).pack(anchor="w", pady=(0, 3))
                self._entry(group, variable, width=150).pack(anchor="w")
            for col in range(3):
                grid.grid_columnconfigure(col, weight=1)
            self._label(host, note, font=SMALL_FONT, justify="left", anchor="w", wraplength=850).pack(fill="x", padx=12, pady=(0, 8))
            return host

        cleaning = parameter_card("文本清理", [("min_token_length", "最小词长（1–20 字符）")],
            "保留原文供查看。最小词长和纯数字过滤仅作用于分词；网址移除和统一小写作用于分析文本，也影响字符 TF-IDF / BGE。方法默认：字符 TF-IDF 转小写，其他方法保留大小写。")
        case_row = ctk.CTkFrame(cleaning, fg_color="transparent")
        case_row.pack(fill="x", padx=12, pady=(0, 8))
        self._label(case_row, "英文大小写", font=SMALL_FONT).pack(side="left", padx=(0, 8))
        self.case_mode_var = tk.StringVar(value=defaults.case_mode)
        self._combo(case_row, self.case_mode_var, ["方法默认", "统一小写", "保持原样"], width=160).pack(side="left")
        tfidf = parameter_card("TF-IDF 权重", [],
            "对数词频使用 1 + ln(tf)；IDF 平滑使用加一的文档频次公式。适用于词级 / 字符 TF-IDF 及相关方法；LDA 始终使用原始词项计数。")
        for host, name, label in [(cleaning, "remove_numbers", "过滤纯数字 token（保留含字母或汉字的编号）"),
                                  (cleaning, "remove_urls", "从分析文本中移除 http(s) / www 网址"),
                                  (tfidf, "tfidf_sublinear_tf", "使用对数词频（sublinear_tf）"),
                                  (tfidf, "tfidf_smooth_idf", "启用 IDF 平滑（smooth_idf）")]:
            variable = tk.BooleanVar(value=getattr(defaults, name))
            self.extra_bool_vars[name] = variable
            checkbox = self._register_control(ctk.CTkCheckBox(host, text=label, variable=variable,
                text_color=INK, fg_color=GREEN, hover_color="#13a955", border_color=INK, font=SMALL_FONT))
            checkbox.pack(anchor="w", padx=14, pady=(0, 8))
        parameter_card("主题与聚类展示", [("topic_top_words", "主题词表每主题词数（1–100）"),
            ("profile_top_words", "画像关键词数量（1–100）"), ("representative_count", "每组代表文本数（1–50）")],
            "主题词表、主题 / 聚类画像、代表文档 ID 导出和详情摘要使用各自设置；不足指定数量时显示全部可用项。")
        parameter_card("高级模型设置", [("nmf_tol", "NMF 收敛阈值（正数）"), ("kmeans_tol", "KMeans 收敛阈值（正数）"),
            ("lda_mean_change_tol", "LDA 文档更新阈值（正数）"), ("lda_perp_tol", "LDA 困惑度阈值（正数）"),
            ("lda_evaluate_every", "LDA 评估间隔（-1 或 1–100）"),
            ("lda_doc_topic_prior", "LDA α 文档—主题先验（空=自动）"),
            ("lda_topic_word_prior", "LDA η 主题—词先验（空=自动）")],
            "先验填有限正数或留空使用 1 / 主题数。LDA 评估间隔 -1 表示关闭困惑度评估，此时困惑度阈值不参与停止判断；开启评估会增加耗时。文档更新阈值控制 LDA 内部文档迭代。")
        details = self._card(page, LEMON, pady=8)
        self._label(details, "方法范围", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(9, 4))
        self._label(details, "词 TF-IDF、字符 TF-IDF、LSA、NMF、LDA（计数输入）、窗口共现/PPMI、外部文本 word2vec，以及本地 BGE-small-zh-v1.5 ONNX。BGE 超长文本按 510 个正文 token 分块、按块 token 数加权平均 CLS 后 L2 归一化；该聚合是长文本近似。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 10))
        self._update_method_hint()

    def _build_results_page(self) -> None:
        page = self._scroll_page("results", self.PAGE_COLORS["results"])
        card = self._card(page, WHITE, pady=8)
        toolbar = ctk.CTkFrame(card, fg_color="transparent")
        toolbar.pack(fill="x", padx=12, pady=(9, 2))
        self._label(toolbar, "表格支持筛选和完整分页；双击文档／近邻或按 Enter 查看原文。", font=SMALL_FONT).pack(anchor="w", pady=(0, 3))
        result_select_row = ctk.CTkFrame(card, fg_color="transparent")
        result_select_row.pack(fill="x", padx=12, pady=(0, 5))
        self._label(result_select_row, "当前结果").pack(side="left", padx=(0, 8))
        self.result_method_var = tk.StringVar(value="")
        self.result_method_combo = self._combo(result_select_row, self.result_method_var, [], width=220)
        self.result_method_combo.pack(side="left")
        self.result_method_combo.configure(command=lambda _value: self._select_result_method())
        export_row = result_select_row
        self.xlsx_var = tk.BooleanVar(value=False)
        self.xlsx_check = self._register_control(ctk.CTkCheckBox(export_row, text="同时生成 XLSX", variable=self.xlsx_var,
                                                          text_color=INK, fg_color=GREEN, hover_color="#13a955",
                                                          border_color=INK, font=SMALL_FONT))
        self.xlsx_check.pack(side="right", padx=(8, 12))
        self.export_button = self._button(export_row, "导出选定结果", self._export_clicked, primary=True, width=130)
        self.export_button.pack(side="right")
        self.export_button.winfo_children()[0].configure(state="disabled")
        self.export_all_button = self._button(export_row, "导出全部与比较表", self._export_all_clicked, width=160)
        self.export_all_button.pack(side="right", padx=(0, 8))
        self.export_all_button.winfo_children()[0].configure(state="disabled")

        self.results_tab = ctk.CTkTabview(card, fg_color=WHITE, segmented_button_fg_color=LAVENDER,
                                           segmented_button_selected_color=GREEN,
                                           segmented_button_selected_hover_color="#13a955",
                                           segmented_button_unselected_color=WHITE,
                                           segmented_button_unselected_hover_color=SKY,
                                           text_color=INK, border_color=INK, border_width=2, corner_radius=10)
        self.results_tab.pack(fill="both", expand=True, padx=11, pady=(4, 11))
        for name in ["概览", "文档指标", "词频", "近邻", "主题与聚类", "相似度热图"]:
            self.results_tab.add(name)
        self.overview_box = ctk.CTkTextbox(self.results_tab.tab("概览"), height=360, fg_color=WHITE, text_color=INK,
                                           border_color=INK, border_width=1, font=BODY_FONT, wrap="word")
        self.overview_box.pack(fill="both", expand=True, padx=7, pady=7)
        self._set_readonly_text(self.overview_box, "尚未运行分析。\n\n先在“数据”页导入文本，再到“方法”页选择算法并开始计算。")
        self.explorer = ResultsExplorer(self, self.results_tab, ink=INK, white=WHITE,
                                       font=BODY_FONT, small_font=SMALL_FONT)
        self.heatmap_frame = self.results_tab.tab("相似度热图")
        self.heatmap_label = self._label(self.heatmap_frame, "尚无相似度矩阵", font=SMALL_FONT, anchor="nw", justify="left")
        self.heatmap_label.pack(fill="x", padx=8, pady=8)
        self.heatmap_canvas: tk.Canvas | None = None

    def _make_tree(self, parent: Any) -> ttk.Treeview:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Toolkit.Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK,
                        rowheight=28, bordercolor=INK, borderwidth=1, font=("Microsoft YaHei UI", 10))
        style.configure("Toolkit.Treeview.Heading", background=LAVENDER, foreground=INK,
                        font=("Microsoft YaHei UI", 10, "bold"), relief="flat")
        style.map("Toolkit.Treeview", background=[("selected", GREEN)], foreground=[("selected", INK)])
        return ttk.Treeview(parent, show="headings", style="Toolkit.Treeview")

    def _build_help_page(self) -> None:
        page = self._scroll_page("help", self.PAGE_COLORS["help"])
        card = self._card(page, WHITE, pady=8)
        self._label(card, "本地使用", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(10, 4))
        self._label(card, "程序启动不请求网络；默认 BGE 模型已随程序打包。模型资源、分词器、参数和来源保存在离线导出中。大规模结果以 NPZ 保存，界面表格和热图只显示有限预览。完整相似度矩阵最多 1,500 篇文档；超过时保留分块近邻。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 10))
        links = self._card(page, LAVENDER, pady=8)
        self._label(links, "帮助资料", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(10, 5))
        buttons = ctk.CTkFrame(links, fg_color="transparent")
        buttons.pack(fill="x", padx=12, pady=(0, 8))
        self._button(buttons, "计算口径与方法", lambda: self._show_help_resource("METHOD_NOTES.md"), width=180).pack(side="left", padx=(0, 8))
        self._button(buttons, "来源文献", lambda: self._show_help_resource("references.json"), width=130).pack(side="left", padx=0)
        self._button(buttons, "第三方许可", lambda: self._show_help_resource("THIRD_PARTY_NOTICES.txt"), width=140).pack(side="left", padx=8)
        self._label(links, "点击文献 URL 可打开系统浏览器；使用本地方法无需浏览。许可文件可能较长，已按需加载。", font=SMALL_FONT,
                    anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 9))
        limits = self._card(page, LEMON, pady=8)
        self._label(limits, "解释边界", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(10, 4))
        self._label(limits, "相似度是指定表示空间中的余弦值，不是概率或准确率；聚类是固定随机种子的探索性结果。轮廓系数只在样本和簇数满足条件时计算，超过 2,000 个非零表示时使用固定种子抽样。LSA 文档数或词项不足时会明确拒绝。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 10))
        self._label(limits, "阶段和全任务剩余时间会先按方法、规模与参数给出初始估计，再按本次工作量或成功历史校准；阶段参考范围是工程参考，不是统计置信区间。拟合等无法回调的步骤会保留真实阶段，超出初始时长后持续更新剩余估计。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 10))

    def _build_tasks_page(self) -> None:
        page = self._scroll_page("tasks", self.PAGE_COLORS["tasks"])
        log_card = self._card(page, WHITE, pady=7)
        self._label(log_card, "运行日志", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(8, 3))
        self._label(log_card, "日志保存在本机，按大小轮换；普通运行日志不记录粘贴文本。可筛选、复制或导出完整内容。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 5))
        log_buttons = ctk.CTkFrame(log_card, fg_color="transparent")
        log_buttons.pack(fill="x", padx=10, pady=(0, 7))
        self._button(log_buttons, "查看 / 筛选日志", self._open_log_viewer, width=185).pack(side="left", padx=(0, 8))
        self._button(log_buttons, "导出全部日志", self._export_logs, width=165).pack(side="left", padx=(0, 8))
        self._button(log_buttons, "刷新任务历史", self._refresh_task_history, width=170).pack(side="left")

        project_card = self._card(page, SKY, pady=7)
        self._label(project_card, "项目与恢复", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(8, 3))
        self._label(project_card, "项目用 JSON + NPZ 保存语料、清洗记录、设置与已完成矩阵；打开已有项目后可直接查看和导出结果。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=850).pack(fill="x", padx=12, pady=(0, 5))
        project_buttons = ctk.CTkFrame(project_card, fg_color="transparent")
        project_buttons.pack(fill="x", padx=10, pady=(0, 7))
        self._button(project_buttons, "保存当前项目", self._save_project_clicked, primary=True, width=170).pack(side="left", padx=(0, 8))
        self._button(project_buttons, "打开已保存项目", self._open_project_clicked, width=190).pack(side="left", padx=(0, 8))
        self._button(project_buttons, "打开历史结果", self._open_history_clicked, width=175).pack(side="left")
        self.project_state_label = self._label(project_card, "尚未打开或保存项目。", font=SMALL_FONT, anchor="w", justify="left")
        self.project_state_label.pack(fill="x", padx=12, pady=(0, 8))

        history_card = self._card(page, WHITE, pady=7)
        self._label(history_card, "任务历史", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(8, 4))
        self.task_history_tree = self._make_tree(history_card)
        self.task_history_tree.configure(columns=("started", "status", "methods", "summary"))
        for col, title, width in [("started", "开始时间", 170), ("status", "状态", 90), ("methods", "方法", 250), ("summary", "结果位置 / 说明", 430)]:
            self.task_history_tree.heading(col, text=title)
            self.task_history_tree.column(col, width=width, minwidth=75, stretch=True)
        self.task_history_tree.pack(fill="x", padx=10, pady=(0, 8))
        self.task_history_tree.bind("<Double-1>", lambda _event: self._open_history_clicked())

        preset_card = self._card(page, LAVENDER, pady=7)
        self._label(preset_card, "参数预设与上次设置", font=SECTION_FONT).pack(anchor="w", padx=12, pady=(8, 4))
        preset_row = ctk.CTkFrame(preset_card, fg_color="transparent")
        preset_row.pack(fill="x", padx=10, pady=(0, 7))
        self.preset_name_var = tk.StringVar(value="")
        self.preset_picker_var = tk.StringVar(value="")
        self._entry(preset_row, self.preset_name_var, width=180, placeholder="命名预设").pack(side="left", padx=(0, 8))
        self.preset_combo = self._combo(preset_row, self.preset_picker_var, [], width=190)
        self.preset_combo.pack(side="left", padx=(0, 8))
        self._button(preset_row, "保存预设", self._save_preset_clicked, width=135).pack(side="left", padx=(0, 8))
        self._button(preset_row, "加载预设", self._load_preset_clicked, width=135).pack(side="left", padx=(0, 8))
        self._button(preset_row, "删除预设", self._delete_preset_clicked, width=135).pack(side="left")

        compare_card = self._card(page, MINT, pady=7)
        compare_row = ctk.CTkFrame(compare_card, fg_color="transparent")
        compare_row.pack(fill="x", padx=12, pady=(8, 4))
        self._label(compare_row, "当前结果比较（适用的聚类指标单独列出，不做统一准确率排名）",
                    font=SECTION_FONT, anchor="w").pack(side="left")
        self.comparison_tree = self._make_tree(compare_card)
        self.comparison_tree.configure(columns=("method", "elapsed", "dimension", "documents", "silhouette"))
        for col, title, width in [("method", "方法", 250), ("elapsed", "分析耗时", 110),
                                  ("dimension", "向量维数", 100), ("documents", "有效文档", 100),
                                  ("silhouette", "轮廓系数（如适用）", 170)]:
            self.comparison_tree.heading(col, text=title)
            self.comparison_tree.column(col, width=width, minwidth=80, stretch=True)
        self.comparison_tree.pack(fill="x", padx=10, pady=(0, 8))
        self._refresh_task_history()
        self._refresh_preset_names()

    def _fit_dialog(self, window: Any, width: int, height: int) -> None:
        """Cap popups to the active monitor work area after CTk scaling."""
        try:
            scale = float(ctk.ScalingTracker.get_window_scaling(self))
        except Exception:
            scale = 1.0
        scale = max(0.75, scale)
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                class MonitorInfo(ctypes.Structure):
                    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]
                user32 = ctypes.windll.user32
                parent_hwnd = user32.GetAncestor(int(self.winfo_id()), 2) or int(self.winfo_id())
                monitor = user32.MonitorFromWindow(parent_hwnd, 2)
                info = MonitorInfo()
                info.cbSize = ctypes.sizeof(info)
                if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
                    work_w = info.rcWork.right - info.rcWork.left
                    work_h = info.rcWork.bottom - info.rcWork.top
                    width = min(width, max(420, int((work_w - 56) / scale)))
                    height = min(height, max(360, int((work_h - 96) / scale)))
            except Exception:
                pass
        else:
            width = min(width, max(420, self.winfo_screenwidth() - 40))
            height = min(height, max(360, self.winfo_screenheight() - 100))
        window.geometry(f"{width}x{height}")
        window.transient(self)

    def _show_help_resource(self, filename: str) -> None:
        path = resource_path(filename)
        if not path.is_file():
            messagebox.showerror("帮助资源缺失", f"程序资源中找不到：{filename}", parent=self)
            return
        if filename == "references.json":
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                body = self._format_references(raw)
            except Exception as exc:
                body = f"文献资料读取失败：{exc}"
        else:
            body = path.read_text(encoding="utf-8", errors="replace")
        window = ctk.CTkToplevel(self)
        window.title({"METHOD_NOTES.md": "计算口径与方法", "references.json": "来源文献", "THIRD_PARTY_NOTICES.txt": "第三方许可"}.get(filename, filename))
        self._fit_dialog(window, 900, 700)
        window.configure(fg_color=CANVAS)
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        text = tk.Text(window, wrap="word", bg=WHITE, fg=INK, relief="solid", bd=2,
                       font=("Microsoft YaHei UI", 11), padx=12, pady=10)
        yscroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.grid(row=0, column=0, sticky="nsew", padx=(14, 0), pady=14)
        yscroll.grid(row=0, column=1, sticky="ns", padx=(0, 14), pady=14)
        text.insert("1.0", body)
        text.tag_configure("url", foreground="#0969a2", underline=True)
        for match in __import__("re").finditer(r"https?://[^\s<>\]\)]+", body):
            start = f"1.0+{match.start()}c"
            end = f"1.0+{match.end()}c"
            text.tag_add("url", start, end)
        def open_clicked_link(event: Any) -> None:
            index = text.index(f"@{event.x},{event.y}")
            url = text.get(f"{index} wordstart", f"{index} wordend")
            if url.startswith(("http://", "https://")):
                webbrowser.open(url.rstrip(".,;"))
        text.tag_bind("url", "<Button-1>", open_clicked_link)
        text.configure(state="disabled")

    def _format_references(self, value: Any) -> str:
        if not isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, indent=2)
        lines = ["来源文献与方法资料\n", f"核对日期：{value.get('checked_on', '未记录')}", f"范围：{value.get('scope', '未记录')}", ""]
        for idx, row in enumerate(value.get("sources", []), 1):
            lines.extend([f"{idx}. {row.get('title', '未命名')}",
                          f"   作者 / 年份：{row.get('authors', '')} / {row.get('year', '')}",
                          f"   URL：{row.get('url', '')}",
                          f"   方法：{row.get('method', '')}",
                          f"   发现：{row.get('finding', '')}",
                          f"   应用：{row.get('application', '')}",
                          f"   限制：{row.get('limitation', '')}", ""])
        return "\n".join(lines)

    def _settings_snapshot(self) -> dict[str, Any]:
        return {
            "method": next((key for key, label in METHODS.items() if label == self.method_var.get()), "bge"),
            "segmentation": self.segmentation_var.get(), "stopwords": self.stopwords_var.get(),
            "custom_stopwords_path": self.custom_stopwords_var.get().strip(),
            "user_dictionary_path": self.user_dictionary_var.get().strip(),
            "word2vec_path": self.word2vec_path_var.get().strip(),
            "word2vec_tfidf_weight": bool(self.word2vec_tfidf_var.get()),
            "compute_similarity": bool(self.similarity_var.get()),
            "compute_clusters": bool(self.compute_clusters_var.get()),
            **{key: variable.get() for key, variable in self.extra_parameter_vars.items()},
            **{key: bool(variable.get()) for key, variable in self.extra_bool_vars.items()},
            "case_mode": self.case_mode_var.get(),
            "max_features": self.max_features_var.get(), "min_frequency": self.min_frequency_var.get(),
            "topic_count": self.topic_count_var.get(), "cooc_window": self.cooc_window_var.get(),
            "cooc_vocab_size": self.cooc_vocab_var.get(), "clusters": self.clusters_var.get(),
            "split_mode": self.split_var.get(), "text_column": self.text_column_var.get(),
            "id_column": self.id_column_var.get(), "deduplicate": bool(self.dedupe_var.get()),
            "selected_paths": list(self.selected_paths),
            "batch_methods": [method for method, var in self.batch_vars.items() if var.get()],
            "xlsx": bool(self.xlsx_var.get()),
        }

    def _apply_settings(self, settings: dict[str, Any]) -> str:
        if not isinstance(settings, dict):
            return ""
        method = settings.get("method")
        if method in METHODS:
            self.method_var.set(METHODS[method])
        for key, variable in [("segmentation", self.segmentation_var), ("stopwords", self.stopwords_var),
                              ("custom_stopwords_path", self.custom_stopwords_var), ("user_dictionary_path", self.user_dictionary_var),
                              ("word2vec_path", self.word2vec_path_var), ("max_features", self.max_features_var),
                              ("min_frequency", self.min_frequency_var), ("topic_count", self.topic_count_var),
                              ("cooc_window", self.cooc_window_var), ("cooc_vocab_size", self.cooc_vocab_var),
                              ("clusters", self.clusters_var), ("split_mode", self.split_var),
                              ("text_column", self.text_column_var), ("id_column", self.id_column_var)]:
            if key in settings:
                variable.set(str(settings[key]))
        if "deduplicate" in settings:
            self.dedupe_var.set(bool(settings["deduplicate"]))
        if "word2vec_tfidf_weight" in settings:
            self.word2vec_tfidf_var.set(bool(settings["word2vec_tfidf_weight"]))
        if "compute_similarity" in settings:
            self.similarity_var.set(bool(settings["compute_similarity"]))
        self.compute_clusters_var.set(bool(settings.get("compute_clusters", True)))
        defaults = AnalysisConfig()
        for key, variable in self.extra_parameter_vars.items():
            value = settings.get(key, getattr(defaults, key))
            variable.set("" if value is None else str(value))
        for key, variable in self.extra_bool_vars.items():
            variable.set(bool(settings.get(key, getattr(defaults, key))))
        self.case_mode_var.set(str(settings.get("case_mode", defaults.case_mode)))
        if "xlsx" in settings:
            self.xlsx_var.set(bool(settings["xlsx"]))
        batch_methods = set(settings.get("batch_methods", []))
        for name, variable in self.batch_vars.items():
            variable.set(name in batch_methods)
        paths = [str(path) for path in settings.get("selected_paths", []) if isinstance(path, str)]
        self.selected_paths = paths
        existing = [path for path in paths if Path(path).exists()]
        if existing:
            names = [Path(path).name for path in paths]
            self.files_label.configure(text="\n".join(names[:8]) + (f"\n… 共 {len(names)} 个来源" if len(names) > 8 else ""))
        missing = [path for path in paths if not Path(path).exists()]
        missing.extend(path for path in [self.user_dictionary_var.get(), self.custom_stopwords_var.get(), self.word2vec_path_var.get()]
                       if path and not Path(path).exists())
        self._update_method_hint()
        return "；".join(dict.fromkeys(missing))

    def _restore_last_settings(self) -> None:
        missing = self._apply_settings(self.runtime.load_last_settings())
        if missing:
            self._set_status(f"已恢复上次设置；以下输入资源当前不可用，请在方法或数据页重新选择：\n{missing}")

    def _refresh_preset_names(self) -> None:
        names = sorted(self.runtime.presets())
        self.preset_combo.configure(values=names)
        if self.preset_picker_var.get() not in names:
            self.preset_picker_var.set(names[0] if names else "")

    def _save_preset_clicked(self) -> None:
        name = self.preset_name_var.get().strip()
        if not name:
            messagebox.showinfo("需要名称", "请先输入预设名称。", parent=self)
            return
        try:
            self.runtime.save_preset(name, self._settings_snapshot())
            self._refresh_preset_names()
            self.preset_picker_var.set(name)
            self._set_status(f"已保存参数预设“{name}”。预设不包含粘贴文本。")
        except Exception as exc:
            messagebox.showerror("保存预设失败", str(exc), parent=self)

    def _load_preset_clicked(self) -> None:
        name = self.preset_picker_var.get()
        settings = self.runtime.presets().get(name)
        if not settings:
            messagebox.showinfo("没有预设", "请选择已有的命名预设。", parent=self)
            return
        missing = self._apply_settings(settings)
        self.runtime.save_last_settings(self._settings_snapshot())
        self._set_status(f"已加载预设“{name}”。" + (f"\n资源缺失：{missing}" if missing else ""))

    def _delete_preset_clicked(self) -> None:
        name = self.preset_picker_var.get()
        if not name:
            return
        self.runtime.delete_preset(name)
        self._refresh_preset_names()

    def _refresh_task_history(self) -> None:
        tree = getattr(self, "task_history_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        for row in self.runtime.history()[:60]:
            methods = ", ".join(row.get("methods", []))
            result_dirs = row.get("result_directories", {})
            summary = row.get("summary", "") or ", ".join(str(x) for x in result_dirs.values())
            tree.insert("", "end", iid=str(row.get("task_id", uuid.uuid4().hex)),
                        values=(row.get("started_at", ""), row.get("status", ""), methods, summary))

    def _open_log_viewer(self) -> None:
        window = ctk.CTkToplevel(self)
        window.title("中文语义工具包 · 运行日志")
        self.log_window = window
        self._fit_dialog(window, 860, 590)
        window.configure(fg_color=CANVAS)
        window.grid_rowconfigure(1, weight=1)
        window.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(window, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=10)
        filter_var = tk.StringVar(value="")
        self._label(top, "筛选").pack(side="left", padx=(0, 6))
        self._entry(top, filter_var, width=250, placeholder="任务 ID、阶段、警告或错误").pack(side="left", padx=(0, 8))
        text = tk.Text(window, wrap="char", height=14, bg=WHITE, fg=INK, relief="solid", bd=2,
                       font=("Consolas", 10), padx=8, pady=7)
        yscroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 6))
        yscroll.grid(row=1, column=1, sticky="ns", padx=(0, 12), pady=(0, 6))
        buttons = ctk.CTkFrame(window, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=9)

        def render(*_args: Any) -> None:
            lines = self.runtime.read_logs(filter_var.get())
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", "\n".join(lines))
            text.configure(state="disabled")
            text.see("end")

        def copy_log() -> None:
            value = text.get("1.0", "end-1c")
            window.clipboard_clear()
            window.clipboard_append(value)
            self._set_status("已复制当前筛选后的完整日志。")

        def export_log() -> None:
            chosen = filedialog.asksaveasfilename(parent=window, title="导出筛选后的完整日志",
                                                 defaultextension=".jsonl", filetypes=[("JSONL", "*.jsonl"), ("文本", "*.txt")])
            if chosen:
                self.runtime.export_logs(chosen, filter_var.get())
                messagebox.showinfo("日志已导出", f"日志已写入：\n{chosen}", parent=window)

        self._button(buttons, "刷新", render, width=110).pack(side="left", padx=(0, 8))
        self._button(buttons, "复制当前内容", copy_log, width=150).pack(side="left", padx=(0, 8))
        self._button(buttons, "导出筛选结果", export_log, width=150).pack(side="left")
        filter_var.trace_add("write", render)
        render()

    def start_log_viewer_smoke(self, output_dir: str | Path) -> None:
        """Capture only the log popup with isolated synthetic runtime records."""
        self.log_smoke_errors: list[str] = []
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        run_id = "SYNTHETIC_LOG_VIEWER_SMOKE"
        rows = [
            {"type": "task-start", "label": "合成日志弹窗检查", "method": "SYNTHETIC", "status": "running"},
            {"type": "stage-start", "stage_id": "term-matrix", "label": "合成阶段日志", "method": "SYNTHETIC"},
            {"type": "warning", "warning": "合成警告：仅用于日志窗口布局检查。", "method": "SYNTHETIC"},
            {"type": "error", "error_type": "SyntheticSmoke", "details": "合成错误详情与 traceback 占位；不含用户语料。", "method": "SYNTHETIC"},
            {"type": "terminal", "status": "success", "message_code": "synthetic-smoke"},
        ]
        for sequence, row in enumerate(rows, 1):
            self.runtime.append_log({"run_id": run_id, "task_id": run_id, "seq": sequence, **row}, force=True)

        def show_viewer() -> None:
            try:
                self._open_log_viewer()
                self.after(700, capture_viewer)
            except Exception as exc:
                self.log_smoke_errors.append(f"{type(exc).__name__}: {exc}")
                self.destroy()

        def capture_viewer() -> None:
            try:
                self.update_idletasks()
                window = self.log_window
                window.update_idletasks()
                from PIL import ImageGrab
                ImageGrab.grab(window=int(window.winfo_id())).save(output / "10-log-viewer.png")
                (output / "ui-log-smoke.json").write_text(json.dumps({
                    "status": "PASS", "geometry": window.winfo_geometry(),
                    "requested_size": [window.winfo_reqwidth(), window.winfo_reqheight()],
                    "dpi_scale": ctk.ScalingTracker.get_window_scaling(self),
                    "screenshot": str(output / "10-log-viewer.png"),
                    "profile": str(self.runtime.root),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                self.log_smoke_errors.append(f"{type(exc).__name__}: {exc}")
                (output / "ui-log-smoke-errors.txt").write_text("\n".join(self.log_smoke_errors), encoding="utf-8")
            finally:
                self.destroy()

        self.after(250, show_viewer)

    def _export_logs(self) -> None:
        chosen = filedialog.asksaveasfilename(parent=self, title="导出完整运行日志", defaultextension=".jsonl",
                                              filetypes=[("JSONL", "*.jsonl"), ("文本", "*.txt")])
        if chosen:
            self.runtime.export_logs(chosen)
            self._set_status(f"完整运行日志已导出：{chosen}")

    def _emit_for_run(self, run_id: str, sequence: list[int], emitter: Callable[[dict[str, Any]], None],
                      event: dict[str, Any]) -> None:
        sequence[0] += 1
        payload = dict(event)
        payload.update({"run_id": run_id, "task_id": run_id, "seq": sequence[0]})
        if not payload.get("timestamp"):
            from datetime import datetime
            payload["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
        if payload.get("monotonic") is None:
            payload["monotonic"] = time.monotonic()
        self.runtime.append_log(payload, force=payload.get("type") in {
            "stage-start", "stage-complete", "stage-failed", "terminal", "method-terminal", "warning", "error"})
        emitter(payload)

    def _start_runtime_job(self, label: str, task_kind: str, methods: list[str],
                           plan: list[StageEstimate], config_record: dict[str, Any],
                           job: Callable[[str, Event, Callable[[dict[str, Any]], None]], Any],
                           on_finish: Callable[[Any, str, str], None] | None = None) -> str:
        if self.busy:
            raise RuntimeError("已有任务正在运行。")
        run_id = uuid.uuid4().hex
        self.active_run_id = run_id
        self.eta_tracker = EtaTracker(plan, history=self.runtime.eta_history())
        self.cancel_event = Event()
        self.busy = True
        self._set_busy_controls(True)
        self.progress.set(0)
        self.cancel_button.configure(state="normal")
        self._busy_label = label
        self._busy_methods = list(methods)
        self._task_event_sequence = [0]
        self._completed_stage_samples: set[tuple[Any, ...]] = set()
        task_row = {
            "task_id": run_id, "run_id": run_id, "kind": task_kind, "methods": methods,
            "config": config_record, "result_directories": {}, "status": "running",
            "summary": f"{label}进行中",
        }
        self.runtime.begin_task(task_row)
        self.runtime.append_log({"run_id": run_id, "task_id": run_id, "type": "task-start",
                                 "label": label, "method": ",".join(methods), "status": "running"}, force=True)
        self._set_status(self._format_eta_status())

        def emit(event: dict[str, Any]) -> None:
            self._emit_for_run(run_id, self._task_event_sequence, lambda value: self.worker_queue.put(("runtime-event", value)), event)

        first = plan[0] if plan else StageEstimate("worker-startup", STAGE_LABELS["worker-startup"], 1.0)
        emit({"type": "stage-start", "stage_id": first.stage_id, "label": first.label,
              "method": first.method, "work_total": first.work_total, "unit": first.unit,
              "initial_seconds": first.initial_seconds})
        cancel_event = self.cancel_event

        def run_job() -> None:
            result: Any = None
            status = "success"
            summary = ""
            error_text = ""
            try:
                result = job(run_id, cancel_event, emit)
                if isinstance(result, dict) and "_task_status" in result:
                    status = str(result.pop("_task_status"))
                summary = str(result.get("_summary", "")) if isinstance(result, dict) else ""
            except InterruptedError as exc:
                status = "cancelled"
                summary = str(exc) or "任务已取消。"
            except BaseException as exc:
                status = "failed"
                summary = f"{type(exc).__name__}: {exc}"
                error_text = traceback.format_exc()
                emit({"type": "error", "error_type": type(exc).__name__, "details": error_text,
                      "status": status, "message_code": "task-failure"})
            finally:
                self.worker_queue.put(("runtime-finished", (run_id, status, result, summary, error_text, on_finish)))

        self.worker = Thread(target=run_job, name=f"semantic-task-{task_kind}", daemon=False)
        self.worker.start()
        return run_id

    def _format_eta_status(self) -> str:
        if not self.eta_tracker:
            return self._last_status_payload or "就绪。"
        snapshot = self.eta_tracker.snapshot()
        status = snapshot.get("terminal")
        if status:
            elapsed = self._duration(snapshot.get("task_elapsed_seconds", 0.0))
            final = {"success": "已完成", "partial": "部分方法完成，其余方法失败",
                     "cancelled": "已取消", "failed": "失败"}.get(status, status)
            return f"{self._busy_label}：{final}；总耗时 {elapsed}。\n任务历史与阶段日志已更新。"
        if self.cancel_event and self.cancel_event.is_set():
            method = snapshot.get("current_method", "")
            index, total = snapshot.get("method_index"), snapshot.get("method_total")
            progress = f"第 {index}/{total} 项　" if index and total else ""
            if method:
                progress += f"方法：{METHODS.get(method, method)}　"
            return (f"{self._busy_label}：正在取消并回收后台任务　{progress}阶段：{snapshot.get('stage_label', '')}；"
                    f"阶段已耗时 {self._duration(snapshot.get('stage_elapsed_seconds', 0.0))}\n"
                    f"全任务已耗时 {self._duration(snapshot.get('task_elapsed_seconds', 0.0))}；预计取消回收 1–10 秒，实际取决于当前系统调用。")
        method = snapshot.get("current_method", "")
        method_index = snapshot.get("method_index")
        method_total = snapshot.get("method_total")
        index_text = f"第 {method_index}/{method_total} 项　" if method_index and method_total else ""
        method_text = f"{index_text}方法：{METHODS.get(method, method)}　" if method else index_text
        stage_elapsed = self._duration(snapshot.get("stage_elapsed_seconds", 0.0))
        task_elapsed = self._duration(snapshot.get("task_elapsed_seconds", 0.0))
        stage_eta = snapshot.get("stage_remaining_seconds")
        task_eta = snapshot.get("task_remaining_seconds")
        source = snapshot.get("source", "初始估计")
        if source == "本次实测":
            source_text = f"本次实测，{snapshot.get('sample_count', 0)} 个观测批次"
        elif source == "成功历史":
            source_text = f"成功历史 {snapshot.get('history_samples', snapshot.get('sample_count', 0))} 次"
        else:
            source_text = source
        stage_eta_text = self._duration(stage_eta or 0.0)
        task_eta_text = self._duration(task_eta) if task_eta is not None else "—"
        range_value = snapshot.get("stage_remaining_range", [0, 0])
        confidence = f"；阶段工程参考范围 {self._duration(range_value[0])}–{self._duration(range_value[1])}" if len(range_value) == 2 else ""
        work = snapshot.get("work")
        work_text = ""
        if work and work.get("total"):
            work_text = f"；工作量 {work['done']:,.0f}/{work['total']:,.0f} {work['unit']}"
        rate = snapshot.get("observed_rate")
        speed_text = (f"实测速度 {rate:,.2f} {snapshot.get('observed_rate_unit', '工作量/秒')}"
                      if rate is not None else "实测速度尚未测得")
        overrun_text = "；已超出原估计，剩余估计会动态更新" if snapshot.get("overrun") else ""
        return (f"{self._busy_label}进行中　{method_text}阶段：{snapshot.get('stage_label', '')}；阶段已耗时 {stage_elapsed}，阶段预计剩余 {stage_eta_text}{work_text}\n"
                f"全任务已耗时 {task_elapsed}，全任务预计剩余 {task_eta_text}；{speed_text}；来源：{source_text}{confidence}{overrun_text}")

    @staticmethod
    def _duration(seconds: Any) -> str:
        try:
            value = max(0, int(round(float(seconds or 0))))
        except (TypeError, ValueError):
            value = 0
        if value < 60:
            if 0 < float(seconds or 0) < 1.0:
                return "不足 1 秒"
            return f"{value} 秒"
        minutes, secs = divmod(value, 60)
        if minutes < 60:
            return f"{minutes} 分 {secs} 秒"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} 小时 {minutes} 分"

    def _worker_command(self, task_directory: Path) -> tuple[list[str], str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, "--runtime-worker", str(task_directory)], str(Path(sys.executable).parent)
        main_path = Path(__file__).resolve().parent.parent / "main.py"
        return [sys.executable, str(main_path), "--runtime-worker", str(task_directory)], str(main_path.parent)

    def _run_worker_process(self, run_id: str, operation: str, method: str, method_index: int,
                            method_total: int, request_values: dict[str, Any],
                            emit: Callable[[dict[str, Any]], None], cancel_event: Event,
                            *, owned_paths: list[Path] | None = None) -> tuple[Path, dict[str, Any]]:
        work_root = self.runtime.root / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        task_dir = work_root / f"{run_id}-{method_index:02d}-{uuid.uuid4().hex[:8]}"
        task_dir.mkdir()
        child_run_id = f"{run_id}-{method_index:02d}-{operation}"
        emit({"type": "stage-start", "stage_id": "worker-startup", "label": "启动后台工作进程",
              "method": method, "method_index": method_index, "method_total": method_total,
              "initial_seconds": 1.2})
        request = dict(request_values)
        request.update({"run_id": child_run_id, "operation": operation, "method": method,
                        "method_index": method_index, "method_total": method_total})
        def bind_task_paths(value: Any) -> Any:
            if isinstance(value, str):
                return value.replace("$TASK_DIR", str(task_dir))
            if isinstance(value, list):
                return [bind_task_paths(item) for item in value]
            if isinstance(value, dict):
                return {key: bind_task_paths(item) for key, item in value.items()}
            return value
        request = bind_task_paths(request)
        request_path = task_dir / "request.json"
        events_path = task_dir / "events.jsonl"
        events_path.write_text("", encoding="utf-8")
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        command, cwd = self._worker_command(task_dir)
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        process: subprocess.Popen | None = None
        last_child_seq = -1
        killed_for_cancel = False
        cleanup_task_dir = False
        stdout_path = task_dir / "worker.stdout.log"
        stderr_path = task_dir / "worker.stderr.log"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        try:
            process = popen_owned_process(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout_handle,
                                          stderr=stderr_handle, startupinfo=startupinfo,
                                          creationflags=creationflags)
            with self._process_lock:
                self.current_process = process
            with events_path.open("r", encoding="utf-8", errors="replace") as event_handle:
                while True:
                    line = event_handle.readline()
                    if line:
                        try:
                            child_event = json.loads(line)
                            if child_event.get("run_id") != child_run_id:
                                continue
                            child_seq = int(child_event.get("seq", -1))
                            if child_seq <= last_child_seq:
                                continue
                            last_child_seq = child_seq
                            normalized = dict(child_event)
                            normalized.pop("run_id", None)
                            normalized.pop("seq", None)
                            normalized["type"] = str(normalized.get("type", "event"))
                            if normalized.get("type") == "stage-start" and normalized.get("stage_id") == "worker-startup":
                                continue
                            if normalized["type"] == "terminal":
                                if operation == "analyze":
                                    normalized["type"] = "method-terminal"
                                    normalized["method"] = method
                                else:
                                    continue
                            normalized["method_index"] = method_index
                            normalized["method_total"] = method_total
                            emit(normalized)
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass
                        continue
                    return_code = process.poll()
                    if return_code is not None:
                        # Drain any final complete records before consuming the terminal result.
                        final_line = event_handle.readline()
                        if final_line:
                            try:
                                child_event = json.loads(final_line)
                                if child_event.get("run_id") == child_run_id and int(child_event.get("seq", -1)) > last_child_seq:
                                    last_child_seq = int(child_event["seq"])
                                    normalized = dict(child_event)
                                    normalized.pop("run_id", None); normalized.pop("seq", None)
                                    if normalized.get("type") == "stage-start" and normalized.get("stage_id") == "worker-startup":
                                        continue
                                    normalized["method_index"] = method_index
                                    normalized["method_total"] = method_total
                                    if normalized.get("type") == "terminal":
                                        if operation == "analyze":
                                            normalized["type"] = "method-terminal"
                                            normalized["method"] = method
                                        else:
                                            continue
                                    emit(normalized)
                                    continue
                            except (ValueError, TypeError, json.JSONDecodeError):
                                pass
                        break
                    if cancel_event.is_set():
                        killed_for_cancel = True
                        terminate_owned_process(process, 5)
                        break
                    time.sleep(0.08)
            return_code = process.wait(timeout=5)
            orphaned_worker_tree = not owned_process_tree_finished(process, 0.0)
            if orphaned_worker_tree:
                terminate_owned_process(process, 5)
                if return_code == 0:
                    return_code = 1
            with self._process_lock:
                if self.current_process is process:
                    self.current_process = None
            if killed_for_cancel or cancel_event.is_set() and return_code != 0:
                for owned in owned_paths or []:
                    shutil.rmtree(owned, ignore_errors=True)
                cleanup_task_dir = True
                raise InterruptedError("已取消并结束本任务的后台进程。")
            if return_code != 0:
                detail = ("后台启动器已退出，但本任务仍有工作子进程；该进程树已被回收。"
                          if orphaned_worker_tree else "")
                error_type = "WorkerError"
                try:
                    error_value = json.loads((task_dir / "error.json").read_text(encoding="utf-8"))
                    error_type = str(error_value.get("type", error_type))
                    worker_detail = str(error_value.get("traceback", error_value.get("message", "")))
                    detail = (detail + "\n" + worker_detail).strip()
                except Exception:
                    detail = ""
                stderr_handle.flush()
                stdout_handle.flush()
                try:
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    stderr_text = ""
                if stderr_text:
                    detail = (detail + "\n\nworker stderr:\n" + stderr_text)[-100_000:]
                if not detail:
                    detail = f"worker process returned {return_code}"
                evidence_dir = self.runtime.logs_dir / "worker-errors" / f"{run_id}-{method_index:02d}-{method or operation}"
                try:
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    for evidence_file in (stdout_path, stderr_path, task_dir / "error.json"):
                        if evidence_file.is_file():
                            shutil.copy2(evidence_file, evidence_dir / evidence_file.name)
                    self.runtime.prune_worker_error_archives()
                    detail += f"\n完整 worker 启动日志：{evidence_dir}"
                except OSError as exc:
                    detail += f"\nworker 启动日志归档失败：{exc}"
                emit({"type": "error", "method": method, "error_type": error_type,
                      "details": detail, "status": "failed", "message_code": "worker-failure"})
                for owned in owned_paths or []:
                    shutil.rmtree(owned, ignore_errors=True)
                raise RuntimeError(f"{method or operation} 失败：{error_type}")
            output_path = task_dir / "output.json"
            output = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
            return task_dir, output
        except InterruptedError:
            cleanup_task_dir = True
            raise
        except BaseException:
            with self._process_lock:
                if self.current_process is process:
                    self.current_process = None
            if process is not None and (process.poll() is None or not owned_process_tree_finished(process, 0.0)):
                terminate_owned_process(process, 5)
            for owned in owned_paths or []:
                shutil.rmtree(owned, ignore_errors=True)
            cleanup_task_dir = True
            raise
        finally:
            try:
                stdout_handle.close()
            except OSError:
                pass
            try:
                stderr_handle.close()
            except OSError:
                pass
            if process is not None:
                close_owned_process_tree(process)
            if cleanup_task_dir:
                shutil.rmtree(task_dir, ignore_errors=True)

    def _persist_successful_stages(self, method: str) -> None:
        if not self.eta_tracker:
            return
        for sample in self.eta_tracker.completed:
            if sample.get("method") != method or sample.get("successful") is False or sample.get("skipped"):
                continue
            key = (method, sample.get("stage_id"), sample.get("elapsed_seconds"), sample.get("work_total"))
            if key in self._completed_stage_samples or float(sample.get("elapsed_seconds", 0) or 0) <= 0:
                continue
            self._completed_stage_samples.add(key)
            self.runtime.successful_stage({"method": method, "stage_id": sample.get("stage_id"),
                                           "cost_key": sample.get("cost_key", ""),
                                           "elapsed_seconds": sample.get("elapsed_seconds"),
                                           "work_total": sample.get("work_total"), "successful": True})

    def _browse_files(self) -> None:
        chosen = filedialog.askopenfilenames(parent=self, title="选择文本/表格文件",
                                             filetypes=[("支持的文本", "*.txt *.csv *.tsv *.xlsx *.jsonl"),
                                                        ("所有文件", "*.*")])
        if chosen:
            self._set_paths(list(chosen))

    def _browse_directory(self) -> None:
        chosen = filedialog.askdirectory(parent=self, title="选择包含 TXT 文件的目录")
        if chosen:
            self._set_paths([chosen])

    def _set_paths(self, paths: list[str]) -> None:
        for path in paths:
            if path not in self.selected_paths:
                self.selected_paths.append(path)
        names = [Path(path).name for path in self.selected_paths]
        self.files_label.configure(text="\n".join(names[:8]) + (f"\n… 共 {len(names)} 个来源" if len(names) > 8 else ""))
        table_path = next((p for p in self.selected_paths if Path(p).suffix.lower() in {".csv", ".tsv", ".xlsx", ".jsonl"}), None)
        if table_path:
            try:
                columns = table_columns(table_path)
                self.text_column_combo.configure(values=["自动"] + columns)
                self.id_column_combo.configure(values=["无"] + columns)
                if self.text_column_var.get() not in ["自动"] + columns:
                    self.text_column_var.set("自动")
                if self.id_column_var.get() not in ["无"] + columns:
                    self.id_column_var.set("无")
            except Exception as exc:
                self._set_status(f"读取表格列名失败：{exc}")

    def _browse_user_dictionary(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择 jieba 自定义词典", filetypes=[("文本", "*.txt"), ("所有文件", "*.*")])
        if path:
            self.user_dictionary_var.set(path)

    def _browse_stopwords(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择停用词文件", filetypes=[("文本", "*.txt"), ("所有文件", "*.*")])
        if path:
            self.custom_stopwords_var.set(path)

    def _browse_word2vec(self) -> None:
        path = filedialog.askopenfilename(parent=self, title="选择 word2vec 文本向量", filetypes=[("文本向量", "*.txt *.vec"), ("所有文件", "*.*")])
        if path:
            self.word2vec_path_var.set(path)

    def _load_clicked(self) -> None:
        pasted = self.paste_box.get("1.0", "end").strip()
        if not self.selected_paths and not pasted:
            messagebox.showinfo("需要输入", "请粘贴文本或选择文件。", parent=self)
            return
        self.runtime.save_last_settings(self._settings_snapshot())
        text_column = "" if self.text_column_var.get() == "自动" else self.text_column_var.get()
        id_column = "" if self.id_column_var.get() == "无" else self.id_column_var.get()
        split_mode, dedupe = self.split_var.get(), self.dedupe_var.get()
        paths = list(self.selected_paths)
        plan = [StageEstimate("load-corpus", STAGE_LABELS["load-corpus"],
                              max(1.0, len(paths) * 0.7 + len(pasted) / 250_000),
                              max(1, len(paths)), "文件/文本")]

        def job(_run_id: str, cancel: Event, emit: Callable[[dict[str, Any]], None]) -> Corpus:
            def progress(done: int, total: int, message: str) -> None:
                emit({"type": "stage-progress", "stage_id": "load-corpus", "work_done": done,
                      "work_total": max(1, total), "unit": "文件", "progress_message": message})
            corpus = load_corpus(paths, pasted_text=pasted, text_column=text_column,
                                 id_column=id_column, split_mode=split_mode, deduplicate=dedupe,
                                 cancel_event=cancel, progress=progress)
            emit({"type": "stage-complete", "stage_id": "load-corpus",
                  "work_done": max(1, len(paths)), "work_total": max(1, len(paths)), "unit": "文件"})
            return corpus

        self._start_runtime_job("导入", "load", [], plan, self._settings_snapshot(), job,
                                lambda result, _status, _summary: self._loaded(result))

    def _loaded(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self.result = None
        self.results_by_method.clear()
        self.result_sources.clear()
        self.result_corpora.clear()
        self.result_runtime_metadata.clear()
        self.result_elapsed.clear()
        self._refresh_result_picker()
        self.input_summary.configure(text=f"保留 {len(corpus.documents):,} 篇；空值排除 {corpus.stats['空值排除']:,}；重复排除 {corpus.stats['重复排除']:,}")
        self._update_export_buttons()
        self._set_status("导入完成。可到“方法”页选择分析方式。")

    def _update_method_hint(self) -> None:
        selected = self.method_var.get()
        method = next((key for key, label in METHODS.items() if label == selected), "bge")
        descriptions = {
            "bge": "本地 BGE-small-zh-v1.5 ONNX：CLS、L2 归一化；长文分块聚合是近似。网址 / 大小写清理影响编码文本；分词、停词、最小词长与数字 token 过滤仅影响辅助统计。",
            "tfidf-word": "词级 TF-IDF 稀疏表示；适合检查词项重合，不会识别同义词。",
            "tfidf-char": "按字符 n-gram 范围构建 TF-IDF；网址 / 大小写清理影响字符特征；分词、停词、最小词长与数字 token 过滤仅影响辅助统计。",
            "lsa": "在词 TF-IDF 上做 TruncatedSVD；至少需要 2 篇文档和 2 个词项。",
            "nmf": "非负矩阵分解，输入为词级 TF-IDF；输出文档主题权重与主题词。",
            "lda": "批量 LDA，输入为词项计数矩阵；输出文档主题分布与主题词。",
            "cooc-ppmi": "指定窗口内统计词对，再计算正 PMI；共现词表独立限额。",
            "word2vec": "流式扫描外部文本向量，仅保留当前词表命中向量；可按文档 TF-IDF 加权平均。",
        }
        self.method_hint.configure(text=descriptions[method])

    def _config_from_ui(self) -> AnalysisConfig:
        method = next((key for key, label in METHODS.items() if label == self.method_var.get()), "bge")
        try:
            numeric = {}
            for key, variable in self.extra_parameter_vars.items():
                value = variable.get().strip()
                if key.endswith("prior") and not value:
                    numeric[key] = None
                elif key == "max_df" or key.endswith("tol") or key.endswith("prior"):
                    numeric[key] = float(value)
                else:
                    numeric[key] = int(value)
            return AnalysisConfig(method=method, segmentation=self.segmentation_var.get(), stopwords=self.stopwords_var.get(),
                                  custom_stopwords_path=self.custom_stopwords_var.get().strip(),
                                  user_dictionary_path=self.user_dictionary_var.get().strip(),
                                  max_features=int(self.max_features_var.get()), min_frequency=int(self.min_frequency_var.get()),
                                  topic_count=int(self.topic_count_var.get()), cooc_window=int(self.cooc_window_var.get()),
                                  cooc_vocab_size=int(self.cooc_vocab_var.get()), clusters=int(self.clusters_var.get()),
                                  word2vec_path=self.word2vec_path_var.get().strip(),
                                  word2vec_tfidf_weight=self.word2vec_tfidf_var.get(),
                                  compute_similarity=self.similarity_var.get(),
                                  compute_clusters=self.compute_clusters_var.get(),
                                  case_mode=self.case_mode_var.get(),
                                  **{key: variable.get() for key, variable in self.extra_bool_vars.items()},
                                  **numeric).normalized()
        except ValueError as exc:
            raise ValueError(f"请检查数值参数：{exc}") from exc

    def _run_clicked(self) -> None:
        if self.corpus is None:
            messagebox.showinfo("需要数据", "请先在“数据”页读取文本。", parent=self)
            return
        try:
            config = self._config_from_ui()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return

        methods = [method for method, variable in self.batch_vars.items() if variable.get()]
        if not methods:
            methods = [config.method]
        base_config = config
        config_dicts = {method: asdict(replace(base_config, method=method)) for method in methods}
        corpus = self.corpus
        self.runtime.save_last_settings(self._settings_snapshot())
        doc_count = len(corpus.documents)
        char_count = sum(len(document.text) for document in corpus.documents)
        plan = estimate_stage_plan(methods, doc_count, char_count, config_dicts, history=self.runtime.eta_history())

        def job(run_id: str, cancel: Event, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            result_values: dict[str, AnalysisResult] = {}
            result_paths: dict[str, str] = {}
            corpora: dict[str, Corpus] = {}
            load_metadata: dict[str, Any] = {}
            failures: list[str] = []
            result_root = self.runtime.root / "results" / run_id
            result_root.mkdir(parents=True, exist_ok=True)
            for index, method in enumerate(methods, 1):
                if cancel.is_set():
                    raise InterruptedError("已取消；已完成方法的结果已保留。")
                request = {"corpus": corpus_to_dict(corpus), "config": config_dicts[method],
                           "result_directory": "$TASK_DIR/result-cache"}
                try:
                    task_dir, _ = self._run_worker_process(run_id, "analyze", method, index, len(methods),
                                                           request, emit, cancel)
                    staged_cache = task_dir / "result-cache"
                    permanent_cache = result_root / f"{method}.cst"
                    permanent_cache.parent.mkdir(parents=True, exist_ok=True)
                    if permanent_cache.exists():
                        raise FileExistsError(f"結果緩存已存在：{permanent_cache}")
                    os.replace(staged_cache, permanent_cache)
                    emit({"type": "stage-start", "stage_id": "result-load", "label": "载入分析结果",
                          "method": method, "work_total": 1, "unit": "结果包", "initial_seconds": 0.8})
                    cache_corpus, cache_results, _cache_settings, cache_metadata = load_project_full(permanent_cache)
                    if method not in cache_results:
                        raise ValueError(f"后台分析没有生成 {method} 结果。")
                    emit({"type": "stage-complete", "stage_id": "result-load", "method": method,
                          "work_done": 1, "work_total": 1, "unit": "结果包"})
                    result_values[method] = cache_results[method]
                    result_paths[method] = str(permanent_cache)
                    # Keep the exact input corpus in memory for source-linked browsing.
                    # Disk caches still deliberately omit text.
                    corpora[method] = corpus
                    load_metadata[method] = cache_metadata.get(method, {})
                    shutil.rmtree(task_dir, ignore_errors=True)
                except InterruptedError:
                    if result_values:
                        return {"results": result_values, "result_paths": result_paths, "corpora": corpora,
                                "runtime_metadata": load_metadata, "failures": failures, "_result_paths": result_paths,
                                "_task_status": "cancelled", "_summary": "已取消；已完成方法的结果已保留。"}
                    raise
                except Exception as exc:
                    failures.append(f"{METHODS[method]}：{type(exc).__name__}: {exc}")
                    emit({"type": "warning", "method": method, "warning": failures[-1],
                          "message_code": "method-failure-continue"})
                    continue
                if cancel.is_set():
                    return {"results": result_values, "result_paths": result_paths, "corpora": corpora,
                            "runtime_metadata": load_metadata, "failures": failures, "_result_paths": result_paths,
                            "_task_status": "cancelled", "_summary": "已取消；已完成方法的结果已保留。"}
            if not result_values:
                raise RuntimeError("所有所选方法都未能完成。" + ("\n" + "\n".join(failures) if failures else ""))
            status = "partial" if failures else "success"
            return {"results": result_values, "result_paths": result_paths, "corpora": corpora,
                    "runtime_metadata": load_metadata, "failures": failures, "_result_paths": result_paths, "_task_status": status,
                    "_summary": ("部分完成：" + "；".join(failures)) if failures else f"完成 {len(result_values)} 种方法"}

        run_id = self._start_runtime_job("分析", "analysis", methods, plan, config_dicts, job,
                                         lambda result, _status, _summary: self._analysis_finished(result))
        if self.ui_smoke_dir:
            self._smoke_analysis_run_id = run_id

    def _analysis_finished(self, payload: dict[str, Any]) -> None:
        self.results_by_method.update(payload.get("results", {}))
        self.result_sources.update({key: Path(value) for key, value in payload.get("result_paths", {}).items()})
        self.result_corpora.update(payload.get("corpora", {}))
        self.result_runtime_metadata.update(payload.get("runtime_metadata", {}))
        self.result_elapsed.update({method: float(value.get("elapsed_seconds", 0.0) or 0.0)
                                    for method, value in self.result_runtime_metadata.items()})
        for method, result in self.results_by_method.items():
            self.result_elapsed.setdefault(method, 0.0)
            metadata = result.clustering_summary
            self.result_runtime_metadata.setdefault(method, {})
        self._refresh_result_picker()
        if self.result:
            self._populate_results(self.result)
        self._refresh_comparison()
        self._update_export_buttons()
        if payload.get("failures"):
            self._set_status("部分方法完成，其余方法失败并已记录；可在任务页查看完整日志。\n" + "\n".join(payload["failures"]))
        elif self.result:
            self._set_status("分析完成。相似度是指定表示空间中的余弦值，不是概率或准确率。")
        if self.result:
            self.show_page("results")

    def _analyzed(self, result: AnalysisResult) -> None:
        self.results_by_method[result.method] = result
        if self.corpus is not None and self.corpus.ids == result.doc_ids:
            self.result_corpora[result.method] = self.corpus
        self.result = result
        self._populate_results(result)
        self._refresh_result_picker()
        self._update_export_buttons()
        self.show_page("results")

    def _selected_result_method(self) -> str:
        label = self.result_method_var.get()
        return next((method for method, title in METHODS.items() if title == label), "")

    def _refresh_result_picker(self) -> None:
        methods = list(self.results_by_method)
        labels = [METHODS.get(method, method) for method in methods]
        self.result_method_combo.configure(values=labels)
        current = self._selected_result_method()
        if current not in methods:
            current = methods[-1] if methods else ""
        self.result_method_var.set(METHODS.get(current, "") if current else "")
        if current:
            self.result = self.results_by_method[current]
            self._populate_results(self.result)
        else:
            self.result = None
            self.explorer.clear()
            self._set_readonly_text(self.overview_box, "尚未运行分析。请在方法页开始计算。")
            for child in self.heatmap_frame.winfo_children():
                child.destroy()
        self._update_export_buttons()

    def _select_result_method(self) -> None:
        method = self._selected_result_method()
        result = self.results_by_method.get(method)
        if result is None:
            return
        self.result = result
        self._populate_results(result)

    def _update_export_buttons(self) -> None:
        selected = self._selected_result_method()
        one = selected in self.result_sources
        many = any(method in self.result_sources for method in self.results_by_method)
        self.export_button.winfo_children()[0].configure(state="normal" if one else "disabled")
        self.export_all_button.winfo_children()[0].configure(state="normal" if many else "disabled")

    def _refresh_comparison(self) -> None:
        tree = getattr(self, "comparison_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        for method, result in self.results_by_method.items():
            summary = result.clustering_summary
            silhouette = summary.get("轮廓系数") if summary.get("状态") == "已计算" else "—"
            self.result_elapsed.setdefault(method, 0.0)
            self.result_runtime_metadata.setdefault(method, {})
            tree.insert("", "end", values=(METHODS.get(method, method),
                                            self._duration(self.result_elapsed[method]),
                                            result.doc_matrix.shape[1], len(result.doc_ids),
                                            silhouette if silhouette is not None else "—"))

    def _write_comparison_csv(self, path: Path) -> None:
        import csv
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["方法", "分析耗时（秒）", "向量维数", "有效文档", "轮廓系数（如适用）"])
            for method, result in self.results_by_method.items():
                summary = result.clustering_summary
                silhouette = summary.get("轮廓系数") if summary.get("状态") == "已计算" else ""
                writer.writerow([METHODS.get(method, method), self.result_elapsed.get(method, 0.0),
                                 result.doc_matrix.shape[1], len(result.doc_ids),
                                 silhouette if silhouette is not None else ""])

    def _open_project_clicked(self) -> None:
        chosen = filedialog.askdirectory(parent=self, title="选择已保存的语义分析项目目录")
        if not chosen:
            return
        self._open_project_path(Path(chosen))

    def _open_project_path(self, project_path: Path) -> None:
        plan = [StageEstimate("project-load", STAGE_LABELS["project-load"], 2.0, 1, "项目包")]

        def job(_run_id: str, _cancel: Event, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            emit({"type": "stage-start", "stage_id": "project-load", "label": "读取并校验项目",
                  "work_total": 1, "unit": "项目包", "initial_seconds": 2.0})
            corpus, results, settings, metadata = load_project_full(project_path)
            emit({"type": "stage-complete", "stage_id": "project-load", "work_done": 1,
                  "work_total": 1, "unit": "项目包"})
            return {"corpus": corpus, "results": results, "settings": settings, "metadata": metadata,
                    "project_path": str(project_path)}

        self._start_runtime_job("打开项目", "project-open", [], plan, {}, job,
                                lambda payload, _status, _summary: self._project_opened(payload))

    def _project_opened(self, payload: dict[str, Any]) -> None:
        self.corpus = payload["corpus"]
        self.results_by_method = dict(payload["results"])
        self.result = None
        self.result_sources = {method: Path(payload["project_path"]) for method in self.results_by_method}
        self.result_corpora = {method: self.corpus for method in self.results_by_method}
        metadata = payload.get("metadata", {})
        self.result_runtime_metadata = dict(metadata)
        self.result_elapsed = {method: float(metadata.get(method, {}).get("elapsed_seconds", 0.0) or 0.0)
                               for method in self.results_by_method}
        missing = self._apply_settings(payload.get("settings", {}))
        self.runtime.save_last_settings(self._settings_snapshot())
        self.input_summary.configure(text=f"项目语料 {len(self.corpus.documents):,} 篇；可用结果 {len(self.results_by_method)} 种")
        self._refresh_result_picker()
        self._refresh_comparison()
        self._update_export_buttons()
        self.project_state_label.configure(text=f"已打开：{payload['project_path']}" + (f"\n资源缺失：{missing}" if missing else ""))
        if self.result:
            self.show_page("results")
        self._set_status(f"项目已打开，可直接查看或导出 {len(self.results_by_method)} 种已完成结果。" if self.results_by_method
                         else "项目已打开；当前项目含语料和设置，尚无分析结果。")

    def _save_project_clicked(self) -> None:
        if self.corpus is None:
            messagebox.showinfo("需要语料", "请先导入语料，或打开已有项目。", parent=self)
            return
        parent = filedialog.askdirectory(parent=self, title="选择项目保存位置")
        if not parent:
            return
        name = simpledialog.askstring("项目名称", "请输入新项目文件夹名称：", parent=self)
        if not name or not name.strip():
            return
        folder_name = Path(name.strip()).name
        destination = Path(parent) / folder_name
        if destination.exists():
            messagebox.showerror("项目已存在", f"请换一个名称；已有目录不会被覆盖：\n{destination}", parent=self)
            return
        caches = list(dict.fromkeys(str(self.result_sources[m]) for m in self.results_by_method
                                    if m in self.result_sources and self.result_sources[m].is_dir()))
        methods = list(self.results_by_method)
        plan = [StageEstimate("worker-startup", STAGE_LABELS["worker-startup"], 1.0),
                StageEstimate("project-load", STAGE_LABELS["project-load"],
                              max(0.6, (len(caches) + 1) * 0.5), len(caches) + 1, "项目/结果包")]
        plan.append(StageEstimate("project-save", STAGE_LABELS["project-save"],
                                  max(1.0, len(methods) * 0.6 + len(self.corpus.documents) * 0.0005),
                                  max(1, len(methods)), "分析结果"))
        self._save_last_settings_from_ui()
        run_settings = self._settings_snapshot()
        metadata = dict(self.result_runtime_metadata)
        staging = destination.parent / f".{destination.name}.pending-{uuid.uuid4().hex}"

        def job(run_id: str, cancel: Event, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            if cancel.is_set():
                raise InterruptedError("项目保存已取消。")
            _task_dir, output = self._run_worker_process(
                run_id, "save-project", "", 1, 1,
                {"destination": str(destination), "staging_directory": str(staging),
                 "corpus": corpus_to_dict(self.corpus), "result_directories": caches,
                 "settings": run_settings, "runtime_metadata": metadata}, emit, cancel,
                owned_paths=[staging])
            return {"path": str(output.get("path", destination))}

        self._start_runtime_job("保存项目", "project-save", methods, plan, run_settings, job,
                                lambda payload, _status, _summary: self._project_saved(payload))

    def _save_last_settings_from_ui(self) -> None:
        self.runtime.save_last_settings(self._settings_snapshot())

    def _project_saved(self, payload: dict[str, Any]) -> None:
        path = payload.get("path", "")
        self.project_state_label.configure(text=f"已保存项目：{path}")
        self._set_status(f"项目已安全保存：\n{path}")

    def _open_history_clicked(self) -> None:
        selected = self.task_history_tree.selection()
        if not selected:
            messagebox.showinfo("选择历史任务", "请先在任务历史表中选择一个任务。", parent=self)
            return
        row = next((item for item in self.runtime.history() if item.get("task_id") == selected[0]), None)
        paths = row.get("result_directories", {}) if row else {}
        caches = {method: Path(path) for method, path in paths.items()
                  if isinstance(path, str) and Path(path).is_dir()}
        if not caches:
            messagebox.showinfo("没有已保存结果", "该任务没有可重新打开的已完成结果。", parent=self)
            return
        methods = list(caches)
        plan = [StageEstimate("project-load", STAGE_LABELS["project-load"],
                              max(0.8, len(methods)), 1, "结果包")]

        def job(_run_id: str, _cancel: Event, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            results: dict[str, AnalysisResult] = {}
            corpora: dict[str, Corpus] = {}
            metadata_rows: dict[str, Any] = {}
            emit({"type": "stage-start", "stage_id": "project-load", "label": "读取历史结果",
                  "work_total": max(1, len(caches)), "unit": "结果包", "initial_seconds": max(0.8, len(caches))})
            for index, (method, path) in enumerate(caches.items(), 1):
                if _cancel.is_set():
                    raise InterruptedError("已取消载入历史结果。")
                corpus, found, _settings, metadata = load_project_full(path)
                if method not in found:
                    raise ValueError(f"历史结果包缺少方法 {method}。")
                results[method] = found[method]
                corpora[method] = corpus
                emit({"type": "stage-progress", "stage_id": "project-load", "method": method,
                      "work_done": index, "work_total": max(1, len(caches)), "unit": "结果包"})
                metadata_rows[method] = metadata.get(method, {})
            emit({"type": "stage-complete", "stage_id": "project-load", "work_done": max(1, len(caches)),
                  "work_total": max(1, len(caches)), "unit": "结果包"})
            return {"results": results, "corpora": corpora, "paths": {k: str(v) for k, v in caches.items()},
                    "metadata": metadata_rows}

        self._start_runtime_job("打开历史结果", "history-open", methods, plan, {}, job,
                                lambda payload, _status, _summary: self._history_opened(payload))

    def _history_opened(self, payload: dict[str, Any]) -> None:
        self.results_by_method = dict(payload["results"])
        self.result_sources = {method: Path(path) for method, path in payload["paths"].items()}
        self.result_corpora = dict(payload["corpora"])
        self.result_runtime_metadata = dict(payload.get("metadata", {}))
        restored_corpus = next(iter(self.result_corpora.values()), None)
        # Matching IDs alone do not prove that the currently imported text belongs
        # to these historical results. Preserve the history's exact corpus binding.
        self.corpus = restored_corpus
        self.result_elapsed = {method: float(meta.get("elapsed_seconds", 0.0) or 0.0)
                               for method, meta in self.result_runtime_metadata.items()}
        self._refresh_result_picker()
        self._refresh_comparison()
        self._update_export_buttons()
        if self.result:
            self.input_summary.configure(text=f"历史结果：{len(self.results_by_method)} 种方法，可直接查看和导出。")
            self.show_page("results")
        self._set_status(f"已恢复 {len(self.results_by_method)} 种历史结果，可直接查看或导出。")

    def _export_clicked(self) -> None:
        method = self._selected_result_method()
        if not method or method not in self.result_sources:
            messagebox.showinfo("需要结果", "请先完成一次分析。", parent=self)
            return
        chosen = filedialog.askdirectory(parent=self, title="选择导出父目录")
        if not chosen:
            return
        self._export_methods([method], Path(chosen))

    def _export_all_clicked(self) -> None:
        methods = [method for method in self.results_by_method if method in self.result_sources]
        if not methods:
            messagebox.showinfo("需要结果", "当前没有可导出的已完成方法。", parent=self)
            return
        chosen = filedialog.askdirectory(parent=self, title="选择导出父目录")
        if chosen:
            self._export_methods(methods, Path(chosen))

    def _export_methods(self, methods: list[str], parent: Path) -> None:
        stages: list[StageEstimate] = []
        for method in methods:
            stages.extend([
                StageEstimate("worker-startup", STAGE_LABELS["worker-startup"], 1.0, method=method),
                StageEstimate("project-load", STAGE_LABELS["project-load"], 0.8, 1, "结果包", method),
                StageEstimate("export-tables", STAGE_LABELS["export-tables"], 2.0, 14, "导出项目", method),
                StageEstimate("export-report", STAGE_LABELS["export-report"], 0.8, 4, "文件", method),
            ])
            if self.xlsx_var.get():
                stages.append(StageEstimate("export-xlsx", STAGE_LABELS["export-xlsx"], 2.0, 1, "工作簿", method))
        stages.append(StageEstimate("comparison-export", "生成方法比较表", 0.3, 1, "表格"))
        settings = self._settings_snapshot()
        settings["export_methods"] = methods
        xlsx = bool(self.xlsx_var.get())
        cluster_names = {method: dict(self.result_runtime_metadata.get(method, {}).get("cluster_names", {}))
                         for method in methods}

        def job(run_id: str, cancel: Event, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
            paths: list[str] = []
            errors: list[str] = []
            for index, method in enumerate(methods, 1):
                if cancel.is_set():
                    raise InterruptedError("已取消；此前已完成的导出目录仍然保留。")
                staging = parent / f".cst-pending-{run_id}-{method}-{uuid.uuid4().hex[:6]}"
                try:
                    task_dir, output = self._run_worker_process(
                        run_id, "export", method, index, len(methods),
                        {"result_directory": str(self.result_sources[method]), "parent_directory": str(parent),
                         "staging_directory": str(staging), "include_xlsx": xlsx,
                         "cluster_names": cluster_names[method]}, emit, cancel,
                        owned_paths=[staging])
                    paths.append(str(output.get("path", "")))
                    shutil.rmtree(task_dir, ignore_errors=True)
                except InterruptedError:
                    raise
                except Exception as exc:
                    errors.append(f"{METHODS[method]}：{type(exc).__name__}: {exc}")
                    emit({"type": "warning", "method": method, "warning": errors[-1],
                          "message_code": "export-failure-continue"})
            emit({"type": "stage-start", "stage_id": "comparison-export", "label": "生成方法比较表",
                  "work_total": 1, "unit": "表格", "initial_seconds": 0.3})
            self._write_comparison_csv(parent / f"方法比较_{run_id[:8]}.csv")
            emit({"type": "stage-complete", "stage_id": "comparison-export", "work_done": 1,
                  "work_total": 1, "unit": "表格"})
            if not paths:
                raise RuntimeError("未能导出任何结果。" + ("\n" + "\n".join(errors) if errors else ""))
            return {"paths": paths, "failures": errors, "_task_status": "partial" if errors else "success",
                    "_summary": f"导出 {len(paths)} 种方法" + (f"；失败 {len(errors)} 种" if errors else "")}

        self._start_runtime_job("导出", "export", methods, stages, settings, job,
                                lambda result, _status, _summary: self._exported(result))

    def _exported(self, payload: dict[str, Any]) -> None:
        paths = payload.get("paths", []) if isinstance(payload, dict) else [str(payload)]
        self._set_status("導出完成：\n" + "\n".join(paths))
        if payload.get("failures"):
            self._set_status("部分结果已导出，失败方法已记录。\n" + "\n".join(payload["failures"]))

    def cancel_work(self) -> None:
        if self.cancel_event and self.busy:
            self.cancel_event.set()
            self.cancel_button.configure(state="disabled")
            self._set_status("正在取消；本任务后台进程结束后会停止并清理本任务暂存文件。")

    def _on_close(self) -> None:
        if not self.busy:
            self.destroy()
            return
        choice = messagebox.askyesnocancel("任务仍在运行", "仍有任务正在运行。\n\n是：取消任务并在本任务进程结束后关闭\n否：继续等待\n取消：返回程序",
                                           parent=self)
        if choice is True:
            self._close_after_task = True
            self.cancel_work()

    def _handle_runtime_event(self, event: dict[str, Any]) -> None:
        if event.get("run_id") != self.active_run_id or self.eta_tracker is None:
            return
        seq = int(event.get("seq", -1))
        if seq <= getattr(self, "_last_ui_event_seq", -1):
            return
        self._last_ui_event_seq = seq
        accepted = self.eta_tracker.accept(event)
        if not accepted:
            return
        kind = event.get("type")
        if kind == "method-terminal" and event.get("status") == "success":
            method = str(event.get("method", ""))
            self._persist_successful_stages(method)
            if event.get("elapsed_seconds") is not None:
                self.result_elapsed[method] = float(event["elapsed_seconds"])
                self._refresh_comparison()
        if kind == "stage-start":
            self.progress.set(0.0)
        elif kind == "stage-progress":
            total = event.get("work_total")
            done = event.get("work_done")
            if total and done is not None:
                self.progress.set(max(0.0, min(0.98, float(done) / max(float(total), 1e-12))))
        elif kind == "terminal":
            if event.get("status") == "success":
                self.progress.set(1.0)
            self._refresh_task_history()
        if kind in {"warning", "error"}:
            self._refresh_task_history()
        self._set_status(self._format_eta_status())

    def _set_busy_controls(self, busy: bool) -> None:
        for name, page in self.page_widgets.items():
            if name != "tasks":
                self._set_widget_tree_state(page, busy)
        for button in self.nav_buttons.values():
            button.configure(state="normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.cancel_button.lift()

    def _set_widget_tree_state(self, parent: Any, busy: bool) -> None:
        for child in parent.winfo_children():
            if isinstance(child, (ctk.CTkButton, ctk.CTkEntry, ctk.CTkComboBox, ctk.CTkTextbox, ctk.CTkCheckBox)):
                try:
                    if busy:
                        self._busy_widget_states[child] = str(child.cget("state"))
                        child.configure(state="disabled")
                    elif child in self._busy_widget_states:
                        child.configure(state=self._busy_widget_states.pop(child))
                except Exception:
                    pass
            elif isinstance(child, ttk.Treeview):
                continue
            else:
                self._set_widget_tree_state(child, busy)

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "runtime-event":
                    self._handle_runtime_event(payload)
                elif kind == "runtime-finished":
                    run_id, status, result, summary, details, callback = payload
                    if run_id != self.active_run_id:
                        continue
                    has_cancelled_results = (status == "cancelled" and isinstance(result, dict)
                                             and bool(result.get("results")))
                    if callback and (status in {"success", "partial"} or has_cancelled_results):
                        try:
                            callback(result, status, summary)
                        except Exception as exc:
                            details = traceback.format_exc()
                            summary = f"结果显示失败：{type(exc).__name__}: {exc}"
                            status = "failed"
                            self._emit_for_run(run_id, self._task_event_sequence,
                                               lambda value: self._handle_runtime_event(value),
                                               {"type": "error", "error_type": type(exc).__name__,
                                                "details": details, "status": status,
                                                "message_code": "result-display-failure"})
                    elapsed = max(0.0, time.monotonic() - self.eta_tracker.started_at) if self.eta_tracker else 0.0
                    self._emit_for_run(run_id, self._task_event_sequence,
                                       lambda value: self._handle_runtime_event(value),
                                       {"type": "terminal", "status": status, "elapsed_seconds": elapsed,
                                        "method": ",".join(self._busy_methods), "message_code": self._busy_label})
                    result_paths = result.get("_result_paths", {}) if isinstance(result, dict) else {}
                    from datetime import datetime
                    self.runtime.update_task(run_id, status=status,
                                             ended_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                                             summary=summary or ("已完成" if status == "success" else status),
                                             result_directories=result_paths or {})
                    self.busy = False
                    self._set_busy_controls(False)
                    self._refresh_task_history()
                    self._set_status(self._format_eta_status())
                    if status == "failed":
                        log_path = self.runtime.logs_dir / "latest-error.log"
                        log_path.write_text(details or summary, encoding="utf-8")
                        messagebox.showerror("操作失败", f"{summary}\n\n完整日志可在任务页查看。", parent=self)
                    if self._close_after_task:
                        self.destroy()
                        return
                elif kind == "cancelled":
                    self.busy = False
                    self._set_busy_controls(False)
                    self._set_status(payload or "操作已取消。")
        except Empty:
            pass
        if self.busy and self.eta_tracker:
            self._set_status(self._format_eta_status())
        try:
            exists = bool(self.winfo_exists())
        except tk.TclError:
            return
        if exists:
            try:
                self.after(100, self._poll_worker)
            except tk.TclError:
                return

    def _set_status(self, message: str) -> None:
        self.status_label.configure(text=message)

    def _populate_results(self, result: AnalysisResult) -> None:
        lines = [f"方法：{result.method_label}", f"文档数：{len(result.doc_ids):,}",
                 f"文档表示：{result.doc_matrix_name}，形状 {result.doc_matrix.shape[0]} × {result.doc_matrix.shape[1]}",
                 f"文档词项计数：{result.word_count_matrix.shape[0]} × {result.word_count_matrix.shape[1]}（稀疏）",
                 f"词级 TF-IDF：{result.word_tfidf_matrix.shape[0]} × {result.word_tfidf_matrix.shape[1]}（稀疏）",
                 f"完整相似度矩阵：{'已生成 ' + str(result.similarity_matrix.shape) if result.similarity_matrix is not None else '未生成（可使用近邻结果）'}",
                 f"全零表示：{len(result.zero_document_ids)} 篇。全零项不计算相似度，不视为完美自相似。",
                 f"KMeans：{result.clustering_summary.get('状态')}；轮廓系数：{result.clustering_summary.get('轮廓系数')}；{result.clustering_summary.get('轮廓系数说明')}",
                 "", "注意事项："]
        lines.extend([f"• {warning}" for warning in result.warnings] or ["• 无"])
        self._set_readonly_text(self.overview_box, "\n".join(lines))
        corpus = self.result_corpora.get(result.method)
        self.explorer.set_result(result, corpus,
                                 self.result_runtime_metadata.get(result.method, {}).get("cluster_names", {}).copy())
        self._draw_heatmap(result)

    def _set_tree(self, tree: ttk.Treeview, frame: Any, row_limit: int = 300) -> None:
        columns = [str(c) for c in list(frame.columns)[:12]]
        tree.delete(*tree.get_children())
        tree.configure(columns=columns)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150 if col in {"文档ID", "近邻ID", "来源"} else 115, minwidth=70, stretch=True)
        for row in frame.loc[:, columns].head(row_limit).itertuples(index=False, name=None):
            values = []
            for item in row:
                if isinstance(item, float):
                    values.append(f"{item:.6g}")
                else:
                    values.append(str(item))
            tree.insert("", "end", values=values)

    def _draw_heatmap(self, result: AnalysisResult) -> None:
        for child in self.heatmap_frame.winfo_children():
            child.destroy()
        if result.similarity_matrix is None:
            self._label(self.heatmap_frame, "未生成完整矩阵。文档超过 1,500 篇或已关闭矩阵计算时，可在“近邻”页查看按 Top-K 设置分块计算的近邻。",
                        font=BODY_FONT, anchor="nw", justify="left", wraplength=800).pack(fill="x", padx=12, pady=14)
            return
        limit = min(len(result.doc_ids), 80)
        self._label(self.heatmap_frame, f"前 {limit} × {limit} 篇文档预览；导出可获得完整矩阵。格子颜色表示余弦相似度，坐标按当前文档顺序。",
                    font=SMALL_FONT, anchor="w", justify="left", wraplength=830).pack(fill="x", padx=8, pady=(8, 2))
        host = ctk.CTkFrame(self.heatmap_frame, fg_color=WHITE, border_color=INK, border_width=1, corner_radius=8)
        host.pack(fill="both", expand=True, padx=8, pady=6)
        canvas = tk.Canvas(host, bg=WHITE, highlightthickness=0, width=840, height=540)
        yscroll = ttk.Scrollbar(host, orient="vertical", command=canvas.yview)
        xscroll = ttk.Scrollbar(host, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        host.grid_rowconfigure(0, weight=1)
        host.grid_columnconfigure(0, weight=1)
        self.heatmap_canvas = canvas
        cell = 9
        left, top = 62, 58
        matrix = result.similarity_matrix[:limit, :limit]
        canvas.create_text(left, 17, text="列：文档顺序", fill=INK, anchor="w", font=("Microsoft YaHei UI", 9))
        canvas.create_text(18, top, text="行", fill=INK, anchor="w", font=("Microsoft YaHei UI", 9))
        for i in range(0, limit, 10):
            canvas.create_text(left + i * cell + cell / 2, top - 8, text=str(i + 1), fill=INK, font=("Segoe UI", 8))
            canvas.create_text(left - 12, top + i * cell + cell / 2, text=str(i + 1), fill=INK, font=("Segoe UI", 8))
        for r in range(limit):
            for c in range(limit):
                value = matrix[r, c]
                if not np_isfinite(value):
                    color = "#d7dbe4"
                else:
                    val = max(-1.0, min(1.0, float(value)))
                    target = (32, 196, 106) if val >= 0 else (255, 180, 170)
                    amount = abs(val)
                    rgb = tuple(int(255 * (1 - amount) + component * amount) for component in target)
                    color = "#%02x%02x%02x" % rgb
                x0, y0 = left + c * cell, top + r * cell
                canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=color, outline="")
        canvas.configure(scrollregion=(0, 0, left + limit * cell + 15, top + limit * cell + 15))

    def _set_readonly_text(self, widget: Any, value: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def _smoke_step(self, index: int) -> None:
        pages = ["data", "methods", "results", "tasks", "help"]
        if index >= len(pages):
            self.after(300, self.quit)
            return
        self.show_page(pages[index])
        self.update_idletasks()
        self.after(500, lambda: self._capture_smoke_page(pages, index))

    def _capture_smoke_page(self, pages: list[str], index: int) -> None:
        self._grab_smoke(f"{index + 1:02d}-{pages[index]}.png")
        if pages[index] == "methods":
            canvas = getattr(self.page_widgets["methods"], "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_moveto(1.0)
            self.update_idletasks()
            self.after(450, lambda: self._capture_smoke_extra(pages, index, "05-methods-bottom.png", reset_methods=True))
            return
        if pages[index] == "results":
            self.after(300, lambda: self._capture_smoke_extra(pages, index, "06-results-export.png"))
            return
        if pages[index] == "tasks":
            self._smoke_start_batch(index, pages)
            return
        self.after(300, lambda: self._smoke_step(index + 1))

    def _capture_smoke_extra(self, pages: list[str], index: int, filename: str, *, reset_methods: bool = False) -> None:
        self._grab_smoke(filename)
        if reset_methods:
            canvas = getattr(self.page_widgets["methods"], "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_moveto(0.0)
        self.after(300, lambda: self._smoke_step(index + 1))

    def _grab_smoke(self, filename: str) -> None:
        try:
            self.lift()
            self.update_idletasks()
            self.update()
            from PIL import ImageGrab
            self.ui_smoke_dir.mkdir(parents=True, exist_ok=True)
            image = ImageGrab.grab(window=int(self.winfo_id()))
            image.save(self.ui_smoke_dir / filename)
            self.smoke_capture_count += 1
        except Exception as exc:
            self.smoke_errors.append(f"{filename}: {type(exc).__name__}: {exc}")
            self._set_status(f"界面截图失败：{exc}")

    def _smoke_start_batch(self, page_index: int, pages: list[str]) -> None:
        self.corpus = Corpus(documents=[
            Document("smoke-1", "苹果手机相机拍摄清晰，电池续航持久。", "ui-smoke", 1),
            Document("smoke-2", "安卓手机摄影夜景明亮，电池使用时间很长。", "ui-smoke", 2),
            Document("smoke-3", "足球联赛球队训练，球员传球射门得分。", "ui-smoke", 3),
            Document("smoke-4", "篮球球队比赛训练，球员投篮防守得分。", "ui-smoke", 4),
            Document("smoke-5", "天气预报显示晴天，城市公园适合步行。", "ui-smoke", 5),
            Document("smoke-6", "城市天气温暖，公园绿地适合散步。", "ui-smoke", 6),
        ], cleaning_log=[{"来源": "ui-smoke 合成语料", "读入行/段": 6}],
            stats={"读入记录": 6, "空值排除": 0, "重复排除": 0, "保留文档": 6})
        self.input_summary.configure(text="UI smoke 合成语料：6 篇；批量运行 TF-IDF 两种方法。")
        for method, variable in self.batch_vars.items():
            variable.set(method in {"tfidf-word", "tfidf-char"})
        self.method_var.set(METHODS["tfidf-word"])
        self.show_page("tasks")
        self._run_clicked()
        self.after(250, lambda: self._smoke_running_capture(page_index, pages))

    def _smoke_running_capture(self, page_index: int, pages: list[str]) -> None:
        self._grab_smoke("07-tasks-running-eta.png")
        self._smoke_wait_for_analysis(page_index, pages, time.monotonic() + 90.0)

    def _smoke_wait_for_analysis(self, page_index: int, pages: list[str], deadline: float) -> None:
        if self.busy and time.monotonic() < deadline:
            self.after(150, lambda: self._smoke_wait_for_analysis(page_index, pages, deadline))
            return
        if self.busy:
            self.smoke_errors.append("批量 smoke 超过 90 秒仍在运行。")
            self.cancel_work()
            return
        run_id = getattr(self, "_smoke_analysis_run_id", None)
        if not run_id or not self.results_by_method:
            self.smoke_errors.append("批量 smoke 没有恢复已完成结果。")
            self.after(200, lambda: self._smoke_step(page_index + 1))
            return
        if any(not any(doc.text for doc in corpus.documents) for corpus in self.result_corpora.values()):
            self.smoke_errors.append("新分析的内存语料丢失原文，无法用于原文联动。")
        if run_id in self.task_history_tree.get_children(""):
            self.task_history_tree.selection_set(run_id)
        else:
            candidates = [row for row in self.runtime.history() if row.get("task_id") == run_id]
            if candidates:
                self._refresh_task_history()
                if run_id in self.task_history_tree.get_children(""):
                    self.task_history_tree.selection_set(run_id)
        self._open_history_clicked()
        self.after(250, lambda: self._smoke_wait_for_history(page_index, pages, time.monotonic() + 45.0))

    def _smoke_wait_for_history(self, page_index: int, pages: list[str], deadline: float) -> None:
        if self.busy and time.monotonic() < deadline:
            self.after(150, lambda: self._smoke_wait_for_history(page_index, pages, deadline))
            return
        if self.busy:
            self.smoke_errors.append("历史结果恢复 smoke 超过 45 秒仍在运行。")
            self.cancel_work()
            return
        if self.corpus is None or not self.results_by_method:
            self.smoke_errors.append("历史结果恢复后没有可用语料和方法结果。")
            self.after(200, lambda: self._smoke_step(page_index + 1))
            return
        self.ui_smoke_dir.mkdir(parents=True, exist_ok=True)
        project_path = self.ui_smoke_dir / f"project-roundtrip-{uuid.uuid4().hex[:8]}"
        try:
            from .runtime import save_project
            save_project(project_path, self.corpus, self.results_by_method,
                         settings=self._settings_snapshot(), runtime_metadata=self.result_runtime_metadata)
            self._open_project_path(project_path)
            self.after(250, lambda: self._smoke_wait_for_project(page_index, pages, time.monotonic() + 45.0))
        except Exception as exc:
            self.smoke_errors.append(f"UI smoke 项目保存失败：{type(exc).__name__}: {exc}")
            self.after(200, lambda: self._smoke_step(page_index + 1))

    def _smoke_wait_for_project(self, page_index: int, pages: list[str], deadline: float) -> None:
        if self.busy and time.monotonic() < deadline:
            self.after(150, lambda: self._smoke_wait_for_project(page_index, pages, deadline))
            return
        if self.busy:
            self.smoke_errors.append("项目重新打开 smoke 超过 45 秒仍在运行。")
            self.cancel_work()
            return
        self.show_page("results")
        self._grab_smoke("08-project-restored-results.png")
        self.show_page("tasks")
        canvas = getattr(self.page_widgets["tasks"], "_parent_canvas", None)
        if canvas is not None:
            canvas.yview_moveto(1.0)
        self.after(300, lambda: self._smoke_capture_tasks_bottom(page_index, pages))

    def _smoke_capture_tasks_bottom(self, page_index: int, pages: list[str]) -> None:
        self._grab_smoke("09-tasks-history-presets.png")
        self._open_log_viewer()
        self.after(300, lambda: self._smoke_capture_log_viewer(page_index, pages))

    def _smoke_capture_log_viewer(self, page_index: int, pages: list[str]) -> None:
        window = getattr(self, "log_window", None)
        if window is not None and window.winfo_exists():
            try:
                self.update_idletasks()
                from PIL import ImageGrab
                image = ImageGrab.grab(window=int(window.winfo_id()))
                image.save(self.ui_smoke_dir / "10-log-viewer.png")
                self.smoke_capture_count += 1
            except Exception as exc:
                self.smoke_errors.append(f"10-log-viewer.png: {type(exc).__name__}: {exc}")
            window.destroy()
        self.after(250, lambda: self._smoke_step(page_index + 1))


def np_isfinite(value: Any) -> bool:
    try:
        import math
        return math.isfinite(float(value))
    except Exception:
        return False


def pd_concat_safe(frames: list[Any], axis: int = 0):
    import pandas as pd
    return pd.concat(frames, axis=axis, ignore_index=True)

