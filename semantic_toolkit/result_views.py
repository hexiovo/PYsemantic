"""Paged result tables and source-linked profiles for the desktop application."""
from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable
import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk
import pandas as pd

from .exploration import build_profiles, filter_rows, page_rows, profile_table


class PagedTable:
    def __init__(self, app: Any, parent: Any, *, open_row: Callable | None = None,
                 select_row: Callable | None = None, height: int = 8):
        self.app = app
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.pack(fill="both", expand=True)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)
        self.tree = app._make_tree(self.frame)
        self.tree.configure(height=height, selectmode="browse")
        self.tree.grid(row=1, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        vertical.grid(row=1, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(self.frame, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=2, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        bar = ctk.CTkFrame(self.frame, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(4, 6))
        self.previous = app._button(bar, "上一页", lambda: self.move(-1), width=78)
        self.previous.pack(side="left")
        self.next = app._button(bar, "下一页", lambda: self.move(1), width=78)
        self.next.pack(side="left", padx=4)
        self.status = app._label(bar, "0 条", width=220)
        self.status.pack(side="left", padx=5)
        self.size = tk.StringVar(value="50")
        size_box = app._combo(bar, self.size, ["50", "100", "300"], width=80)
        size_box.configure(state="readonly", command=lambda _: self.move(0, reset=True))
        size_box.pack(side="right", padx=4)
        app._label(bar, "每页").pack(side="right")
        if open_row:
            self.open_button = app._button(bar, "查看原文", lambda: self.activate(open_row), width=94)
            self.open_button.pack(side="right", padx=4)
            self.tree.bind("<Double-1>", lambda event: self._double_click(event, open_row))
            self.tree.bind("<Return>", lambda _: self.activate(open_row))
        if select_row:
            self.tree.bind("<<TreeviewSelect>>", lambda _: self.activate(select_row))
        self.data = pd.DataFrame()
        self.visible = pd.DataFrame()
        self.page = 0

    def _double_click(self, event: Any, callback: Callable) -> None:
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
            self.activate(callback)

    def activate(self, callback: Callable) -> None:
        if self.app.busy:
            return
        selected = self.tree.selection()
        if selected:
            index = int(selected[0])
            if index < len(self.visible):
                callback(self.visible.iloc[index])

    def set_data(self, frame: pd.DataFrame) -> None:
        self.data = frame.reset_index(drop=True)
        self.move(0, reset=True)

    def move(self, delta: int, *, reset: bool = False) -> None:
        self.visible, self.page, pages = page_rows(self.data, 0 if reset else self.page + delta, int(self.size.get()))
        self.tree.delete(*self.tree.get_children())
        columns = list(self.data.columns)
        self.tree.configure(columns=columns)
        for column in columns:
            width = 300 if column in {"关键词", "原文摘要", "代表文档ID"} else 145
            self.tree.heading(column, text=column)
            self.tree.column(column, width=width, minwidth=75, stretch=True)
        for index, row in enumerate(self.visible.itertuples(index=False, name=None)):
            values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
            self.tree.insert("", "end", iid=str(index), values=values)
        first = self.page * int(self.size.get()) + 1 if len(self.data) else 0
        last = first + len(self.visible) - 1 if len(self.visible) else 0
        self.status.configure(text=f"{first}–{last} / {len(self.data):,} 条 · {self.page + 1}/{pages} 页")
        self.previous.winfo_children()[0].configure(state="normal" if self.page else "disabled")
        self.next.winfo_children()[0].configure(state="normal" if self.page + 1 < pages else "disabled")


class ResultsExplorer:
    def __init__(self, app: Any, tabs: Any, *, ink: str, white: str, font: tuple, small_font: tuple):
        self.app, self.tabs = app, tabs
        self.ink, self.white, self.font, self.small_font = ink, white, font, small_font
        self.result = None
        self.documents = {}
        self.names = {}
        self.profiles = []
        self.generation = 0
        self.profile_queue = Queue()
        self.profile_cache = {}
        self.tables = {}
        self.filters = {}
        self.raw_windows = []
        for tab, key in (("文档指标", "metrics"), ("词频", "frequency"), ("近邻", "neighbors")):
            host = tabs.tab(tab)
            controls = ctk.CTkFrame(host, fg_color="transparent")
            controls.pack(fill="x", padx=5, pady=4)
            query = tk.StringVar()
            app._label(controls, "词语" if key == "frequency" else "文档 ID").pack(side="left")
            entry = app._entry(controls, query, width=170)
            entry.pack(side="left", padx=5)
            entry.bind("<Return>", lambda _, k=key: self.apply_filter(k))
            cluster = tk.StringVar(value="全部")
            cluster_box = None
            if key != "frequency":
                app._label(controls, "源文档簇" if key == "neighbors" else "簇").pack(side="left")
                cluster_box = app._combo(controls, cluster, ["全部"], width=110)
                cluster_box.configure(state="readonly")
                cluster_box.pack(side="left", padx=5)
            app._button(controls, "筛选", lambda k=key: self.apply_filter(k), width=72).pack(side="left", padx=3)
            app._button(controls, "重置", lambda k=key: self.reset_filter(k), width=72).pack(side="left", padx=3)
            lower, upper = tk.StringVar(), tk.StringVar()
            if key == "neighbors":
                range_bar = ctk.CTkFrame(host, fg_color="transparent")
                range_bar.pack(fill="x", padx=5, pady=(0, 3))
                app._label(range_bar, "相似度范围").pack(side="left")
                for variable, label in ((lower, "至"), (upper, "留空不限；ID 匹配任一端，簇筛选源文档。")):
                    field = app._entry(range_bar, variable, width=75)
                    field.pack(side="left", padx=5)
                    field.bind("<Return>", lambda _: self.apply_filter("neighbors"))
                    app._label(range_bar, label, font=small_font).pack(side="left")
            self.filters[key] = (query, cluster, lower, upper, cluster_box)
            self.tables[key] = PagedTable(app, host, open_row=self.open_row if key != "frequency" else None)
        self._build_profiles(tabs.tab("主题与聚类"))
        app.after(100, self._poll_profiles)

    def _build_profiles(self, host: Any) -> None:
        row = ctk.CTkFrame(host, fg_color="transparent")
        row.pack(fill="x", pady=4)
        self.kind = tk.StringVar(value="聚类画像")
        picker = self.app._combo(row, self.kind, ["聚类画像", "主题画像", "主题词表"], width=140)
        picker.configure(state="readonly", command=lambda _: self.show_profiles())
        picker.pack(side="left", padx=4)
        self.app._label(row, "选中组查看画像、代表文本和全部成员", font=self.small_font).pack(side="left", padx=6)
        self.profile_status = self.app._label(host, "尚未运行分析。", font=self.small_font,
                                               anchor="w", wraplength=760, justify="left")
        self.profile_status.pack(fill="x", padx=6, pady=(0, 4))
        self.profile_table = PagedTable(self.app, host, select_row=self.select_profile, height=5)

    def clear(self) -> None:
        self.generation += 1
        self.result = None
        self.documents = {}
        self.profiles = []
        self.profile_cache.clear()
        for window in self.raw_windows:
            if window.winfo_exists():
                window.destroy()
        self.raw_windows.clear()
        self._close_detail()
        for table in self.tables.values():
            table.set_data(pd.DataFrame())
        self.profile_table.set_data(pd.DataFrame())
        self.profile_status.configure(text="尚未运行分析。")

    def set_result(self, result: Any, corpus: Any, names: dict) -> None:
        if result is self.result and corpus is getattr(self, "corpus", None):
            return
        self.generation += 1
        generation = self.generation
        self._close_detail()
        for window in self.raw_windows:
            if window.winfo_exists():
                window.destroy()
        self.raw_windows.clear()
        self.result, self.corpus, self.names = result, corpus, names
        self.documents = {doc.doc_id: doc for doc in corpus.documents} if corpus is not None else {}
        clusters = sorted(result.cluster_labels["簇"].unique()) if not result.cluster_labels.empty else []
        self.cluster_ids = {f"簇 {int(value)}": set(result.cluster_labels.loc[result.cluster_labels["簇"] == value, "文档ID"].astype(str))
                            for value in clusters}
        self.cluster_ids["未分簇"] = set(result.doc_ids) - set().union(*self.cluster_ids.values())
        for key, (_, _, _, _, picker) in self.filters.items():
            if picker is not None:
                picker.configure(values=["全部", *self.cluster_ids])
            self.reset_filter(key)
        cached = self.profile_cache.get(result.method)
        if cached is not None and cached[0] is result:
            self.profiles = cached[1]
            self.show_profiles()
            return
        self.profiles = []
        self.profile_status.configure(text="正在整理画像，可继续浏览文档与近邻…")
        self.profile_table.set_data(pd.DataFrame())

        def calculate() -> None:
            try:
                profiles = build_profiles(result)
                self.profile_queue.put((generation, result, profiles, None))
            except Exception as exc:
                self.profile_queue.put((generation, result, [], str(exc)))
        Thread(target=calculate, daemon=True, name="result-profiles").start()

    def _poll_profiles(self) -> None:
        try:
            while True:
                generation, result, profiles, error = self.profile_queue.get_nowait()
                if generation != self.generation or result is not self.result:
                    continue
                if error:
                    self.profile_status.configure(text=f"画像生成失败：{error}")
                else:
                    self.profile_cache[result.method] = (result, profiles)
                    self.profiles = profiles
                    self.show_profiles()
        except Empty:
            pass
        self.app.after(100, self._poll_profiles)

    def reset_filter(self, key: str) -> None:
        query, cluster, lower, upper, _ = self.filters[key]
        query.set("")
        cluster.set("全部")
        lower.set("")
        upper.set("")
        self.apply_filter(key)

    def apply_filter(self, key: str) -> None:
        if self.result is None:
            return
        query, cluster, lower, upper, _ = self.filters[key]
        source = {"metrics": self.result.document_metrics, "frequency": self.result.word_frequencies,
                  "neighbors": self.result.nearest_neighbors}[key]
        try:
            low = float(lower.get()) if lower.get().strip() else None
            high = float(upper.get()) if upper.get().strip() else None
            frame = filter_rows(source, query=query.get(), cluster_ids=self.cluster_ids.get(cluster.get()),
                                minimum=low, maximum=high)
        except ValueError as exc:
            messagebox.showerror("筛选条件无效", f"请输入有效的相似度范围。\n{exc}", parent=self.app)
            return
        if key == "metrics":
            lookup = dict(zip(self.result.cluster_labels["文档ID"], self.result.cluster_labels["簇"]))
            frame = frame.copy()
            frame.insert(1, "簇", [f"簇 {int(lookup[value])}" if value in lookup else "未分簇" for value in frame["文档ID"]])
        self.tables[key].set_data(frame)

    def show_profiles(self) -> None:
        self._close_detail()
        if self.result is None:
            return
        if self.kind.get() == "主题词表":
            self.shown_profiles = []
            self.profile_table.set_data(self.result.topic_terms)
            self.profile_status.configure(text=self.result.topic_weight_kind or "当前方法没有主题词矩阵。")
            return
        clusters = self.kind.get() == "聚类画像"
        self.shown_profiles = [profile for profile in self.profiles if (profile.kind == "聚类") == clusters]
        self.profile_table.set_data(profile_table(self.shown_profiles, self.names))
        if self.shown_profiles:
            note = "点击任意组查看详情；双击成员或按 Enter 查看原文。"
            if clusters:
                note += " 未分簇文档可在“文档指标”中筛选查看。"
            self.profile_status.configure(text=note)
        else:
            self.profile_status.configure(text=("没有可用的聚类。" + str(self.result.clustering_summary.get("轮廓系数说明", "")))
                                          if clusters else "当前方法没有主题画像；可切换到 NMF、LDA 或 LSA 的分析结果。")

    def _close_detail(self) -> None:
        detail = getattr(self, "detail_window", None)
        if detail is not None and detail.winfo_exists():
            detail.destroy()
        self.detail_window = None

    def _window(self, title: str, width: int = 1060, height: int = 680) -> Any:
        window = ctk.CTkToplevel(self.app)
        window.title(title)
        window.configure(fg_color=self.white)
        window.transient(self.app)
        self.app._fit_dialog(window, width, height)
        return window

    def select_profile(self, row: pd.Series) -> None:
        if "组别" not in row:
            return
        profile = next((item for item in self.shown_profiles if item.label == row["组别"]), None)
        if profile is None:
            return
        self._close_detail()
        self.selected_profile = profile
        window = self.detail_window = self._window(f"{profile.label} · 画像与成员")
        header = ctk.CTkFrame(window, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))
        self.app._label(header, f"{profile.label}  ·  {len(profile.member_ids):,} 篇").pack(side="left")
        self.name_var = tk.StringVar(value=self.names.get(profile.key, ""))
        if profile.kind == "聚类":
            self.app._label(header, "自定义名称").pack(side="left", padx=(18, 5))
            self.app._entry(header, self.name_var, width=240).pack(side="left")
            self.app._button(header, "应用名称", self.rename_cluster, width=104).pack(side="left", padx=5)
        self.detail_name_status = self.app._label(window, "名称修改后请在任务页保存项目；名称仅属于当前方法。",
                                                   font=self.small_font, anchor="w")
        self.detail_name_status.pack(fill="x", padx=14)
        if profile.kind != "聚类":
            self.detail_name_status.configure(text="主题成员按下面注明的规则展示。")
        summary = ctk.CTkTextbox(window, height=165, fg_color=self.white, text_color=self.ink, font=self.font, wrap="word")
        summary.pack(fill="x", padx=12, pady=6)
        lines = ["关键词：" + ("、".join(profile.keywords) or "没有非零词项"), profile.explanation, "", f"代表文本（最多 {profile.representative_count} 篇）："]
        for doc_id in profile.member_ids[:profile.representative_count]:
            doc = self.documents.get(doc_id)
            excerpt = " ".join(doc.text.split())[:220] if doc and doc.text else "原文不可用：历史缓存不保存原文，请打开含语料的已保存项目。"
            lines.append(f"{doc_id}：{excerpt}")
        self.app._set_readonly_text(summary, "\n".join(lines))
        member_bar = ctk.CTkFrame(window, fg_color="transparent")
        member_bar.pack(fill="x", padx=12, pady=3)
        self.member_query = tk.StringVar()
        self.app._label(member_bar, "成员 ID").pack(side="left")
        entry = self.app._entry(member_bar, self.member_query, width=200)
        entry.pack(side="left", padx=5)
        entry.bind("<Return>", lambda _: self.filter_members())
        self.app._button(member_bar, "筛选成员", self.filter_members, width=100).pack(side="left")
        holder = ctk.CTkFrame(window, fg_color="transparent")
        holder.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self.members_table = PagedTable(self.app, holder, open_row=self.open_row, height=6)
        self.member_data = pd.DataFrame([
            {"文档ID": doc_id, profile.score_label: score,
             "原文摘要": " ".join(self.documents[doc_id].text.split())[:180] if doc_id in self.documents and self.documents[doc_id].text else "原文不可用",
             "来源": self.documents[doc_id].source if doc_id in self.documents else ""}
            for doc_id, score in zip(profile.member_ids, profile.scores)],
            columns=["文档ID", profile.score_label, "原文摘要", "来源"])
        self.filter_members()

    def filter_members(self) -> None:
        self.members_table.set_data(filter_rows(self.member_data, query=self.member_query.get()))

    def rename_cluster(self) -> None:
        if self.app.busy or self.result is None:
            return
        name = self.name_var.get().strip()
        if len(name) > 80 or any(ord(char) < 32 for char in name):
            messagebox.showerror("名称无效", "名称最多 80 个字符，不能含换行或控制字符。", parent=self.detail_window)
            return
        key = self.selected_profile.key
        if name:
            self.names[key] = name
        else:
            self.names.pop(key, None)
        self.app.result_runtime_metadata.setdefault(self.result.method, {})["cluster_names"] = dict(self.names)
        self.profile_table.set_data(profile_table(self.shown_profiles, self.names))
        self.detail_name_status.configure(text="名称已应用。请到“任务、日志与项目”页保存项目，以便下次恢复。")

    def open_row(self, row: pd.Series) -> None:
        if self.result is None or "文档ID" not in row:
            return
        ids = [str(row["文档ID"])]
        if "近邻ID" in row:
            ids.append(str(row["近邻ID"]))
        window = self._window("原文对照" if len(ids) == 2 else "文档原文")
        self.raw_windows.append(window)
        window.grid_rowconfigure(1, weight=1)
        caption = self.result.method_label
        if "余弦相似度" in row:
            caption += f"  ·  余弦相似度 {float(row['余弦相似度']):.6g}"
        self.app._label(window, caption, anchor="w").grid(row=0, column=0, columnspan=len(ids), sticky="ew", padx=12, pady=8)
        for column, doc_id in enumerate(ids):
            window.grid_columnconfigure(column, weight=1, uniform="documents")
            panel = ctk.CTkFrame(window, fg_color="transparent")
            panel.grid(row=1, column=column, sticky="nsew", padx=10, pady=(0, 10))
            panel.grid_columnconfigure(0, weight=1)
            panel.grid_rowconfigure(1, weight=1)
            doc = self.documents.get(doc_id)
            source = f"来源：{doc.source}\n行／段：{doc.source_row if doc.source_row is not None else '—'}" if doc else "来源不可用"
            metadata = ctk.CTkTextbox(panel, height=100, fg_color=self.white, text_color=self.ink, font=self.small_font, wrap="word")
            metadata.grid(row=0, column=0, sticky="ew", pady=(0, 5))
            self.app._set_readonly_text(metadata, f"文档 ID：{doc_id}\n{source}")
            content = ctk.CTkTextbox(panel, fg_color=self.white, text_color=self.ink, font=self.font, wrap="word",
                                     border_width=1, border_color="#D5DEE8")
            content.grid(row=1, column=0, sticky="nsew")
            self.app._set_readonly_text(content, doc.text if doc and doc.text else
                                       "原文不可用。\n\n历史结果缓存不保存原文。请打开包含语料文本的已保存项目，或重新导入原文并运行分析。")
