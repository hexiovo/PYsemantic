"""Small, clock-injectable stage ETA model for the desktop task runner.

The model keeps a stable estimate until a stage starts, learns immediately from
work observed in the current run, and only persists completed stage samples.
It deliberately makes no claim that opaque estimator fits have callback-level
progress.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import median
from time import monotonic
from typing import Any, Callable, Iterable


STAGE_LABELS = {
    "worker-startup": "启动后台分析进程",
    "load-corpus": "读取与清理语料",
    "tokenize": "中文分词与停词处理",
    "term-matrix": "构建词项矩阵",
    "bge-model-load": "加载本地 BGE 模型",
    "bge-encode": "BGE 分词与分块编码",
    "char-fit": "拟合字符 TF-IDF",
    "lsa-fit": "拟合 LSA 降维",
    "nmf-fit": "拟合 NMF 主题模型",
    "lda-fit": "拟合 LDA 主题模型",
    "cooc-ppmi": "统计窗口共现并计算 PPMI",
    "word2vec-scan": "扫描 word2vec 文件",
    "word2vec-average": "生成 word2vec 文档表示",
    "document-metrics": "计算文档指标",
    "similarity": "计算完整相似度矩阵",
    "neighbors": "分块搜索近邻",
    "kmeans": "拟合 KMeans 聚类",
    "silhouette": "计算轮廓系数",
    "result-assembly": "整理分析结果",
    "result-save": "安全保存分析结果",
    "result-load": "载入分析结果",
    "project-save": "保存项目",
    "project-load": "读取并校验项目",
    "cancel-probe": "检查任务进程可中断性",
    "export-tables": "导出结果表格与矩阵",
    "export-report": "生成离线报告",
    "export-xlsx": "生成 XLSX 工作簿",
}


@dataclass(frozen=True)
class StageEstimate:
    stage_id: str
    label: str
    initial_seconds: float
    work_total: float | None = None
    unit: str = "工作量"
    method: str = ""
    cost_key: str = ""
    source: str = "初始估计"
    history_samples: int = 0


class EtaTracker:
    """ETA state machine. `clock` may be injected for deterministic checks."""

    def __init__(
        self,
        plan: Iterable[StageEstimate | dict[str, Any]],
        *,
        clock: Callable[[], float] = monotonic,
        history: Iterable[dict[str, Any]] = (),
    ) -> None:
        self.clock = clock
        self.started_at = float(clock())
        self.plan = [self._coerce_step(item) for item in plan]
        if not self.plan:
            self.plan = [StageEstimate("worker-startup", STAGE_LABELS["worker-startup"], 1.0)]
        self.history = [dict(item) for item in history]
        self.index = 0
        self.stage_id = self.plan[0].stage_id
        self.stage_label = self.plan[0].label
        self.stage_started_at = self.started_at
        self.stage_work_done = 0.0
        self.stage_work_total = self.plan[0].work_total
        self.stage_unit = self.plan[0].unit
        self.stage_initial_seconds = self._historical_or_initial(self.plan[0])[0]
        self.stage_source, self.history_samples = self._historical_or_initial(self.plan[0])[1:]
        self.completed: list[dict[str, Any]] = []
        self._stage_complete = False
        self._observation_count = 0
        self._terminal_at: float | None = None
        self.terminal: str | None = None
        self._sequence = -1
        self._run_id: str | None = None
        self._saw_work = False
        self.current_method = self.plan[0].method
        self.method_index: int | None = None
        self.method_total: int | None = None

    @staticmethod
    def _coerce_step(item: StageEstimate | dict[str, Any]) -> StageEstimate:
        if isinstance(item, StageEstimate):
            return item
        stage_id = str(item.get("stage_id", "worker-startup"))
        return StageEstimate(
            stage_id=stage_id,
            label=str(item.get("label") or STAGE_LABELS.get(stage_id, stage_id)),
            initial_seconds=max(0.2, float(item.get("initial_seconds", 1.0))),
            work_total=None if item.get("work_total") is None else max(0.0, float(item["work_total"])),
            unit=str(item.get("unit", "工作量")),
            method=str(item.get("method", "")),
            cost_key=str(item.get("cost_key", "")),
        )

    def _historical_or_initial(self, step: StageEstimate) -> tuple[float, str, int]:
        compatible = [row for row in self.history
                      if row.get("method") == step.method
                      and row.get("stage_id") == step.stage_id
                      and row.get("cost_key") == step.cost_key
                      and float(row.get("elapsed_seconds", 0) or 0) > 0]
        if compatible:
            # Work-bearing stages scale by normalized per-unit duration. Opaque
            # fits use cost_key (including size and fit parameters) and exact runs.
            per_unit = [float(row["elapsed_seconds"]) / max(float(row.get("work_total") or 0), 1.0)
                        for row in compatible if row.get("work_total") is not None]
            if step.work_total is not None and per_unit:
                return max(0.2, median(per_unit) * max(step.work_total, 1.0)), "成功历史", len(compatible)
            return max(0.2, median(float(row["elapsed_seconds"]) for row in compatible)), "成功历史", len(compatible)
        return max(0.2, float(step.initial_seconds)), "初始估计", 0

    def accept(self, event: dict[str, Any]) -> bool:
        """Apply one structured event, rejecting stale runs/sequences/terminals."""
        run_id = event.get("run_id")
        if self._run_id is None and run_id is not None:
            self._run_id = str(run_id)
        if run_id is not None and self._run_id is not None and str(run_id) != self._run_id:
            return False
        sequence = event.get("seq")
        if sequence is not None:
            try:
                sequence = int(sequence)
            except (TypeError, ValueError):
                return False
            if sequence <= self._sequence:
                return False
            self._sequence = sequence
        if self.terminal is not None:
            return False
        kind = str(event.get("type", ""))
        try:
            now = float(event["monotonic"]) if event.get("monotonic") is not None else float(self.clock())
        except (TypeError, ValueError):
            now = float(self.clock())
        if kind == "stage-start":
            event_stage = str(event.get("stage_id", self.stage_id))
            event_method = str(event.get("method", ""))
            if (not self._stage_complete and event_stage == self.stage_id
                    and (not event_method or not self.current_method or event_method == self.current_method)):
                # Parent and child may both report the same startup boundary.
                # Keep the earlier parent timestamp so elapsed time cannot go backward.
                if event.get("label"):
                    self.stage_label = str(event["label"])
                if event.get("work_total") is not None:
                    total = max(self.stage_work_done, float(event["work_total"]))
                    self.stage_work_total = total
                    if self.index < len(self.plan):
                        self.plan[self.index] = replace(self.plan[self.index], work_total=total,
                                                       unit=str(event.get("unit") or self.plan[self.index].unit))
                if event_method:
                    self.current_method = event_method
                if event.get("method_index") is not None:
                    try:
                        self.method_index = int(event["method_index"])
                        self.method_total = int(event.get("method_total", 0)) or None
                    except (TypeError, ValueError):
                        self.method_index = self.method_total = None
            else:
                self._enter_stage(event, now)
        elif kind == "stage-progress":
            if event.get("stage_id") != self.stage_id:
                return False
            if event.get("work_done") is not None:
                updated = max(self.stage_work_done, float(event["work_done"]))
                if updated > self.stage_work_done:
                    self._observation_count += 1
                self.stage_work_done = updated
                self._saw_work = self.stage_work_done > 0
            if event.get("work_total") is not None:
                total = float(event["work_total"])
                self.stage_work_total = max(self.stage_work_done, total)
            if event.get("unit"):
                self.stage_unit = str(event["unit"])
        elif kind == "stage-complete":
            if event.get("stage_id") != self.stage_id:
                return False
            done = event.get("work_done")
            total = event.get("work_total")
            if done is not None:
                updated = max(self.stage_work_done, float(done))
                if updated > self.stage_work_done:
                    self._observation_count += 1
                self.stage_work_done = updated
            if total is not None:
                self.stage_work_total = max(self.stage_work_done, float(total))
            self.completed.append({
                "stage_id": self.stage_id,
                "label": self.stage_label,
                "method": self.current_method,
                "cost_key": self.plan[self.index].cost_key if self.index < len(self.plan) else "",
                "elapsed_seconds": max(0.0, now - self.stage_started_at),
                "work_done": self.stage_work_done,
                "work_total": self.stage_work_total,
                "successful": True,
            })
            self._stage_complete = True
        elif kind == "stage-skipped":
            if event.get("stage_id") != self.stage_id:
                self._enter_stage(event, now)
            self.completed.append({"stage_id": self.stage_id, "skipped": True, "elapsed_seconds": 0.0,
                                   "method": self.current_method, "successful": True})
            self._stage_complete = True
        elif kind == "stage-failed":
            if event.get("stage_id") != self.stage_id:
                self._enter_stage(event, now)
            self.completed.append({"stage_id": self.stage_id, "elapsed_seconds": max(0.0, now - self.stage_started_at),
                                   "method": self.current_method,
                                   "cost_key": self.plan[self.index].cost_key if self.index < len(self.plan) else "",
                                   "work_done": self.stage_work_done, "work_total": self.stage_work_total,
                                   "successful": False})
            self._stage_complete = True
        elif kind == "method-terminal":
            if str(event.get("status")) in {"failed", "cancelled"}:
                failed_method = str(event.get("method", self.current_method))
                self.plan = self.plan[:self.index + 1] + [step for step in self.plan[self.index + 1:]
                                                          if step.method != failed_method]
        elif kind == "terminal":
            status = str(event.get("status", "failed"))
            self.terminal = status
            self._terminal_at = now
            if status == "success":
                self.stage_work_done = self.stage_work_total or self.stage_work_done
                self._stage_complete = True
            return True
        return True

    def _enter_stage(self, event: dict[str, Any], now: float) -> None:
        stage_id = str(event.get("stage_id", self.stage_id))
        method = str(event.get("method", ""))
        search_from = self.index + (1 if self._stage_complete else 0)
        found = next((i for i in range(search_from, len(self.plan))
                      if self.plan[i].stage_id == stage_id and (not method or not self.plan[i].method or self.plan[i].method == method)), None)
        if found is None:
            # Unknown stages still receive a finite estimate and never inherit
            # progress or the previous method's rate.
            step = StageEstimate(stage_id, str(event.get("label") or STAGE_LABELS.get(stage_id, stage_id)),
                                 max(0.5, float(event.get("initial_seconds", 2.0))),
                                 None if event.get("work_total") is None else float(event["work_total"]),
                                 str(event.get("unit", "工作量")), method)
            self.plan.insert(search_from, step)
            self.index = search_from
        else:
            self.index = found
            step = self.plan[found]
        if event.get("work_total") is not None:
            step = replace(step, work_total=max(0.0, float(event["work_total"])),
                           unit=str(event.get("unit") or step.unit))
            self.plan[self.index] = step
        self.stage_id = stage_id
        self.stage_label = str(event.get("label") or step.label)
        self.current_method = method or step.method
        if event.get("method_index") is not None:
            try:
                self.method_index = int(event["method_index"])
                self.method_total = int(event.get("method_total", 0)) or None
            except (TypeError, ValueError):
                self.method_index = self.method_total = None
        self.stage_started_at = now
        self._stage_complete = False
        self._observation_count = 0
        self.stage_work_done = 0.0
        self.stage_work_total = step.work_total
        if event.get("work_total") is not None:
            self.stage_work_total = max(0.0, float(event["work_total"]))
        self.stage_unit = str(event.get("unit") or step.unit)
        initial = self._historical_or_initial(step)
        self.stage_initial_seconds, self.stage_source, self.history_samples = initial
        self._saw_work = False

    def _stage_remaining(self, now: float) -> tuple[float, bool, str, int, float, float]:
        elapsed = max(0.0, now - self.stage_started_at)
        if self._stage_complete:
            return 0.0, False, "阶段已完成", max(1, int(self.stage_work_done)), 0.0, 0.0
        if self.stage_work_total is not None and self.stage_work_total > 0 and self.stage_work_done > 0 and elapsed > 0:
            rate = self.stage_work_done / elapsed
            remaining = max(0.0, self.stage_work_total - self.stage_work_done) / max(rate, 1e-9)
            source = "本次实测"
            samples = max(1, self._observation_count)
            # A broad uncertainty interval is useful early and shrinks as more
            # units are observed; the central value remains hand-auditable.
            relative = max(0.15, min(0.75, 2.0 / max(self._observation_count, 1)))
            low = max(0.0, remaining * (1.0 - relative))
            high = remaining * (1.0 + relative)
            return remaining, False, source, samples, low, high
        if elapsed < self.stage_initial_seconds:
            remaining = self.stage_initial_seconds - elapsed
            return remaining, False, self.stage_source, self.history_samples, remaining * 0.5, remaining * 2.0
        # Opaque fit exceeded the initial budget. Keep a positive and gently
        # growing estimate, mark it low confidence, and retain the true stage.
        remaining = max(1.0, elapsed * 0.2, self.stage_initial_seconds * 0.25)
        return remaining, True, "初始估计已超时 · 低置信度", self.history_samples, max(0.5, remaining * 0.25), remaining * 3.0

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        current = float(self._terminal_at if self._terminal_at is not None else (self.clock() if now is None else now))
        total_elapsed = max(0.0, current - self.started_at)
        stage_elapsed = max(0.0, current - self.stage_started_at)
        if self.terminal == "success":
            remaining, overrun, source, samples, low, high = 0.0, False, "已完成", self._observation_count, 0.0, 0.0
            task_remaining: float | None = 0.0
        elif self.terminal is not None:
            remaining, overrun, source, samples, low, high = 0.0, False, self.terminal, self._observation_count, 0.0, 0.0
            task_remaining = None
        else:
            remaining, overrun, source, samples, low, high = self._stage_remaining(current)
            later = 0.0
            for step in self.plan[self.index + 1:]:
                later += self._historical_or_initial(step)[0]
            task_remaining = remaining + later
        work = None
        if self.stage_work_total is not None:
            work = {"done": self.stage_work_done, "total": self.stage_work_total, "unit": self.stage_unit}
        observed_rate = None
        if self.stage_work_done > 0 and stage_elapsed > 0:
            observed_rate = self.stage_work_done / stage_elapsed
        return {
            "stage_id": self.stage_id,
            "stage_label": self.stage_label,
            "current_method": self.current_method,
            "method_index": self.method_index,
            "method_total": self.method_total,
            "stage_elapsed_seconds": stage_elapsed,
            "task_elapsed_seconds": total_elapsed,
            "stage_remaining_seconds": remaining,
            "task_remaining_seconds": task_remaining,
            "stage_remaining_range": [low, high],
            "source": source,
            "sample_count": samples,
            "overrun": overrun,
            "work": work,
            "observed_rate": observed_rate,
            "observed_rate_unit": f"{self.stage_unit}/秒" if self.stage_unit else "工作量/秒",
            "terminal": self.terminal,
        }

    def finish(self, status: str) -> None:
        if self.terminal is None:
            self.terminal = status
            self._terminal_at = float(self.clock())


def estimate_stage_plan(methods: list[str], doc_count: int, char_count: int, configs: dict[str, dict[str, Any]],
                        *, history: Iterable[dict[str, Any]] = ()) -> list[StageEstimate]:
    """Create a finite, method-aware cold estimate without running a benchmark."""
    n = max(1, int(doc_count))
    chars = max(n, int(char_count))
    history_rows = list(history)
    plan: list[StageEstimate] = []

    def add(method: str, stage_id: str, seconds: float, units: float | None = None, unit: str = "工作量",
            detail: str = "") -> None:
        cfg = configs.get(method, {})
        signature_values = [str(cfg.get("segmentation", "jieba精确模式")), str(cfg.get("stopwords", "保守内置"))]
        def resource_key(raw_path: Any) -> str:
            value = str(raw_path or "").strip()
            if not value:
                return "empty"
            from pathlib import Path
            path = Path(value)
            try:
                stat = path.stat()
                return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{int(path.is_file())}"
            except OSError:
                return f"{value}:missing"

        if stage_id == "tokenize":
            signature_values.extend([resource_key(cfg.get("user_dictionary_path")),
                                     resource_key(cfg.get("custom_stopwords_path"))])
        # Cleaning and weighting also affect downstream dimensions and cost.
        if stage_id != "worker-startup":
            signature_values.extend(str(cfg.get(key, default)) for key, default in [
                ("min_token_length", 1), ("remove_numbers", False), ("remove_urls", False),
                ("case_mode", "方法默认"), ("tfidf_sublinear_tf", False), ("tfidf_smooth_idf", True),
                ("nmf_tol", .0001), ("kmeans_tol", .0001), ("lda_mean_change_tol", .001),
                ("lda_perp_tol", .1), ("lda_evaluate_every", -1),
                ("lda_doc_topic_prior", None), ("lda_topic_word_prior", None)])
        if stage_id in {"term-matrix", "char-fit"}:
            signature_values.extend([str(cfg.get("max_features", 4000)), str(cfg.get("min_frequency", 1))])
        if stage_id not in {"worker-startup", "tokenize", "bge-model-load", "bge-encode"}:
            signature_values.extend(str(cfg.get(key, default)) for key, default in [
                ("max_df", 1.0), ("word_ngram_min", 1), ("word_ngram_max", 1),
                ("char_ngram_min", 1), ("char_ngram_max", 2), ("random_state", 42)])
        if stage_id == "neighbors":
            signature_values.append(str(cfg.get("neighbors_k", 10)))
        if stage_id in {"nmf-fit", "lda-fit"}:
            signature_values.append(str(cfg.get("nmf_max_iter" if stage_id == "nmf-fit" else "lda_max_iter", 300 if stage_id == "nmf-fit" else 20)))
        if stage_id in {"lsa-fit", "nmf-fit", "lda-fit"}:
            signature_values.extend([str(cfg.get("topic_count", 8)), str(cfg.get("max_features", 4000)),
                                     str(cfg.get("min_frequency", 1))])
        elif stage_id == "cooc-ppmi":
            signature_values.extend([str(cfg.get("cooc_window", 5)), str(cfg.get("cooc_vocab_size", 400))])
        elif stage_id == "word2vec-scan":
            signature_values.append(resource_key(cfg.get("word2vec_path")))
        elif stage_id == "word2vec-average":
            signature_values.append(str(bool(cfg.get("word2vec_tfidf_weight", True))))
        elif stage_id == "kmeans":
            signature_values.extend([str(cfg.get("clusters", 5)), str(cfg.get("topic_count", 8))])
        elif stage_id == "similarity":
            signature_values.append(str(cfg.get("compute_similarity", True)))
        if units is None or unit in {"拟合", "计算"}:
            signature_values.extend([str(n // 25), str(chars // 10_000)])
        cost_key = f"{method}:{stage_id}:{':'.join(signature_values)}{detail}"
        plan.append(StageEstimate(stage_id, STAGE_LABELS.get(stage_id, stage_id), max(0.3, seconds), units, unit,
                                  method, cost_key))

    for method in methods:
        cfg = configs.get(method, {})
        add(method, "worker-startup", 1.2)
        add(method, "tokenize", max(0.5, n * 0.0015 + chars * 0.00004), n, "文档")
        add(method, "term-matrix", max(0.4, n * 0.001 + min(int(cfg.get("max_features", 4000)), 20_000) * chars * 0.00000004), n, "文档")
        if method == "bge":
            add(method, "bge-model-load", 2.5)
            estimated_chunks = max(n, int(chars / 1_500) + n)
            add(method, "bge-encode", max(1.0, estimated_chunks * 0.15 + n * 0.02), estimated_chunks, "估算分块")
        elif method == "tfidf-char":
            add(method, "char-fit", max(0.5, chars * 0.00004 + n * 0.002), n, "文档")
        elif method == "lsa":
            dims = max(2, int(cfg.get("topic_count", 8)))
            add(method, "lsa-fit", max(0.8, n * dims * 0.003 + chars * dims * 0.00004), 1, "拟合")
        elif method == "nmf":
            dims = max(2, int(cfg.get("topic_count", 8)))
            add(method, "nmf-fit", max(1.0, (n * dims * 0.007 + chars * dims * 0.00009) * int(cfg.get("nmf_max_iter", 300)) / 300), 1, "拟合")
        elif method == "lda":
            dims = max(2, int(cfg.get("topic_count", 8)))
            add(method, "lda-fit", max(1.0, (n * dims * 0.004 + chars * dims * 0.00005) * int(cfg.get("lda_max_iter", 20)) / 20), 1, "拟合")
        elif method == "cooc-ppmi":
            add(method, "cooc-ppmi", max(0.5, chars * 0.0001), chars, "字符位置")
        elif method == "word2vec":
            byte_size = 1.0
            try:
                from pathlib import Path
                byte_size = max(1, Path(str(cfg.get("word2vec_path", ""))).stat().st_size)
            except OSError:
                pass
            add(method, "word2vec-scan", max(0.5, byte_size / (10 * 1024 * 1024)), byte_size, "字节")
            add(method, "word2vec-average", max(0.4, chars * 0.000025), n, "文档")
        add(method, "document-metrics", max(0.2, n * 0.0004), n, "文档")
        if bool(cfg.get("compute_similarity", True)) and n <= 1_500:
            add(method, "similarity", max(0.3, n * n / 800_000), n * n, "文档对")
        if bool(cfg.get("compute_neighbors", True)):
            add(method, "neighbors", max(0.5, n * n / 450_000), n * n, "文档对")
        if bool(cfg.get("compute_clusters", True)):
            add(method, "kmeans", max(0.6, n * max(2, int(cfg.get("clusters", 5))) * 0.003), 1, "拟合")
            if bool(cfg.get("compute_silhouette", True)):
                add(method, "silhouette", max(0.3, min(n, 2_000) ** 2 / 2_000_000), 1, "计算")
        add(method, "result-assembly", max(0.2, n * 0.0003), n, "文档")
        add(method, "result-save", max(0.5, n * 0.0005 + chars * 0.00001), n, "文档")
        add(method, "result-load", max(0.5, n * 0.0008 + chars * 0.00002), n, "文档")
    return plan

