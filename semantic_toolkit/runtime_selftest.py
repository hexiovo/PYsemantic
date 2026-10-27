"""Bounded executable integration check for runtime lifecycle behavior."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any
import json
import os
import shutil
import subprocess
import sys
import time
import uuid

from .analysis import AnalysisConfig, METHODS
from .data_io import Corpus, Document
from .eta import EtaTracker, StageEstimate
from .runtime import (RuntimeStore, corpus_to_dict, create_project_from_caches,
                      load_project_full, save_project, terminate_owned_process)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _worker_command(task_dir: Path) -> tuple[list[str], str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--runtime-worker", str(task_dir)], str(Path(sys.executable).parent)
    main_path = Path(__file__).resolve().parent.parent / "main.py"
    return [sys.executable, str(main_path), "--runtime-worker", str(task_dir)], str(main_path.parent)


def _launch_worker(task_dir: Path, request: dict[str, Any], timeout: float = 180.0) -> list[dict[str, Any]]:
    task_dir.mkdir(parents=True, exist_ok=False)
    (task_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    (task_dir / "events.jsonl").write_text("", encoding="utf-8")
    command, cwd = _worker_command(task_dir)
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    with (task_dir / "worker.stdout.log").open("wb") as stdout_handle, \
            (task_dir / "worker.stderr.log").open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=stdout_handle, stderr=stderr_handle,
                                   startupinfo=startupinfo, creationflags=creationflags)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminate_owned_process(process, 5)
            raise TimeoutError(f"worker 超过 {timeout:.0f} 秒未完成：{task_dir.name}")
    events: list[dict[str, Any]] = []
    for line in (task_dir / "events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if code != 0:
        error = {}
        try:
            error = json.loads((task_dir / "error.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        stderr = (task_dir / "worker.stderr.log").read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"worker 退出码 {code}: {error.get('message', '')}\n{error.get('traceback', '')}\n{stderr}")
    return events


def _cancel_probe(root: Path) -> None:
    task_dir = root / "cancel-probe-worker"
    owned = root / "cancel-probe-owned-stage"
    outside = root / "not-owned.txt"
    outside.write_text("must remain", encoding="utf-8")
    task_dir.mkdir()
    request = {"run_id": "cancel-probe", "operation": "runtime-selftest-cancel-probe",
               "method": "self-test", "self_test_only": True, "owned_directory": str(owned)}
    (task_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    events_path = task_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    command, cwd = _worker_command(task_dir)
    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    stdout_handle = (task_dir / "worker.stdout.log").open("wb")
    stderr_handle = (task_dir / "worker.stderr.log").open("wb")
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                               stdout=stdout_handle, stderr=stderr_handle,
                               startupinfo=startupinfo, creationflags=creationflags)
    deadline = time.monotonic() + 30.0
    observed = False
    while time.monotonic() < deadline and process.poll() is None:
        try:
            records = [json.loads(line) for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines()]
        except (OSError, json.JSONDecodeError):
            records = []
        if any(row.get("type") == "stage-start" and row.get("stage_id") == "cancel-probe" for row in records):
            observed = True
            break
        time.sleep(0.03)
    _check(observed, "自有 worker 未进入可取消阶段。")
    _check((owned / "worker-owned.txt").is_file(), "取消样例没有创建声明的自有暂存文件。")
    terminate_owned_process(process, 5)
    stdout_handle.close()
    stderr_handle.close()
    shutil.rmtree(owned, ignore_errors=True)
    shutil.rmtree(task_dir, ignore_errors=True)
    _check(outside.read_text(encoding="utf-8") == "must remain", "取消清理触碰了任务未声明拥有的文件。")


class _Clock:
    def __init__(self, value: float = 0.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


def _eta_checks(store: RuntimeStore) -> None:
    clock = _Clock(0)
    tracker = EtaTracker([
        StageEstimate("stage-a", "阶段 A", 40.0, 100, "单位", "tfidf-word", "same-cost"),
        StageEstimate("stage-b", "阶段 B", 30.0, None, "拟合", "tfidf-word", "fit-cost"),
    ], clock=clock, history=store.eta_history())
    _check(tracker.accept({"type": "stage-start", "stage_id": "stage-a", "method": "tfidf-word",
                           "work_total": 100, "run_id": "eta-run", "seq": 1, "monotonic": 0}),
           "ETA 冷启动阶段没有被接受。")
    cold = tracker.snapshot(now=0)
    _check(cold["stage_remaining_seconds"] > 0 and cold["task_remaining_seconds"] > 0,
           "冷启动没有立即给出有限正 ETA。")
    clock.value = 2
    tracker.accept({"type": "stage-progress", "stage_id": "stage-a", "work_done": 10,
                    "work_total": 100, "run_id": "eta-run", "seq": 2, "monotonic": 2})
    sample_one = tracker.snapshot(now=2)
    _check(abs(sample_one["stage_remaining_seconds"] - 18.0) < 1e-6, "首个观测的剩余时间计算错误。")
    _check(sample_one["sample_count"] == 1, "一个 progress 事件被错误记成多个观测批次。")
    clock.value = 5
    tracker.accept({"type": "stage-progress", "stage_id": "stage-a", "work_done": 20,
                    "work_total": 100, "run_id": "eta-run", "seq": 3, "monotonic": 5})
    sample_two = tracker.snapshot(now=5)
    _check(abs(sample_two["stage_remaining_seconds"] - 20.0) < 1e-6, "两个观测后的剩余时间计算错误。")
    _check(sample_two["sample_count"] == 2, "第二次 progress 观察未增加观测批次。")
    clock.value = 8
    tracker.accept({"type": "stage-complete", "stage_id": "stage-a", "work_done": 100,
                    "work_total": 100, "run_id": "eta-run", "seq": 4, "monotonic": 8})
    _check(abs(tracker.snapshot(now=8)["task_remaining_seconds"] - 30.0) < 1e-6,
           "阶段完成后全任务 ETA 没有准确汇总后续阶段。")
    clock.value = 10
    tracker.accept({"type": "stage-start", "stage_id": "stage-b", "method": "tfidf-word",
                    "run_id": "eta-run", "seq": 5, "monotonic": 10})
    switched = tracker.snapshot(now=10)
    _check(switched["stage_id"] == "stage-b" and switched["stage_elapsed_seconds"] == 0,
           "阶段切换没有重置阶段耗时。")
    _check(not tracker.accept({"type": "stage-progress", "stage_id": "stage-b", "work_done": 5,
                               "run_id": "old-run", "seq": 99, "monotonic": 10}), "旧 run 事件没有被丢弃。")
    clock.value = 12
    tracker.accept({"type": "terminal", "status": "success", "run_id": "eta-run", "seq": 6, "monotonic": 12})
    clock.value = 65
    terminal = tracker.snapshot()
    _check(terminal["task_elapsed_seconds"] == 12 and terminal["task_remaining_seconds"] == 0,
           "成功终态时间没有冻结，或 remaining 不为 0。")
    _check(not tracker.accept({"type": "stage-start", "stage_id": "stage-a", "run_id": "eta-run",
                               "seq": 7, "monotonic": 65}), "终态后的迟到事件没有被丢弃。")

    store.successful_stage({"method": "tfidf-word", "stage_id": "history-stage", "cost_key": "hist-key",
                            "elapsed_seconds": 20.0, "work_total": 100, "successful": True})
    history_tracker = EtaTracker([StageEstimate("history-stage", "历史阶段", 90, 50, "单位",
                                                "tfidf-word", "hist-key")],
                                 clock=_Clock(0), history=store.eta_history())
    _check(history_tracker.snapshot(now=0)["stage_remaining_seconds"] == 10.0,
           "成功历史没有按工作量归一化。")
    over_clock = _Clock(0)
    over = EtaTracker([StageEstimate("opaque", "不透明拟合", 5.0, None, "拟合", "lsa", "opaque-key")],
                      clock=over_clock)
    over.accept({"type": "stage-start", "stage_id": "opaque", "method": "lsa", "run_id": "over", "seq": 1,
                 "initial_seconds": 5.0, "monotonic": 0})
    over_clock.value = 9
    over_snapshot = over.snapshot()
    _check(over_snapshot["overrun"] and over_snapshot["stage_remaining_seconds"] > 0,
           "不透明阶段超估后没有保留正数动态 ETA。")
    cancel_clock = _Clock(0)
    cancelled = EtaTracker([StageEstimate("opaque", "阶段", 4.0)], clock=cancel_clock)
    cancelled.accept({"type": "terminal", "status": "cancelled", "run_id": "cancel", "seq": 1, "monotonic": 3})
    _check(cancelled.snapshot()["task_remaining_seconds"] is None, "取消终态错误显示未来剩余 ETA。")


def run_runtime_self_test(output_directory: str | Path) -> Path:
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = RuntimeStore(output / "test-profile")
    checks: list[str] = []
    workspace_path: Path
    with TemporaryDirectory(prefix="runtime-self-test-", dir=output) as scratch:
        workspace = Path(scratch)
        corpus = Corpus(
            documents=[
                Document("d1", "苹果手机相机拍摄清晰，电池续航持久。", "self-test", 1),
                Document("d2", "安卓手机摄影夜景明亮，电池使用时间很长。", "self-test", 2),
                Document("d3", "足球联赛球队训练，球员传球射门得分。", "self-test", 3),
                Document("d4", "篮球球队比赛训练，球员投篮防守得分。", "self-test", 4),
                Document("d5", "天气预报显示晴天，城市公园适合步行。", "self-test", 5),
                Document("d6", "城市天气温暖，公园绿地适合散步。", "self-test", 6),
            ],
            cleaning_log=[{"来源": "runtime-self-test", "读入行/段": 6}],
            stats={"读入记录": 6, "空值排除": 0, "重复排除": 0, "保留文档": 6},
        )
        results_root = workspace / "results"
        results_root.mkdir()
        configs = [AnalysisConfig(method="tfidf-word", clusters=2, topic_count=2, compute_similarity=True),
                   AnalysisConfig(method="tfidf-char", clusters=2, topic_count=2, compute_similarity=True)]
        caches: list[Path] = []
        worker_events: dict[str, list[dict[str, Any]]] = {}
        task_id = f"runtime-self-test-{uuid.uuid4().hex}"
        profile.begin_task({"task_id": task_id, "kind": "runtime-self-test", "methods": [c.method for c in configs],
                            "status": "running", "result_directories": {}})
        for index, config in enumerate(configs, 1):
            task_dir = workspace / f"worker-{index}"
            cache_dir = results_root / f"{config.method}.cst"
            events = _launch_worker(task_dir, {
                "run_id": f"{task_id}-{index}", "operation": "analyze", "method": config.method,
                "method_index": index, "method_total": len(configs), "config": config.__dict__,
                "corpus": corpus_to_dict(corpus), "result_directory": str(cache_dir),
            })
            worker_events[config.method] = events
            event_types = {(row.get("type"), row.get("stage_id")) for row in events}
            for stage_id in ("worker-startup", "tokenize", "term-matrix", "result-save"):
                _check(any(kind == "stage-start" and stage == stage_id for kind, stage in event_types),
                       f"{config.method} worker 缺少 {stage_id} 阶段起始事件。")
            _check(any(row.get("type") == "method-terminal" and row.get("status") == "success" for row in events),
                   f"{config.method} worker 没有成功终态。")
            caches.append(cache_dir)
        checks.append("two-method real worker batch and structured stages")

        project = workspace / "project.cst"
        staging = workspace / ".project.cst.pending"
        save_task = workspace / "save-project-worker"
        save_events = _launch_worker(save_task, {
            "run_id": f"{task_id}-save", "operation": "save-project", "method": "",
            "destination": str(project), "staging_directory": str(staging),
            "corpus": corpus_to_dict(corpus), "result_directories": [str(path) for path in caches],
            "settings": {"method": "tfidf-word", "batch_methods": [c.method for c in configs], "split_mode": "每段"},
            "runtime_metadata": {"tfidf-word": {"elapsed_seconds": 1.25}, "tfidf-char": {"elapsed_seconds": 1.5}},
        })
        _check(any(row.get("type") == "stage-start" and row.get("stage_id") == "project-load" for row in save_events),
               "保存项目 worker 缺少项目/缓存读取阶段。")
        _check(any(row.get("type") == "stage-start" and row.get("stage_id") == "project-save" for row in save_events),
               "保存项目 worker 缺少项目保存阶段。")
        loaded_corpus, loaded_results, settings, metadata = load_project_full(project)
        _check(loaded_corpus.texts == corpus.texts and set(loaded_results) == {c.method for c in configs},
               "项目往返没有恢复语料和所有方法结果。")
        _check(settings.get("batch_methods") == [c.method for c in configs] and
               metadata["tfidf-word"]["elapsed_seconds"] == 1.25,
               "项目没有恢复界面配置或运行元数据。")
        checks.append("JSON + NPZ project roundtrip and saved UI settings")

        empty_project = workspace / "corpus-only.cst"
        save_project(empty_project, corpus, {}, settings={"method": "tfidf-char", "batch_methods": []})
        empty_corpus, empty_results, _, _ = load_project_full(empty_project)
        _check(empty_corpus.texts == corpus.texts and not empty_results, "无结果语料项目无法恢复。")
        checks.append("corpus-only project save and reopen")

        profile.save_last_settings({"method": "tfidf-word", "split_mode": "每段", "batch_methods": ["tfidf-word"]})
        profile.save_preset("runtime-self-test", {"method": "tfidf-char", "batch_methods": ["tfidf-char"]})
        _check(profile.load_last_settings().get("split_mode") == "每段" and
               "runtime-self-test" in profile.presets(), "last settings / named preset did not roundtrip.")
        profile.append_log({"run_id": task_id, "type": "warning", "warning": "self-test warning detail"}, force=True)
        profile.append_log({"run_id": task_id, "type": "error", "details": "self-test traceback detail"}, force=True)
        _check(any("warning" in line for line in profile.read_logs("warning")) and
               any("traceback detail" in line for line in profile.read_logs("traceback detail")),
               "可筛选日志没有保留 warning 或错误详情。")
        profile.update_task(task_id, status="success", result_directories={config.method: str(cache)
                                                                          for config, cache in zip(configs, caches)})
        _check(any(row.get("task_id") == task_id and row.get("status") == "success" for row in profile.history()),
               "任务历史没有持久化成功状态。")
        checks.append("logs, history, last settings, and named presets")

        _eta_checks(profile)
        checks.append("ETA cold start, one/two observations, stage switch, history, overrun, stale and terminal")

        from .exports import export_results
        result_corpus, result_map, _, _ = load_project_full(caches[0])
        export_root = workspace / "exports"
        export_staging = export_root / ".cancelled-export-owned"
        cancel = Event()
        cancel.set()
        try:
            export_results(result_map["tfidf-word"], result_corpus, export_root,
                           staging_directory=export_staging, cancel_event=cancel)
        except InterruptedError:
            pass
        else:
            raise AssertionError("已取消的导出没有响应取消信号。")
        _check(not export_staging.exists() and
               not any(path.is_dir() for path in export_root.glob("中文语义分析_*")),
               "取消导出留下了看似成功的临时或最终目录。")
        checks.append("cancelled export leaves no partial published bundle")

        _cancel_probe(workspace)
        checks.append("owned worker cancellation and bounded temp cleanup")

        profile.update_task(task_id, status="success", ended_at=profile.history()[0].get("ended_at"),
                            summary="runtime self-test passed")
        workspace_path = workspace

    report = {
        "status": "success", "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_directory": str(output), "profile_directory": str(output / "test-profile"),
        "checks": checks, "methods": ["tfidf-word", "tfidf-char"],
        "worker_stages": {method: sorted({str(row.get("stage_id")) for row in rows
                                          if row.get("type") in {"stage-start", "stage-complete", "stage-progress"}})
                          for method, rows in worker_events.items()},
        "api": {
            "eta": "semantic_toolkit.eta.EtaTracker(plan, clock=callable, history=rows)",
            "runtime": "semantic_toolkit.runtime.RuntimeStore(root), save_project/load_project_full, result_cache_load",
            "worker": "main.py --runtime-worker TASK_DIR with request.json and events.jsonl",
            "self_test": "main.py --runtime-self-test OUTPUT_DIR",
        },
    }
    report_path = output / "runtime-self-test.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
