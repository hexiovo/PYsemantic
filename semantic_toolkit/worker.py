"""Windowless one-shot worker entry; imports scientific packages after routing."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time
import traceback


class EventWriter:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.sequence = 0

    def emit(self, event: dict[str, Any]) -> None:
        self.sequence += 1
        value = dict(event)
        value.update({"run_id": self.run_id, "seq": self.sequence})
        value.setdefault("timestamp", __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"))
        value.setdefault("monotonic", time.monotonic())
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()


def _write_error(task_dir: Path, exc: BaseException) -> None:
    payload = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    (task_dir / "error.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_worker(task_directory: str | Path) -> int:
    task_dir = Path(task_directory).resolve()
    request_path = task_dir / "request.json"
    events_path = task_dir / "events.jsonl"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    writer = EventWriter(events_path, str(request["run_id"]))
    operation = request.get("operation")
    writer.emit({"type": "stage-start", "stage_id": "worker-startup", "label": "启动后台工作进程",
                 "method": str(request.get("method", "")), "initial_seconds": 1.2})
    try:
        # Deliberately delayed imports: frozen child routing is complete before
        # CustomTkinter or scientific libraries enter the import graph.
        if operation == "analyze":
            from .analysis import AnalysisConfig, run_analysis
            from .runtime import corpus_from_dict, save_result_cache
            corpus = corpus_from_dict(request["corpus"])
            config = AnalysisConfig(**request["config"])
            writer.emit({"type": "stage-complete", "stage_id": "worker-startup", "method": config.method})
            analysis_started = time.monotonic()
            result = run_analysis(corpus, config, event_callback=writer.emit)
            analysis_elapsed = max(0.0, time.monotonic() - analysis_started)
            for warning in result.warnings:
                writer.emit({"type": "warning", "method": config.method, "warning": warning})
            writer.emit({"type": "stage-start", "stage_id": "result-save", "label": "安全保存分析结果",
                         "method": config.method, "initial_seconds": 1.0})
            save_result_cache(request["result_directory"], corpus, result,
                              run_info={"elapsed_seconds": analysis_elapsed,
                                        "method_index": request.get("method_index"),
                                        "method_total": request.get("method_total")})
            writer.emit({"type": "stage-complete", "stage_id": "result-save", "method": config.method})
            writer.emit({"type": "method-terminal", "status": "success", "method": config.method,
                         "elapsed_seconds": analysis_elapsed})
            return 0
        if operation == "export":
            from .runtime import load_project_full
            from .exports import export_results
            writer.emit({"type": "stage-complete", "stage_id": "worker-startup", "method": request["method"]})
            writer.emit({"type": "stage-start", "stage_id": "project-load", "label": "载入已完成结果",
                         "method": request["method"], "initial_seconds": 1.0})
            corpus, results, _settings, metadata = load_project_full(request["result_directory"])
            method = request["method"]
            writer.emit({"type": "stage-complete", "stage_id": "project-load", "method": method})
            result = results[method]
            path = export_results(result, corpus, request["parent_directory"], include_xlsx=bool(request.get("include_xlsx")),
                                  staging_directory=request["staging_directory"], progress=writer.emit,
                                  method_index=request.get("method_index"), method_total=request.get("method_total"),
                                  cluster_names=request.get("cluster_names", metadata.get(method, {}).get("cluster_names", {})))
            (task_dir / "output.json").write_text(json.dumps({"path": str(path)}, ensure_ascii=False), encoding="utf-8")
            writer.emit({"type": "method-terminal", "status": "success", "method": method})
            return 0
        if operation == "save-project":
            from .runtime import create_project_from_caches
            writer.emit({"type": "stage-complete", "stage_id": "worker-startup", "method": ""})
            path = create_project_from_caches(request["destination"], request["corpus"], request["result_directories"],
                                              settings=request.get("settings", {}),
                                              runtime_metadata=request.get("runtime_metadata", {}),
                                              staging_directory=request["staging_directory"],
                                              event_callback=writer.emit)
            (task_dir / "output.json").write_text(json.dumps({"path": str(path)}, ensure_ascii=False), encoding="utf-8")
            writer.emit({"type": "operation-complete", "status": "success"})
            return 0
        if operation == "runtime-selftest-cancel-probe" and request.get("self_test_only") is True:
            owned_directory = Path(request["owned_directory"])
            owned_directory.mkdir(parents=True, exist_ok=True)
            (owned_directory / "worker-owned.txt").write_text("owned by runtime self-test", encoding="utf-8")
            writer.emit({"type": "stage-complete", "stage_id": "worker-startup", "method": "self-test"})
            writer.emit({"type": "stage-start", "stage_id": "cancel-probe", "label": "可中断子进程检查",
                         "method": "self-test", "initial_seconds": 60.0})
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline and not (task_dir / "cancel.flag").exists():
                time.sleep(0.02)
            return 0
        raise ValueError(f"Unknown worker operation: {operation}")
    except InterruptedError as exc:
        _write_error(task_dir, exc)
        kind = "method-terminal" if operation == "analyze" else "operation-terminal"
        writer.emit({"type": kind, "status": "cancelled", "error_type": type(exc).__name__,
                     "method": request.get("method")})
        return 2
    except BaseException as exc:
        _write_error(task_dir, exc)
        kind = "method-terminal" if operation == "analyze" else "operation-terminal"
        writer.emit({"type": kind, "status": "failed", "error_type": type(exc).__name__,
                     "method": request.get("method")})
        return 1

