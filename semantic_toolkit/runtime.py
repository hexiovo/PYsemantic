"""Durable runtime state and safe JSON + NPZ project packages.

Runtime logs and settings intentionally omit text content. A user-created
project includes the corpus text and is written only after an explicit save.
Numerical matrices are stored as non-object NPZ arrays; loading always uses
``allow_pickle=False`` and checks the declared shapes before constructing a
result.
"""
from __future__ import annotations

from datetime import datetime
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Iterable
import json
import os
import subprocess
import shutil
import tempfile
from time import monotonic
import uuid

import numpy as np
import pandas as pd
from scipy import sparse

from .analysis import AnalysisResult, METHODS
from .data_io import Corpus, Document
from .resources import user_data_path


PROJECT_SCHEMA = 1
MAX_HISTORY_ROWS = 300
MAX_HISTORY_SAMPLES = 800
LOG_MAX_BYTES = 3 * 1024 * 1024
LOG_BACKUPS = 4
WORKER_ERROR_MAX_ARCHIVES = 10
WORKER_ERROR_MAX_BYTES = 24 * 1024 * 1024


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _win32_kernel32() -> Any:
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _resume_suspended_process(pid: int) -> None:
    """Resume the sole primary thread of a process created with CREATE_SUSPENDED."""
    import ctypes
    from ctypes import wintypes

    class ThreadEntry32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                    ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

    kernel32 = _win32_kernel32()
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry32)]
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry32)]
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or snapshot == invalid_handle:
        raise OSError(ctypes.get_last_error(), "Could not enumerate the suspended worker thread")
    thread_handle = None
    try:
        entry = ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        found = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        thread_id = 0
        while found:
            if int(entry.th32OwnerProcessID) == int(pid):
                thread_id = int(entry.th32ThreadID)
                break
            found = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        if not thread_id:
            raise OSError("Could not find the suspended worker's primary thread")
        thread_handle = kernel32.OpenThread(0x0002, False, thread_id)  # THREAD_SUSPEND_RESUME
        if not thread_handle:
            raise OSError(ctypes.get_last_error(), "Could not open the suspended worker's primary thread")
        if kernel32.ResumeThread(thread_handle) == 0xFFFFFFFF:
            raise OSError(ctypes.get_last_error(), "Could not resume the suspended worker")
    finally:
        if thread_handle:
            kernel32.CloseHandle(thread_handle)
        kernel32.CloseHandle(snapshot)


def _new_kill_on_close_job(process_handle: int) -> int | None:
    """Assign a process to a job before its first instruction can spawn children."""
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimitInformation), ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = _win32_kernel32()
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    information = ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(information), ctypes.sizeof(information)):
        kernel32.CloseHandle(job)
        return None
    if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process_handle)):
        kernel32.CloseHandle(job)
        return None
    return int(job)


def popen_owned_process(*args: Any, **kwargs: Any) -> subprocess.Popen:
    """Start a worker with an inherited Windows job, or fall back to tree-kill."""
    inherited_environment = kwargs.get("env")
    worker_environment = dict(os.environ if inherited_environment is None else inherited_environment)
    worker_environment["PYINSTALLER_SUPPRESS_SPLASH_SCREEN"] = "1"
    kwargs["env"] = worker_environment
    if os.name != "nt":
        return subprocess.Popen(*args, **kwargs)
    flags = int(kwargs.get("creationflags", 0))
    kwargs["creationflags"] = flags | 0x00000004  # CREATE_SUSPENDED; assign job before worker code runs
    process = subprocess.Popen(*args, **kwargs)
    job_handle = None
    try:
        job_handle = _new_kill_on_close_job(int(process._handle))
        if job_handle is not None:
            process._semantic_job_handle = job_handle
        _resume_suspended_process(int(process.pid))
        return process
    except BaseException:
        if job_handle is not None:
            try:
                terminate_owned_process(process, 5)
            finally:
                close_owned_process_tree(process)
        else:
            if process.poll() is None:
                process.kill()
            process.wait()
        raise


def owned_process_tree_finished(process: Any, timeout_seconds: float = 0.0) -> bool:
    """Wait until all descendants in this task's Windows job have exited."""
    job_handle = getattr(process, "_semantic_job_handle", None)
    if job_handle is None:
        return process.poll() is not None
    import ctypes
    from ctypes import wintypes

    class BasicAccountingInformation(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                    ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                    ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    kernel32 = _win32_kernel32()
    kernel32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                   wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    deadline = monotonic() + max(0.0, float(timeout_seconds))
    while True:
        information = BasicAccountingInformation()
        returned = wintypes.DWORD()
        ok = kernel32.QueryInformationJobObject(job_handle, 1, ctypes.byref(information),
                                                ctypes.sizeof(information), ctypes.byref(returned))
        if not ok:
            return False
        if information.ActiveProcesses == 0:
            return True
        if monotonic() >= deadline:
            return False
        __import__("time").sleep(min(0.02, max(0.001, deadline - monotonic())))


def _taskkill_tree_by_owned_pid(process: Any, timeout_seconds: float) -> None:
    """Fallback for hosts that prohibit job assignment; targets only our Popen PID."""
    startupinfo = None
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
    try:
        subprocess.run(["taskkill.exe", "/PID", str(int(process.pid)), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False, timeout=max(1.0, float(timeout_seconds)),
                       startupinfo=startupinfo, creationflags=creationflags)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()


def close_owned_process_tree(process: Any) -> None:
    """Release the task's Windows job after its worker tree is confirmed idle."""
    job_handle = getattr(process, "_semantic_job_handle", None)
    if job_handle is None:
        return
    kernel32 = _win32_kernel32()
    import ctypes
    from ctypes import wintypes
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not owned_process_tree_finished(process, 0.0):
        kernel32.TerminateJobObject(job_handle, 1)
        if not owned_process_tree_finished(process, 5.0):
            raise TimeoutError("本任务后台进程树未能确认结束，保留任务暂存文件以避免子进程继续写入。")
    kernel32.CloseHandle(job_handle)
    process._semantic_job_handle = None


def terminate_owned_process(process: Any, timeout_seconds: float = 5.0) -> int | None:
    """Stop and reap only the subprocess tree owned by the current task."""
    timeout = max(0.1, float(timeout_seconds))
    job_handle = getattr(process, "_semantic_job_handle", None)
    if job_handle is not None:
        kernel32 = _win32_kernel32()
        from ctypes import wintypes
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject(job_handle, 1)
        if not owned_process_tree_finished(process, timeout):
            _taskkill_tree_by_owned_pid(process, timeout)
            if not owned_process_tree_finished(process, timeout):
                raise TimeoutError("本任务后台进程树未能在限定时间内结束。")
        return process.wait(timeout=timeout)
    if os.name == "nt":
        # Popen keeps its process handle open, preventing PID reuse while this
        # exact PID is passed to taskkill. /T is scoped to this worker's tree.
        _taskkill_tree_by_owned_pid(process, timeout)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=timeout)
    if process.poll() is not None:
        return process.returncode
    process.terminate()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=timeout)


@contextmanager
def _runtime_stage(event_callback: Any, stage_id: str, label: str, *, method: str = "",
                   work_total: float | None = None, unit: str = "工作量"):
    started = monotonic()
    if event_callback:
        event_callback({"type": "stage-start", "stage_id": stage_id, "label": label, "method": method,
                        "work_total": work_total, "unit": unit, "initial_seconds": 1.0})
    try:
        yield
    except BaseException:
        if event_callback:
            event_callback({"type": "stage-failed", "stage_id": stage_id, "method": method,
                            "elapsed_seconds": max(0.0, monotonic() - started)})
        raise
    else:
        if event_callback:
            event_callback({"type": "stage-complete", "stage_id": stage_id, "method": method,
                            "work_done": work_total, "work_total": work_total, "unit": unit,
                            "elapsed_seconds": max(0.0, monotonic() - started)})


class RuntimeStore:
    """Bounded, append-only task logs plus atomic history/settings records."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else user_data_path()
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.logs_dir / "runtime.jsonl"
        self.history_path = self.root / "task-history.json"
        self.presets_path = self.root / "presets.json"
        self.last_settings_path = self.root / "last-settings.json"
        self.projects_dir = self.root / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._last_log_tick: dict[str, float] = {}
        self._mark_interrupted()

    def _atomic_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)

    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def _mark_interrupted(self) -> None:
        rows = self._read_json(self.history_path, [])
        changed = False
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            if isinstance(row, dict) and row.get("status") == "running":
                row["status"] = "interrupted"
                row["ended_at"] = now_iso()
                row["summary"] = "应用上次未正常结束；已保存的阶段结果仍可单独打开。"
                changed = True
        if changed:
            self._atomic_json(self.history_path, rows[-MAX_HISTORY_ROWS:])

    def _rotate_logs_if_needed(self) -> None:
        try:
            if self.events_path.stat().st_size < LOG_MAX_BYTES:
                return
        except OSError:
            return
        oldest = self.logs_dir / f"runtime.jsonl.{LOG_BACKUPS}"
        oldest.unlink(missing_ok=True)
        for index in range(LOG_BACKUPS - 1, 0, -1):
            source = self.logs_dir / f"runtime.jsonl.{index}"
            target = self.logs_dir / f"runtime.jsonl.{index + 1}"
            if source.exists():
                os.replace(source, target)
        if self.events_path.exists():
            os.replace(self.events_path, self.logs_dir / "runtime.jsonl.1")

    def prune_worker_error_archives(self) -> None:
        """Keep recent startup failure evidence inside a bounded app-owned area."""
        root = self.logs_dir / "worker-errors"
        try:
            archives = [path for path in root.iterdir() if path.is_dir()]
        except OSError:
            return
        archives.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)

        def archive_size(path: Path) -> int:
            total = 0
            try:
                for child in path.rglob("*"):
                    if child.is_file():
                        total += child.stat().st_size
            except OSError:
                pass
            return total

        retained = archives[:WORKER_ERROR_MAX_ARCHIVES]
        sizes = {path: archive_size(path) for path in retained}
        total = sum(sizes.values())
        # Always retain the newest archive, even if one unusual process emitted
        # a very large log; all older evidence is removed before exceeding the
        # bounded archive count/total in normal operation.
        for path in reversed(retained[1:]):
            if len(retained) <= 1 or (len(retained) <= WORKER_ERROR_MAX_ARCHIVES
                                      and total <= WORKER_ERROR_MAX_BYTES):
                break
            try:
                shutil.rmtree(path)
                total -= sizes.get(path, 0)
                retained.remove(path)
            except OSError:
                continue
        for path in archives[WORKER_ERROR_MAX_ARCHIVES:]:
            try:
                shutil.rmtree(path)
            except OSError:
                pass

    def append_log(self, event: dict[str, Any], *, force: bool = False) -> None:
        """Append a structured event; frequent progress records are throttled."""
        run_id = str(event.get("run_id", ""))
        kind = str(event.get("type", "event"))
        if kind == "stage-progress" and not force:
            import time
            tick = time.monotonic()
            if tick - self._last_log_tick.get(run_id, 0.0) < 2.0:
                return
            self._last_log_tick[run_id] = tick
        allowed = {
            key: event[key] for key in (
                "run_id", "task_id", "seq", "timestamp", "type", "stage_id", "label",
                "method", "status", "work_done", "work_total", "unit", "elapsed_seconds",
                "documents_done", "documents_total", "token_count_done", "tokens_encoded",
                "document_index", "current_document_complete", "method_index", "method_total",
                "monotonic", "message_code", "progress_message", "error_type", "warning", "details",
            ) if key in event
        }
        if isinstance(allowed.get("details"), str):
            allowed["details"] = allowed["details"][:100_000]
        allowed.setdefault("timestamp", now_iso())
        allowed["type"] = kind
        payload = json.dumps(allowed, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._rotate_logs_if_needed()
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)

    def _history(self) -> list[dict[str, Any]]:
        rows = self._read_json(self.history_path, [])
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def history(self) -> list[dict[str, Any]]:
        return list(reversed(self._history()))

    def begin_task(self, task: dict[str, Any]) -> None:
        rows = self._history()
        row = dict(task)
        row.setdefault("started_at", now_iso())
        row.setdefault("status", "running")
        rows.append(row)
        self._atomic_json(self.history_path, rows[-MAX_HISTORY_ROWS:])

    def update_task(self, task_id: str, **values: Any) -> None:
        rows = self._history()
        for row in rows:
            if row.get("task_id") == task_id:
                row.update(values)
                break
        self._atomic_json(self.history_path, rows[-MAX_HISTORY_ROWS:])

    def successful_stage(self, sample: dict[str, Any]) -> None:
        path = self.root / "eta-history.json"
        rows = self._read_json(path, [])
        if not isinstance(rows, list):
            rows = []
        row = dict(sample)
        row["recorded_at"] = now_iso()
        if float(row.get("elapsed_seconds", 0) or 0) <= 0 or row.get("successful") is False:
            return
        rows.append(row)
        self._atomic_json(path, rows[-MAX_HISTORY_SAMPLES:])

    def eta_history(self) -> list[dict[str, Any]]:
        value = self._read_json(self.root / "eta-history.json", [])
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []

    def save_last_settings(self, settings: dict[str, Any]) -> None:
        self._atomic_json(self.last_settings_path, settings)

    def load_last_settings(self) -> dict[str, Any]:
        value = self._read_json(self.last_settings_path, {})
        return value if isinstance(value, dict) else {}

    def presets(self) -> dict[str, dict[str, Any]]:
        value = self._read_json(self.presets_path, {})
        return {str(k): v for k, v in value.items() if isinstance(v, dict)} if isinstance(value, dict) else {}

    def save_preset(self, name: str, settings: dict[str, Any]) -> None:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("预设名称不能为空。")
        values = self.presets()
        values[clean_name] = settings
        self.save_presets(values)

    def save_presets(self, presets: dict[str, dict[str, Any]]) -> None:
        """Atomically replace the named preset collection after an import."""
        if not isinstance(presets, dict) or any(not isinstance(name, str) or not isinstance(settings, dict)
                                                for name, settings in presets.items()):
            raise ValueError("预设必须是名称到设置对象的映射。")
        self._atomic_json(self.presets_path, presets)

    def delete_preset(self, name: str) -> None:
        values = self.presets()
        values.pop(name, None)
        self._atomic_json(self.presets_path, values)

    def read_logs(self, filter_text: str = "") -> list[str]:
        paths = [self.logs_dir / f"runtime.jsonl.{i}" for i in range(LOG_BACKUPS, 0, -1)] + [self.events_path]
        lines: list[str] = []
        needle = filter_text.casefold().strip()
        for path in paths:
            try:
                values = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in values:
                if not needle or needle in line.casefold():
                    lines.append(line)
        return lines

    def export_logs(self, path: str | Path, filter_text: str = "") -> Path:
        out = Path(path)
        out.write_text("\n".join(self.read_logs(filter_text)) + "\n", encoding="utf-8")
        return out


def corpus_to_dict(corpus: Corpus, *, include_text: bool = True) -> dict[str, Any]:
    return {
        "documents": [{"doc_id": d.doc_id, "text": d.text if include_text else "", "source": d.source,
                       "source_row": d.source_row} for d in corpus.documents],
        "cleaning_log": corpus.cleaning_log,
        "stats": corpus.stats,
    }


def corpus_from_dict(value: dict[str, Any]) -> Corpus:
    docs = value.get("documents")
    if not isinstance(docs, list):
        raise ValueError("项目缺少 documents 文档列表。")
    documents: list[Document] = []
    for index, item in enumerate(docs):
        if not isinstance(item, dict) or not isinstance(item.get("doc_id"), str):
            raise ValueError(f"项目文档 {index + 1} 格式错误。")
        text = item.get("text", "")
        if not isinstance(text, str):
            raise ValueError(f"项目文档 {index + 1} 的 text 必须是字符串。")
        documents.append(Document(item["doc_id"], text, str(item.get("source", "")), item.get("source_row")))
    if not documents:
        raise ValueError("项目中没有文档。")
    return Corpus(documents, list(value.get("cleaning_log") or []), dict(value.get("stats") or {}))


def _encode_matrix(key: str, matrix: Any, arrays: dict[str, np.ndarray], specs: dict[str, Any]) -> None:
    if matrix is None:
        specs[key] = None
        return
    if sparse.issparse(matrix):
        value = matrix.tocsr()
        arrays[f"{key}__data"] = np.asarray(value.data)
        arrays[f"{key}__indices"] = np.asarray(value.indices, dtype=np.int64)
        arrays[f"{key}__indptr"] = np.asarray(value.indptr, dtype=np.int64)
        shape = [int(value.shape[0]), int(value.shape[1])]
        arrays[f"{key}__shape"] = np.asarray(shape, dtype=np.int64)
        specs[key] = {"kind": "csr", "shape": shape, "dtype": str(value.dtype)}
    else:
        value = np.asarray(matrix)
        if value.dtype.kind not in "biufc":
            raise ValueError(f"矩阵 {key} 含不支持的数据类型 {value.dtype}。")
        arrays[key] = value
        specs[key] = {"kind": "dense", "shape": [int(x) for x in value.shape], "dtype": str(value.dtype)}


def _result_to_package(result: AnalysisResult, directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    specs: dict[str, Any] = {}
    for key in ("word_count_matrix", "word_tfidf_matrix", "doc_matrix", "similarity_matrix",
                "topic_document_matrix", "topic_weight_matrix", "cooccurrence_counts", "ppmi_matrix"):
        _encode_matrix(key, getattr(result, key), arrays, specs)
    with (directory / "matrices.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    table_files: dict[str, str] = {}
    for key in ("document_metrics", "word_frequencies", "nearest_neighbors", "topic_terms", "cluster_labels"):
        filename = f"{key}.json"
        (directory / filename).write_text(getattr(result, key).to_json(orient="table", force_ascii=False), encoding="utf-8")
        table_files[key] = filename
    metadata = {
        "doc_ids": result.doc_ids,
        "method": result.method,
        "method_label": result.method_label,
        "config": result.config,
        "word_vocabulary": result.word_vocabulary,
        "doc_feature_labels": result.doc_feature_labels,
        "doc_matrix_name": result.doc_matrix_name,
        "topic_weight_kind": result.topic_weight_kind,
        "cooccurrence_vocabulary": result.cooccurrence_vocabulary,
        "clustering_summary": result.clustering_summary,
        "warnings": result.warnings,
        "zero_document_ids": result.zero_document_ids,
        "matrix_specs": specs,
        "table_files": table_files,
    }
    (directory / "result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _decode_matrix(key: str, spec: dict[str, Any] | None, archive: Any) -> Any:
    if spec is None:
        return None
    shape = tuple(int(x) for x in spec.get("shape", []))
    if len(shape) != 2 or any(x < 0 for x in shape):
        raise ValueError(f"矩阵 {key} 声明的形状无效。")
    if spec.get("kind") == "dense":
        value = np.asarray(archive[key])
        if value.shape != shape or value.dtype.kind not in "biufc":
            raise ValueError(f"矩阵 {key} 的数组形状或类型不匹配。")
        return value
    if spec.get("kind") != "csr":
        raise ValueError(f"矩阵 {key} 的存储类型无效。")
    data = np.asarray(archive[f"{key}__data"])
    indices = np.asarray(archive[f"{key}__indices"], dtype=np.int64)
    indptr = np.asarray(archive[f"{key}__indptr"], dtype=np.int64)
    if data.ndim != 1 or indices.ndim != 1 or indptr.shape != (shape[0] + 1,) or len(data) != len(indices):
        raise ValueError(f"矩阵 {key} 的 CSR 形状或索引不匹配。")
    if len(indptr) and (indptr[0] != 0 or indptr[-1] != len(data) or np.any(np.diff(indptr) < 0)):
        raise ValueError(f"矩阵 {key} 的 CSR 行指针不合法。")
    if len(indices) and (indices.min() < 0 or indices.max() >= shape[1]):
        raise ValueError(f"矩阵 {key} 含越界列索引。")
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def _result_from_package(directory: Path) -> AnalysisResult:
    metadata = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    method = metadata.get("method")
    if method not in METHODS or metadata.get("method_label") != METHODS[method]:
        raise ValueError("项目中的分析方法或显示名称不受支持。")
    with np.load(directory / "matrices.npz", allow_pickle=False) as archive:
        specs = metadata.get("matrix_specs", {})
        matrices = {key: _decode_matrix(key, specs.get(key), archive) for key in (
            "word_count_matrix", "word_tfidf_matrix", "doc_matrix", "similarity_matrix",
            "topic_document_matrix", "topic_weight_matrix", "cooccurrence_counts", "ppmi_matrix")}
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in metadata["table_files"].items():
        if Path(filename).name != filename or key not in {"document_metrics", "word_frequencies", "nearest_neighbors", "topic_terms", "cluster_labels"}:
            raise ValueError("项目表格文件名无效。")
        tables[key] = pd.read_json(StringIO((directory / filename).read_text(encoding="utf-8")), orient="table")
    doc_ids = list(map(str, metadata.get("doc_ids", [])))
    word_vocabulary = list(map(str, metadata.get("word_vocabulary", [])))
    cooccurrence_vocabulary = list(map(str, metadata.get("cooccurrence_vocabulary", [])))
    doc_matrix = matrices["doc_matrix"]
    if doc_matrix is None or len(doc_matrix.shape) != 2 or doc_matrix.shape[0] != len(doc_ids):
        raise ValueError("项目文档表示与文档 ID 数量不一致。")
    if matrices["word_count_matrix"].shape[0] != len(doc_ids) or matrices["word_tfidf_matrix"].shape[0] != len(doc_ids):
        raise ValueError("项目词项矩阵与文档数量不一致。")
    if matrices["word_count_matrix"].shape[1] != len(word_vocabulary) or matrices["word_tfidf_matrix"].shape[1] != len(word_vocabulary):
        raise ValueError("项目词项矩阵与词表长度不一致。")
    if len(metadata.get("doc_feature_labels", [])) != doc_matrix.shape[1]:
        raise ValueError("项目文档表示与特征标签数量不一致。")
    if matrices["similarity_matrix"] is not None and matrices["similarity_matrix"].shape != (len(doc_ids), len(doc_ids)):
        raise ValueError("项目相似度矩阵与文档数量不一致。")
    topic_document = matrices["topic_document_matrix"]
    topic_weight = matrices["topic_weight_matrix"]
    if (topic_document is None) != (topic_weight is None):
        raise ValueError("项目主题文档矩阵与主题词权重矩阵必须同时存在或同时为空。")
    if topic_document is not None:
        if getattr(topic_document, "ndim", 0) != 2 or topic_document.shape[0] != len(doc_ids):
            raise ValueError("项目主题文档矩阵必须按文档 ID 对齐。")
        if getattr(topic_weight, "ndim", 0) != 2 or topic_weight.shape[1] != len(word_vocabulary):
            raise ValueError("项目主题权重矩阵列必须与词表对齐。")
        if topic_document.shape[1] != topic_weight.shape[0]:
            raise ValueError("项目文档主题轴与主题权重矩阵主题轴不一致。")
    for key in ("cooccurrence_counts", "ppmi_matrix"):
        matrix = matrices[key]
        if matrix is not None and matrix.shape != (len(cooccurrence_vocabulary), len(cooccurrence_vocabulary)):
            raise ValueError(f"项目 {key} 必须按共现词表构成方阵。")
    if len(tables["document_metrics"]) != len(doc_ids):
        raise ValueError("项目文档指标行数与文档数量不一致。")
    return AnalysisResult(
        doc_ids=doc_ids, method=method, method_label=metadata["method_label"], config=dict(metadata["config"]),
        word_count_matrix=matrices["word_count_matrix"], word_tfidf_matrix=matrices["word_tfidf_matrix"],
        word_vocabulary=word_vocabulary, doc_matrix=doc_matrix,
        doc_feature_labels=list(metadata.get("doc_feature_labels", [])), doc_matrix_name=str(metadata.get("doc_matrix_name", "")),
        document_metrics=tables["document_metrics"], word_frequencies=tables["word_frequencies"],
        nearest_neighbors=tables["nearest_neighbors"], similarity_matrix=matrices["similarity_matrix"],
        topic_document_matrix=matrices["topic_document_matrix"], topic_weight_matrix=matrices["topic_weight_matrix"],
        topic_weight_kind=str(metadata.get("topic_weight_kind", "")), topic_terms=tables["topic_terms"],
        cooccurrence_counts=matrices["cooccurrence_counts"], ppmi_matrix=matrices["ppmi_matrix"],
        cooccurrence_vocabulary=cooccurrence_vocabulary,
        cluster_labels=tables["cluster_labels"], clustering_summary=dict(metadata.get("clustering_summary", {})),
        warnings=list(metadata.get("warnings", [])), zero_document_ids=list(metadata.get("zero_document_ids", [])),
    )


def save_project(directory: str | Path, corpus: Corpus, results: dict[str, AnalysisResult], *,
                 settings: dict[str, Any] | None = None, runtime_metadata: dict[str, Any] | None = None,
                 staging_directory: str | Path | None = None, cancel_event: Any = None,
                 event_callback: Any = None) -> Path:
    """Atomically create a new JSON + NPZ project; existing projects are preserved."""
    destination = Path(directory).resolve()
    if destination.exists():
        raise FileExistsError(f"项目目录已存在：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(staging_directory).resolve() if staging_directory else destination.parent / f".{destination.name}.pending-{uuid.uuid4().hex}"
    if temp.parent != destination.parent:
        raise ValueError("項目暫存目錄必須位於項目目標的父目錄中，才能原子提交。")
    if temp.exists():
        raise FileExistsError(f"項目暫存目錄已存在：{temp}")
    temp.mkdir()
    try:
        with _runtime_stage(event_callback, "project-save", "保存项目",
                            work_total=max(1, len(results)), unit="分析结果"):
            payload = {"schema": PROJECT_SCHEMA, "saved_at": now_iso(), "corpus": corpus_to_dict(corpus),
                       "results": {}, "settings": settings or {}, "runtime_metadata": runtime_metadata or {}}
            for index, (method, result) in enumerate(results.items(), 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("项目保存已取消")
                if method != result.method:
                    raise ValueError("结果集合键和实际 method 不一致。")
                payload["results"][method] = _result_to_package(result, temp / "results" / method)
                if event_callback:
                    event_callback({"type": "stage-progress", "stage_id": "project-save", "work_done": index,
                                    "work_total": max(1, len(results)), "unit": "分析结果"})
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("项目保存已取消")
            (temp / "project.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            # Re-read only the boundary manifest; matrix shapes are checked once
            # when a saved project is reopened.
            _read_project_manifest(temp)
            os.replace(temp, destination)
        return destination
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _read_project_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "project.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PROJECT_SCHEMA:
        raise ValueError(f"不支持的项目格式版本：{manifest.get('schema')}")
    corpus_payload = manifest.get("corpus")
    if not isinstance(corpus_payload, dict) or not isinstance(corpus_payload.get("documents"), list):
        raise ValueError("项目缺少合法的 corpus 文档列表。")
    result_meta = manifest.get("results")
    if not isinstance(result_meta, dict):
        raise ValueError("项目 results 必须是方法字典。")
    for method, metadata in result_meta.items():
        if method not in METHODS:
            raise ValueError(f"项目包含未知方法：{method}")
        if metadata.get("method") != method:
            raise ValueError(f"项目结果索引 {method} 不一致。")
    return manifest


def load_project(directory: str | Path) -> tuple[Corpus, dict[str, AnalysisResult]]:
    corpus, results, _, _ = load_project_full(directory)
    return corpus, results


def load_project_full(directory: str | Path) -> tuple[Corpus, dict[str, AnalysisResult], dict[str, Any], dict[str, Any]]:
    root = Path(directory)
    manifest = _read_project_manifest(root)
    corpus = corpus_from_dict(manifest["corpus"])
    results: dict[str, AnalysisResult] = {}
    for method in manifest["results"]:
        result = _result_from_package(root / "results" / method)
        if result.method != method or result.doc_ids != corpus.ids:
            raise ValueError(f"项目结果 {method} 与语料文档顺序不一致。")
        results[method] = result
    settings = manifest.get("settings", {})
    runtime_metadata = manifest.get("runtime_metadata", {})
    return corpus, results, settings if isinstance(settings, dict) else {}, runtime_metadata if isinstance(runtime_metadata, dict) else {}


def save_result_cache(directory: str | Path, corpus: Corpus, result: AnalysisResult, *,
                      run_info: dict[str, Any] | None = None) -> Path:
    """Persist a reopenable result cache without retaining raw document text."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"结果缓存目录已存在：{destination}")
    temp = destination.parent / f".{destination.name}.pending-{uuid.uuid4().hex}"
    temp.mkdir()
    try:
        manifest = {"schema": PROJECT_SCHEMA, "saved_at": now_iso(), "runtime_metadata": {result.method: run_info or {}},
                    "corpus": corpus_to_dict(corpus, include_text=False), "results": {result.method: None}}
        manifest["results"][result.method] = _result_to_package(result, temp / "results" / result.method)
        (temp / "project.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _read_project_manifest(temp)
        os.replace(temp, destination)
        return destination
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def result_cache_load(directory: str | Path) -> tuple[Corpus, dict[str, AnalysisResult]]:
    return load_project(directory)


def create_project_from_caches(directory: str | Path, corpus_dict: dict[str, Any], caches: Iterable[str | Path],
                               *, settings: dict[str, Any] | None = None, runtime_metadata: dict[str, Any] | None = None,
                               staging_directory: str | Path | None = None, cancel_event: Any = None,
                               event_callback: Any = None) -> Path:
    cache_paths = list(caches)
    results: dict[str, AnalysisResult] = {}
    with _runtime_stage(event_callback, "project-load", "载入语料与已完成结果",
                        work_total=1 + len(cache_paths), unit="项目/结果包"):
        corpus = corpus_from_dict(corpus_dict)
        if event_callback:
            event_callback({"type": "stage-progress", "stage_id": "project-load", "work_done": 1,
                            "work_total": 1 + len(cache_paths), "unit": "项目/结果包"})
        for index, cache_path in enumerate(cache_paths, 1):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("項目保存已取消")
            _, found = result_cache_load(cache_path)
            for method, result in found.items():
                if result.doc_ids != corpus.ids:
                    raise ValueError(f"結果 {method} 的文檔順序與目前語料不同，無法併入項目。")
                results[method] = result
            if event_callback:
                event_callback({"type": "stage-progress", "stage_id": "project-load", "work_done": index + 1,
                                "work_total": 1 + len(cache_paths), "unit": "项目/结果包"})
    return save_project(directory, corpus, results, settings=settings, runtime_metadata=runtime_metadata,
                        staging_directory=staging_directory,
                        cancel_event=cancel_event, event_callback=event_callback)

