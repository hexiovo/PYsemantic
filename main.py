"""Chinese Semantic Toolkit desktop entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线中文语义分析工具包")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", metavar="OUTPUT_DIR", help="在输出目录中运行集成烟测并导出全部方法结果")
    group.add_argument("--runtime-self-test", metavar="OUTPUT_DIR", help="验证本轮任务管理、ETA、项目、日志与取消集成")
    group.add_argument("--ui-smoke", metavar="EVIDENCE_DIR", help="打开五个界面页并用程序窗口截图后退出")
    group.add_argument("--ui-log-smoke", metavar="EVIDENCE_DIR", help="单独截取运行日志弹窗并退出")
    group.add_argument("--runtime-worker", metavar="TASK_DIR", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runtime_worker:
        from semantic_toolkit.worker import run_worker
        return run_worker(args.runtime_worker)
    if args.runtime_self_test:
        from semantic_toolkit.runtime_selftest import run_runtime_self_test
        try:
            run_runtime_self_test(args.runtime_self_test)
            return 0
        except Exception as exc:
            import traceback
            details = f"RUNTIME SELF-TEST FAILED: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            error_path = Path(args.runtime_self_test) / "runtime-self-test-error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(details, encoding="utf-8")
            if sys.stderr is not None:
                print(details, file=sys.stderr, flush=True)
            return 1
    if args.self_test:
        try:
            from semantic_toolkit.selftest import run_self_test
            run_self_test(args.self_test)
            return 0
        except Exception as exc:
            import traceback
            details = f"SELF-TEST FAILED: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            error_path = Path(args.self_test) / "self-test-error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(details, encoding="utf-8")
            if sys.stderr is not None:
                print(details, file=sys.stderr, flush=True)
            return 1
    from semantic_toolkit.ui import SemanticApp
    smoke_output = args.ui_smoke or args.ui_log_smoke
    smoke_root = Path(smoke_output) / "test-profile" if smoke_output else None
    app = SemanticApp(ui_smoke_dir=args.ui_smoke, runtime_root=smoke_root)
    if args.ui_log_smoke:
        app.start_log_viewer_smoke(Path(args.ui_log_smoke))
    app.mainloop()
    if args.ui_smoke:
        if app.smoke_errors:
            Path(args.ui_smoke).mkdir(parents=True, exist_ok=True)
            (Path(args.ui_smoke) / "ui-smoke-errors.txt").write_text("\n".join(app.smoke_errors), encoding="utf-8")
        return 0 if app.smoke_capture_count == app.smoke_required_captures and not app.smoke_errors else 1
    if args.ui_log_smoke:
        return 0 if not app.log_smoke_errors else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

