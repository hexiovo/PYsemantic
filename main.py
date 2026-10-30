"""Chinese Semantic Toolkit desktop entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from semantic_toolkit.resources import close_startup_splash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线中文语义分析工具包")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", metavar="OUTPUT_DIR", help="在输出目录中运行集成烟测并导出全部方法结果")
    group.add_argument("--runtime-self-test", metavar="OUTPUT_DIR", help="验证本轮任务管理、ETA、项目、日志与取消集成")
    group.add_argument("--ui-smoke", metavar="EVIDENCE_DIR", help="打开五个界面页并用程序窗口截图后退出")
    group.add_argument("--ui-log-smoke", metavar="EVIDENCE_DIR", help="单独截取运行日志弹窗并退出")
    group.add_argument("--runtime-worker", metavar="TASK_DIR", help=argparse.SUPPRESS)
    group.add_argument("--check-environment", action="store_true", help="检查环境并打印报告")
    group.add_argument("--repair-environment", action="store_true", help="打开环境检查与依赖修复窗口")
    group.add_argument("--install-missing", action="store_true", help="安装源码环境缺失的固定运行依赖并复检")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        close_startup_splash()
        raise
    if any((args.runtime_worker, args.runtime_self_test, args.self_test, args.ui_smoke, args.ui_log_smoke,
            args.check_environment, args.repair_environment, args.install_missing)):
        # PyInstaller displays the one-file splash before Python can inspect CLI
        # arguments. Close it immediately for worker and smoke-test invocations.
        close_startup_splash()
    if args.check_environment:
        from semantic_toolkit.environment import format_environment_report, inspect_environment
        try:
            report = inspect_environment()
            if sys.stdout is not None:
                print(format_environment_report(report))
            return 0 if report.dependencies_ready else 1
        except Exception as exc:
            if sys.stderr is not None:
                print(f"环境检查失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.repair_environment:
        from semantic_toolkit.environment import launch_recovery_window
        return launch_recovery_window()
    if args.install_missing:
        from semantic_toolkit.environment import (build_install_command, format_environment_report,
                                                  inspect_environment, run_install_command)
        try:
            report = inspect_environment()
            if report.frozen:
                if sys.stdout is not None:
                    print(format_environment_report(report))
                    print("打包 EXE 不能通过 pip 修改内置运行时。")
                return 2
            command = build_install_command(report.missing_dependencies, frozen=report.frozen)
            if not command:
                if sys.stdout is not None:
                    print(format_environment_report(report))
                return 0 if report.dependencies_ready else 1
            code, cancelled = run_install_command(command, on_output=print if sys.stdout is not None else None)
            if cancelled:
                return 2
            if code:
                return code
            refreshed = inspect_environment()
            if sys.stdout is not None:
                print(format_environment_report(refreshed))
                print("请重启程序后使用刚安装的依赖。")
            return 0 if refreshed.dependencies_ready else 1
        except Exception as exc:
            if sys.stderr is not None:
                print(f"依赖安装失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
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
    from semantic_toolkit.environment import (launch_recovery_window, missing_import_specs,
                                              should_open_recovery)
    try:
        missing = missing_import_specs()
        if missing:
            close_startup_splash()
            return launch_recovery_window(initial_error=f"缺少主界面所需运行包：{', '.join(missing)}")
        from semantic_toolkit.ui import SemanticApp
    except (ImportError, OSError) as exc:
        close_startup_splash()
        if should_open_recovery(exc):
            return launch_recovery_window(initial_error=f"{type(exc).__name__}: {exc}")
        raise
    try:
        smoke_output = args.ui_smoke or args.ui_log_smoke
        smoke_root = Path(smoke_output) / "test-profile" if smoke_output else None
        app = SemanticApp(ui_smoke_dir=args.ui_smoke, runtime_root=smoke_root)
        if args.ui_log_smoke:
            app.start_log_viewer_smoke(Path(args.ui_log_smoke))
        app.mainloop()
    except BaseException:
        close_startup_splash()
        raise
    finally:
        close_startup_splash()
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

