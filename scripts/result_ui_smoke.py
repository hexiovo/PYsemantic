"""Deterministic source-linked result UI checks, with application-owned screenshots."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_toolkit.analysis import AnalysisConfig, run_analysis
from semantic_toolkit.data_io import Corpus, Document
from semantic_toolkit.runtime import load_project_full, save_project, save_result_cache
from semantic_toolkit.ui import SemanticApp


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    corpus = Corpus([Document(f"doc-{i:04d}", ["手机 电池 续航 摄影 清晰", "足球 球队 训练 比赛 得分", "公园 天气 城市 散步 温暖"][i % 3] + f"。样本 {i}。", "合成语料.csv", i + 2)
                     for i in range(361)], stats={"空值排除": 0, "重复排除": 0})
    result = run_analysis(corpus, AnalysisConfig(method="tfidf-word", clusters=3))
    topic_corpus = Corpus(corpus.documents[:12])
    topic_result = run_analysis(topic_corpus, AnalysisConfig(method="nmf", clusters=3, topic_count=3))
    app = SemanticApp(runtime_root=output / "profile")
    errors = []
    app.report_callback_exception = lambda kind, value, tb: errors.append("".join(traceback.format_exception(kind, value, tb)))

    def pump(seconds=.3):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.update()
            time.sleep(.01)
        assert not errors, "\n".join(errors)

    def capture(name, window=None):
        from PIL import ImageGrab
        window = window or app
        window.lift()
        pump()
        ImageGrab.grab(window=int(window.winfo_id())).save(output / name)

    def profiles_ready():
        deadline = time.monotonic() + 20
        while not app.explorer.profiles and time.monotonic() < deadline:
            pump(.1)
        assert app.explorer.profiles

    try:
        app.corpus = corpus
        app._analyzed(result)
        pump()
        profiles_ready()
        app.results_tab.set("文档指标")
        table = app.explorer.tables["metrics"]
        table.move(7)
        assert len(table.visible) == 11 and table.visible.iloc[-1]["文档ID"] == "doc-0360"
        capture("01-metrics-page-8.png")
        app.explorer.filters["metrics"][0].set("0360")
        app.explorer.apply_filter("metrics")
        assert len(table.data) == 1 and table.page == 0
        table.tree.selection_set("0")
        table.tree.event_generate("<Return>")
        pump()
        if not app.explorer.raw_windows:
            table.open_button.winfo_children()[0].invoke()
        assert app.explorer.raw_windows
        capture("02-document.png", app.explorer.raw_windows[-1])
        app.explorer.raw_windows[-1].destroy()
        app.results_tab.set("近邻")
        app.explorer.filters["neighbors"][0].set("doc-0000")
        app.explorer.filters["neighbors"][2].set("0.1")
        app.explorer.apply_filter("neighbors")
        neighbors = app.explorer.tables["neighbors"]
        assert len(neighbors.data)
        capture("03-neighbors-filter.png")
        neighbors.tree.selection_set("0")
        neighbors.open_button.winfo_children()[0].invoke()
        capture("04-neighbor-pair.png", app.explorer.raw_windows[-1])
        app.explorer.raw_windows[-1].destroy()
        app.results_tab.set("主题与聚类")
        app.explorer.show_profiles()
        capture("05-cluster-profiles.png")
        app.explorer.profile_table.tree.selection_set("0")
        pump()
        assert app.explorer.detail_window is not None
        app.explorer.name_var.set("长名称验证：手机、电池和摄影体验")
        app.explorer.rename_cluster()
        pump()
        assert app.result_runtime_metadata["tfidf-word"]["cluster_names"]
        app.explorer.members_table.move(1)
        capture("06-cluster-members.png", app.explorer.detail_window)
        save_project(output / "project", corpus, {result.method: result}, runtime_metadata=app.result_runtime_metadata)
        restored, results, settings, metadata = load_project_full(output / "project")
        app._project_opened({"corpus": restored, "results": results, "settings": settings, "metadata": metadata, "project_path": str(output / "project")})
        profiles_ready()
        assert "长名称验证" in app.explorer.names["cluster:1"]
        app.corpus = topic_corpus
        app._analyzed(topic_result)
        app.result_method_var.set(topic_result.method_label)
        app._select_result_method()
        profiles_ready()
        app.explorer.kind.set("主题画像")
        app.explorer.show_profiles()
        assert len(app.explorer.shown_profiles) == 3
        capture("07-topic-profiles.png")
        app.explorer.profile_table.tree.selection_set("0")
        pump()
        capture("08-topic-members.png", app.explorer.detail_window)
        cache = save_result_cache(output / "cache", corpus, result)
        cached, cached_results, _, _ = load_project_full(cache)
        app._project_opened({"corpus": cached, "results": cached_results, "settings": {}, "metadata": {}, "project_path": str(cache)})
        app.explorer.open_row(cached_results[result.method].nearest_neighbors.iloc[0])
        assert not any(doc.text for doc in app.explorer.documents.values())
        capture("09-cache-unavailable.png", app.explorer.raw_windows[-1])
        app._loaded(corpus)
        pump()
        assert app.explorer.result is None and app.explorer.tables["neighbors"].data.empty
        (output / "checks.json").write_text(json.dumps({"status": "passed", "documents": 361, "screenshots": 9,
                                                        "checks": ["pagination", "filters", "source-text", "neighbors", "profiles", "rename", "project-roundtrip", "method-switch", "text-free-cache", "new-corpus-reset"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        app.destroy()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
