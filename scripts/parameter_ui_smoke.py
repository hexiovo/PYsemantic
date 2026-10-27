"""Application-owned UI checks for parameter controls and saved settings."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import ImageGrab
import customtkinter as ctk
from semantic_toolkit.analysis import AnalysisConfig, METHODS, run_analysis
from semantic_toolkit.data_io import Corpus, Document
from semantic_toolkit.ui import SemanticApp


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    app = SemanticApp(runtime_root=output / "profile")
    errors = []
    app.report_callback_exception = lambda kind, value, tb: errors.append("".join(traceback.format_exception(kind, value, tb)))

    def pump():
        deadline = time.monotonic() + .3
        while time.monotonic() < deadline:
            app.update()
            time.sleep(.01)
        assert not errors, "\n".join(errors)

    def capture(filename, window=None):
        window = window or app
        window.lift()
        pump()
        ImageGrab.grab(window=int(window.winfo_id())).save(output / filename)

    try:
        app.show_page("methods")
        pump()
        baseline = app._config_from_ui()
        assert baseline == AnalysisConfig()
        capture("01-methods-top.png")
        values = {"max_df": ".85", "neighbors_k": "17", "random_state": "123",
                  "word_ngram_min": "1", "word_ngram_max": "2", "char_ngram_min": "2", "char_ngram_max": "4",
                  "nmf_max_iter": "500", "lda_max_iter": "45", "min_token_length": "2",
                  "topic_top_words": "4", "profile_top_words": "3", "representative_count": "2",
                  "nmf_tol": ".002", "kmeans_tol": ".003", "lda_mean_change_tol": ".004",
                  "lda_perp_tol": ".05", "lda_evaluate_every": "2", "lda_doc_topic_prior": ".2",
                  "lda_topic_word_prior": ".1"}
        for key, value in values.items():
            app.extra_parameter_vars[key].set(value)
        app.compute_clusters_var.set(False)
        app.case_mode_var.set("统一小写")
        for key, value in {"remove_numbers": True, "remove_urls": True, "compute_neighbors": False,
                           "compute_silhouette": False, "tfidf_sublinear_tf": True, "tfidf_smooth_idf": False}.items():
            app.extra_bool_vars[key].set(value)
        app.method_var.set(METHODS["tfidf-word"])
        app._update_method_hint()
        expected = app._config_from_ui()
        assert expected.neighbors_k == 17 and not expected.compute_clusters
        settings = app._settings_snapshot()
        app.runtime.save_preset("参数测试", settings)
        app.runtime.save_last_settings(settings)
        app._apply_settings(asdict(AnalysisConfig()))
        assert app._config_from_ui() == AnalysisConfig()
        app._apply_settings(app.runtime.presets()["参数测试"])
        assert app._config_from_ui() == expected
        assert app.runtime.load_last_settings() == settings
        app._set_busy_controls(True)
        assert app.clustering_check.cget("state") == "disabled"
        app._set_busy_controls(False)
        canvas = app.page_widgets["methods"]._parent_canvas
        app.update_idletasks()
        # Align the computation switches near the top of the visible page.
        region = [float(value) for value in canvas.cget("scrollregion").split()]
        canvas.yview_moveto(380 / max(1, region[3]))
        capture("02-clustering-switch.png")
        for index, (title, section) in enumerate(app.parameter_sections.items()):
            canvas.yview_moveto(max(0, section.winfo_y() - 10) / max(1, region[3]))
            capture(f"02-section-{index + 1}.png")
        for key, variable in app.extra_parameter_vars.items():
            assert variable.get() or key.endswith("prior")
        app.extra_parameter_vars["max_df"].set("nan")
        try:
            app._config_from_ui()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid max_df accepted")
        app._apply_settings({"method": "tfidf-word"})
        assert app._config_from_ui() == AnalysisConfig(method="tfidf-word")
        corpus = Corpus([Document(str(i), text, "SYNTHETIC_ONLY") for i, text in enumerate([
            "手机 电池 拍照 清晰", "手机 电池 续航 清晰", "足球 球队 比赛 训练", "足球 球队 比赛 得分"])])
        result = run_analysis(corpus, replace(expected, method="nmf", max_df=1, topic_count=2, clusters=2, compute_clusters=True))
        app.corpus = corpus
        app._analyzed(result)
        deadline = time.monotonic() + 15
        while not app.explorer.profiles and time.monotonic() < deadline:
            pump()
        assert app.explorer.profiles
        app.results_tab.set("主题与聚类")
        app.explorer.kind.set("聚类画像")
        app.explorer.show_profiles()
        app.explorer.select_profile(app.explorer.profile_table.data.iloc[0])
        detail = app.explorer.detail_window
        summary = next(child for child in detail.winfo_children() if isinstance(child, ctk.CTkTextbox))
        assert "最多 2 篇" in summary.get("1.0", "end")
        assert len(app.explorer.selected_profile.keywords) <= 3
        capture("03-configured-profile.png", detail)
        detail.destroy()
        for page in ["data", "results", "tasks", "help"]:
            app.show_page(page)
            capture(f"03-{page}.png")
        (output / "checks.json").write_text(json.dumps({"status": "PASS", "config": asdict(expected),
            "geometry": app.winfo_geometry(), "settings_round_trip": True, "old_settings_defaults": True,
            "invalid_parameter_rejected": True, "callback_errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        app.destroy()


if __name__ == "__main__":
    run(Path(sys.argv[1]))
