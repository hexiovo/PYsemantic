"""Verify nondefault advanced parameters through the packaged worker boundary."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_toolkit.analysis import AnalysisConfig
from semantic_toolkit.data_io import Corpus, Document
from semantic_toolkit.exploration import build_profiles
from semantic_toolkit.runtime import corpus_to_dict, load_project_full


def run(executable: Path, output: Path):
    executable, output = executable.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    corpus = Corpus([Document(str(i), text + " ABC 123 https://example.org", "SYNTHETIC_ONLY") for i, text in enumerate([
        "手机 电池 拍照 清晰", "手机 电池 续航 清晰", "足球 球队 比赛 训练", "足球 球队 比赛 得分"])])
    records = []
    for method in ["nmf", "lda", "tfidf-char"]:
        task = output / method
        task.mkdir(exist_ok=False)
        config = AnalysisConfig(method=method, stopwords="无", clusters=2, topic_count=2,
            min_token_length=2, remove_urls=True, remove_numbers=True, case_mode="统一小写",
            compute_neighbors=False, compute_silhouette=False, tfidf_sublinear_tf=True, tfidf_smooth_idf=False,
            topic_top_words=2, profile_top_words=3, representative_count=2,
            nmf_tol=.002, kmeans_tol=.002, lda_mean_change_tol=.002, lda_perp_tol=.05,
            lda_evaluate_every=1, lda_doc_topic_prior=.3, lda_topic_word_prior=.2, lda_max_iter=3)
        task_corpus = Corpus(corpus.documents + [Document("url-only", "https://example.org", "SYNTHETIC_ONLY")]) if method == "tfidf-char" else corpus
        request = {"operation": "analyze", "run_id": f"advanced-{method}", "method": method,
                   "config": asdict(config), "corpus": corpus_to_dict(task_corpus), "result_directory": str(task / "result")}
        (task / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        kwargs = {}
        if os.name == "nt":
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            kwargs.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
        subprocess.run([str(executable), "--runtime-worker", str(task)], check=True, timeout=120, **kwargs)
        _, results, _, _ = load_project_full(task / "result")
        result = results[method]
        assert result.config == asdict(config)
        assert len(result.cluster_labels) == 4 and result.nearest_neighbors.empty
        assert "123" not in result.word_vocabulary and "abc" in result.word_vocabulary
        assert not any("example" in term for term in result.word_vocabulary)
        if method == "tfidf-char":
            assert result.zero_document_ids == ["url-only"]
        else:
            assert (result.topic_terms.groupby("主题").size() == 2).all()
        assert all(profile.representative_count == 2 and len(profile.keywords) <= 3 for profile in build_profiles(result))
        events = [json.loads(line) for line in (task / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        skipped = {event.get("stage_id") for event in events if event["type"] == "stage-skipped"}
        assert {"neighbors", "silhouette"}.issubset(skipped)
        records.append({"method": method, "status": "PASS", "config": result.config})
    (output / "checks.json").write_text(json.dumps({"status": "PASS", "workers": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Advanced packaged workers: PASS (NMF, LDA, character TF-IDF)")


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
