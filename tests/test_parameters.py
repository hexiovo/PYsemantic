from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

from semantic_toolkit.analysis import AnalysisConfig, _fit_word_matrices, run_analysis
from semantic_toolkit.data_io import Corpus, Document
from semantic_toolkit.eta import estimate_stage_plan
from semantic_toolkit.exports import export_results
from semantic_toolkit.runtime import RuntimeStore, corpus_to_dict, load_project_full, save_project
from semantic_toolkit.worker import run_worker


class ParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus([
            Document("a", "共同 手机 电池 摄影", "synthetic"),
            Document("b", "共同 手机 电池 续航", "synthetic"),
            Document("c", "共同 足球 比赛 训练", "synthetic"),
            Document("d", "共同 足球 比赛 得分", "synthetic"),
        ])
        cls.config = AnalysisConfig(method="tfidf-word", stopwords="无", clusters=2)

    def test_default_compatibility_and_validation(self):
        cfg = AnalysisConfig(**{"method": "tfidf-word", "clusters": 2}).normalized()
        self.assertEqual((cfg.max_df, cfg.neighbors_k, cfg.compute_clusters, cfg.random_state), (1., 10, True, 42))
        self.assertEqual((cfg.nmf_max_iter, cfg.lda_max_iter), (300, 20))
        for change in [{"max_df": 0}, {"max_df": 1.1}, {"max_df": float("nan")}, {"max_df": float("inf")},
                       {"word_ngram_min": 3, "word_ngram_max": 2}, {"char_ngram_max": 6},
                       {"word_ngram_min": 1.5}, {"neighbors_k": 0}, {"neighbors_k": 101},
                       {"nmf_max_iter": 0}, {"lda_max_iter": 1001}, {"random_state": -1},
                       {"random_state": 2**32}, {"random_state": True}, {"compute_clusters": "false"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(cfg, **change).normalized()
        replace(cfg, random_state=2**32 - 1, word_ngram_max=3, char_ngram_max=5).normalized()

    def test_max_df_and_word_ngrams_have_exact_counts(self):
        counts, _, vocab = _fit_word_matrices(
            [["共同", "手机", "电池"], ["共同", "手机", "续航"], ["共同", "足球", "比赛"]],
            100, 1, max_df=.8, ngram_range=(1, 2))
        self.assertNotIn("共同", vocab)
        self.assertIn("手机 电池", vocab)
        np.testing.assert_array_equal(counts[:, vocab.index("共同 手机")].toarray().ravel(), [1, 1, 0])
        with self.assertRaisesRegex(ValueError, "最大文档比例"):
            _fit_word_matrices([["a"], ["b"]], 100, 2, max_df=.5)

    def test_word_bigram_analysis_and_top_k(self):
        cfg = replace(self.config, word_ngram_min=2, word_ngram_max=2, neighbors_k=1, max_df=.75)
        result = run_analysis(self.corpus, cfg)
        self.assertTrue(all(len(term.split()) == 2 for term in result.word_vocabulary))
        self.assertEqual(len(result.nearest_neighbors), 4)
        self.assertTrue((result.nearest_neighbors["排名"] == 1).all())
        self.assertTrue((result.nearest_neighbors["文档ID"] != result.nearest_neighbors["近邻ID"]).all())
        expanded = run_analysis(self.corpus, replace(cfg, neighbors_k=100))
        self.assertEqual(len(expanded.nearest_neighbors), 12)

    def test_char_range_is_independent_of_word_range(self):
        result = run_analysis(self.corpus, replace(self.config, method="tfidf-char", char_ngram_min=3,
                                                  char_ngram_max=3, word_ngram_min=2, word_ngram_max=2))
        self.assertTrue(all(len(term) == 3 for term in result.doc_feature_labels))
        self.assertTrue(all(" " not in term for term in result.word_vocabulary))

    def test_disabled_clustering_skips_estimators_and_eta(self):
        events = []
        cfg = replace(self.config, compute_clusters=False)
        with patch("semantic_toolkit.analysis.KMeans") as estimator, patch("semantic_toolkit.analysis.silhouette_score") as score:
            result = run_analysis(self.corpus, cfg, event_callback=events.append)
        estimator.assert_not_called()
        score.assert_not_called()
        self.assertTrue(result.cluster_labels.empty)
        self.assertEqual(result.clustering_summary["状态"], "已关闭聚类")
        skipped = {event["stage_id"] for event in events if event["type"] == "stage-skipped"}
        self.assertTrue({"kmeans", "silhouette"}.issubset(skipped))
        plan = estimate_stage_plan([cfg.method], 4, 100, {cfg.method: asdict(cfg)})
        self.assertFalse({"kmeans", "silhouette"} & {stage.stage_id for stage in plan})
        self.assertFalse(result.nearest_neighbors.empty)

    def test_iteration_limits_and_seed_reach_real_estimators(self):
        for method, name, real in [("nmf", "NMF", NMF), ("lda", "LatentDirichletAllocation", LatentDirichletAllocation),
                                   ("lsa", "TruncatedSVD", TruncatedSVD)]:
            cfg = replace(self.config, method=method, topic_count=2, random_state=7, nmf_max_iter=100, lda_max_iter=3)
            with self.subTest(method=method), patch(f"semantic_toolkit.analysis.{name}", wraps=real) as estimator:
                result = run_analysis(self.corpus, cfg)
                self.assertEqual(estimator.call_args.kwargs["random_state"], 7)
                if method in {"nmf", "lda"}:
                    self.assertEqual(estimator.call_args.kwargs["max_iter"], 100 if method == "nmf" else 3)
            repeated = run_analysis(self.corpus, cfg)
            np.testing.assert_allclose(result.doc_matrix, repeated.doc_matrix, rtol=1e-6, atol=1e-7)
            self.assertEqual(result.clustering_summary["方法"], "KMeans(random_state=7)")
            np.testing.assert_array_equal(result.cluster_labels["簇"], repeated.cluster_labels["簇"])

    def test_zero_vectors_excluded_and_empty_vocabulary_explained(self):
        corpus = Corpus([Document("a", "共同 手机", "synthetic"), Document("b", "共同 足球", "synthetic"),
                         Document("zero", "共同", "synthetic")])
        result = run_analysis(corpus, replace(self.config, max_df=.8, neighbors_k=100, compute_clusters=False))
        self.assertEqual(result.zero_document_ids, ["zero"])
        self.assertEqual(len(result.nearest_neighbors), 2)
        for method in ["tfidf-word", "tfidf-char"]:
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "没有可用"):
                run_analysis(Corpus([Document("a", "手机", "synthetic"), Document("b", "手机", "synthetic")]),
                             replace(self.config, method=method, max_df=.5))

    def test_worker_presets_projects_and_export_round_trip(self):
        cfg = replace(self.config, max_df=.75, word_ngram_max=2, neighbors_k=2, compute_clusters=False,
                      random_state=19, nmf_max_iter=101, lda_max_iter=7, char_ngram_min=2, char_ngram_max=4,
                      min_token_length=2, remove_numbers=True, remove_urls=True, case_mode="统一小写",
                      topic_top_words=5, profile_top_words=4, representative_count=2, compute_neighbors=False,
                      compute_silhouette=False, tfidf_sublinear_tf=True, tfidf_smooth_idf=False,
                      nmf_tol=.002, kmeans_tol=.003, lda_mean_change_tol=.004, lda_perp_tol=.05,
                      lda_evaluate_every=2, lda_doc_topic_prior=.2, lda_topic_word_prior=.1)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            request = {"run_id": "parameter-test", "operation": "analyze", "method": cfg.method,
                       "config": asdict(cfg), "corpus": corpus_to_dict(self.corpus), "result_directory": str(root / "cache")}
            (root / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(run_worker(root), 0)
            _, results, _, _ = load_project_full(root / "cache")
            self.assertEqual(results[cfg.method].config, asdict(cfg))
            store = RuntimeStore(root / "runtime")
            store.save_preset("parameters", asdict(cfg))
            self.assertEqual(store.presets()["parameters"], asdict(cfg))
            save_project(root / "project", self.corpus, results, settings=asdict(cfg))
            _, loaded, settings, _ = load_project_full(root / "project")
            self.assertEqual(settings, asdict(cfg))
            self.assertEqual(loaded[cfg.method].config, asdict(cfg))
            exported = export_results(loaded[cfg.method], self.corpus, root / "export")
            payload = json.loads((exported / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual({key: payload[key] for key in asdict(cfg)}, asdict(cfg))


if __name__ == "__main__":
    unittest.main()
