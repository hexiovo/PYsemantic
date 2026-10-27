from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

from semantic_toolkit.analysis import AnalysisConfig, _fit_word_matrices, _prepare_text, _tokens_for, run_analysis
from semantic_toolkit.data_io import Corpus, Document
from semantic_toolkit.eta import estimate_stage_plan
from semantic_toolkit.exploration import build_profiles, profile_table
from semantic_toolkit.exports import export_results


class AdvancedParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus([Document("a", "手机 电池 拍照 清晰", "synthetic"),
                            Document("b", "手机 电池 续航 清晰", "synthetic"),
                            Document("c", "足球 球队 比赛 训练", "synthetic"),
                            Document("d", "足球 球队 比赛 得分", "synthetic")])
        cls.config = AnalysisConfig(method="tfidf-word", stopwords="无", clusters=2)

    def test_new_parameter_boundaries(self):
        for change in [{"min_token_length": 0}, {"min_token_length": 21}, {"representative_count": 51},
                       {"topic_top_words": 0}, {"profile_top_words": 101}, {"case_mode": "bad"},
                       {"nmf_tol": 0}, {"kmeans_tol": float("nan")}, {"lda_mean_change_tol": -1},
                       {"lda_perp_tol": float("inf")}, {"lda_doc_topic_prior": 0},
                       {"lda_topic_word_prior": -1}, {"lda_evaluate_every": 0}, {"lda_evaluate_every": 101},
                       {"remove_numbers": "false"}, {"compute_neighbors": 1}, {"compute_silhouette": None},
                       {"tfidf_smooth_idf": "True"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.config, **change).normalized()
        replace(self.config, lda_doc_topic_prior=None, lda_topic_word_prior=.3, lda_evaluate_every=1).normalized()

    def test_cleaning_preserves_source_and_filters_only_selected_tokens(self):
        cfg = replace(self.config, min_token_length=2, remove_numbers=True, remove_urls=True, case_mode="统一小写")
        text = "ABC x 123 12.5 A12 手机 https://example.org/path?q=3，中文保留 www.test.org"
        prepared = _prepare_text(text, cfg)
        self.assertNotIn("example", prepared)
        self.assertNotIn("test.org", prepared)
        self.assertIn("，中文保留", prepared)
        tokenizer = SimpleNamespace(cut=lambda value, cut_all=False: value.split())
        tokens = _tokens_for(prepared, cfg, set(), tokenizer)
        self.assertIn("abc", tokens)
        self.assertIn("a12", tokens)
        self.assertNotIn("123", tokens)
        self.assertNotIn("12.5", tokens)
        self.assertNotIn("x", tokens)
        corpus = Corpus([Document("a", text, "synthetic"), Document("b", "ABC 手机", "synthetic")])
        before = [doc.text for doc in corpus.documents]
        run_analysis(corpus, replace(cfg, compute_clusters=False))
        self.assertEqual([doc.text for doc in corpus.documents], before)

    def test_bge_receives_prepared_text_without_token_filtering(self):
        corpus = Corpus([Document("a", "ABC 123 https://test.org", "synthetic"),
                         Document("b", "DEF 手机", "synthetic")])
        cfg = replace(self.config, method="bge", remove_urls=True, remove_numbers=True,
                      min_token_length=2, case_mode="统一小写", compute_clusters=False)
        with patch("semantic_toolkit.analysis._encode_bge", return_value=(np.ones((2, 3), dtype=np.float32), 0)) as encode:
            result = run_analysis(corpus, cfg)
        prepared = encode.call_args.args[0]
        self.assertIn("123", prepared[0])  # Number filtering is a token-level option.
        self.assertNotIn("https", prepared[0])
        self.assertTrue(prepared[0].startswith("abc"))
        self.assertNotIn("123", result.word_vocabulary)

    def test_word_tfidf_matches_sklearn_for_all_weight_switches(self):
        documents = [["手机", "手机", "电池"], ["手机", "足球", "足球"], ["电池", "训练"]]
        for sublinear in [False, True]:
            for smooth in [False, True]:
                counts, actual, _ = _fit_word_matrices(documents, 100, 1, sublinear_tf=sublinear, smooth_idf=smooth)
                expected = TfidfTransformer(sublinear_tf=sublinear, smooth_idf=smooth).fit_transform(counts)
                np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=1e-6)

    def test_char_tfidf_case_policy_and_weight_options(self):
        corpus = Corpus([Document("a", "AAA bb", "synthetic"), Document("b", "AA cc", "synthetic")])
        for mode in ["方法默认", "统一小写", "保持原样"]:
            cfg = replace(self.config, method="tfidf-char", case_mode=mode, tfidf_sublinear_tf=True, tfidf_smooth_idf=False)
            result = run_analysis(corpus, cfg)
            expected = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), lowercase=mode != "保持原样",
                                      sublinear_tf=True, smooth_idf=False).fit_transform([doc.text for doc in corpus.documents])
            np.testing.assert_allclose(result.doc_matrix.toarray(), expected.toarray(), rtol=1e-6)
            self.assertEqual("A" in result.doc_feature_labels, mode == "保持原样")

    def test_url_only_text_has_zero_character_representation_after_cleaning(self):
        corpus = Corpus([Document("url", "https://example.org", "synthetic"),
                         Document("a", "手机电池", "synthetic"), Document("b", "足球比赛", "synthetic")])
        result = run_analysis(corpus, replace(self.config, method="tfidf-char", remove_urls=True))
        self.assertIn("url", result.zero_document_ids)
        self.assertFalse((result.nearest_neighbors["文档ID"] == "url").any())

    def test_external_vector_sublinear_weights_match_tfidf_oracle(self):
        corpus = Corpus([Document("a", "aaab", "synthetic"), Document("b", "abbc", "synthetic")])
        with TemporaryDirectory() as temp:
            vectors = Path(temp) / "vectors.txt"
            vectors.write_text("3 2\na 1 0\nb 0 1\nc 1 1\n", encoding="utf-8")
            for sublinear in [False, True]:
                for smooth in [False, True]:
                    cfg = replace(self.config, method="word2vec", segmentation="字符", word2vec_path=str(vectors),
                                  tfidf_sublinear_tf=sublinear, tfidf_smooth_idf=smooth)
                    result = run_analysis(corpus, cfg)
                    mapping = {"a": [1, 0], "b": [0, 1], "c": [1, 1]}
                    weights = result.word_tfidf_matrix.toarray()
                    expected = weights @ np.array([mapping[word] for word in result.word_vocabulary]) / weights.sum(axis=1, keepdims=True)
                    np.testing.assert_allclose(result.doc_matrix, expected, rtol=1e-6)

    def test_neighbors_bypass_and_independent_silhouette_bypass(self):
        events = []
        cfg = replace(self.config, compute_neighbors=False, compute_silhouette=False)
        with patch("semantic_toolkit.analysis._nearest_neighbors") as neighbors, patch("semantic_toolkit.analysis.silhouette_score") as silhouette:
            result = run_analysis(self.corpus, cfg, event_callback=events.append)
        neighbors.assert_not_called()
        silhouette.assert_not_called()
        self.assertEqual(len(result.cluster_labels), 4)
        self.assertTrue(result.nearest_neighbors.empty)
        self.assertEqual(list(result.nearest_neighbors.columns), ["文档ID", "排名", "近邻ID", "余弦相似度"])
        skipped = {event["stage_id"] for event in events if event["type"] == "stage-skipped"}
        self.assertTrue({"neighbors", "silhouette"}.issubset(skipped))
        plan = estimate_stage_plan([cfg.method], 4, 100, {cfg.method: asdict(cfg)})
        stages = {stage.stage_id for stage in plan}
        self.assertIn("kmeans", stages)
        self.assertFalse({"neighbors", "silhouette"} & stages)

    def test_tolerances_and_priors_forwarded_with_real_estimators(self):
        cfg = replace(self.config, method="lda", topic_count=2, lda_max_iter=3, lda_doc_topic_prior=.3,
                      lda_topic_word_prior=.2, lda_mean_change_tol=.002, lda_perp_tol=.2, lda_evaluate_every=1, kmeans_tol=.002)
        with patch("semantic_toolkit.analysis.LatentDirichletAllocation", wraps=LatentDirichletAllocation) as estimator, \
             patch("semantic_toolkit.analysis.KMeans", wraps=KMeans) as clustering:
            result = run_analysis(self.corpus, cfg)
        self.assertEqual(estimator.call_args.kwargs["doc_topic_prior"], .3)
        self.assertEqual(estimator.call_args.kwargs["topic_word_prior"], .2)
        self.assertEqual(estimator.call_args.kwargs["mean_change_tol"], .002)
        self.assertEqual(estimator.call_args.kwargs["perp_tol"], .2)
        self.assertEqual(estimator.call_args.kwargs["evaluate_every"], 1)
        self.assertEqual(clustering.call_args.kwargs["tol"], .002)
        np.testing.assert_allclose(result.topic_document_matrix.sum(axis=1), 1, atol=1e-6)
        with patch("semantic_toolkit.analysis.NMF", wraps=NMF) as estimator:
            run_analysis(self.corpus, replace(cfg, method="nmf", nmf_tol=.002))
        self.assertEqual(estimator.call_args.kwargs["tol"], .002)

    def test_profile_and_export_counts_and_old_result_defaults(self):
        cfg = replace(self.config, method="nmf", topic_count=2, topic_top_words=2, profile_top_words=1, representative_count=1)
        result = run_analysis(self.corpus, cfg)
        self.assertTrue((result.topic_terms.groupby("主题").size() == 2).all())
        profiles = build_profiles(result)
        self.assertTrue(all(len(profile.keywords) <= 1 for profile in profiles))
        self.assertTrue(all(profile.representative_count == 1 for profile in profiles))
        table = profile_table(profiles)
        self.assertTrue(all("、" not in value for value in table["代表文档ID"]))
        with TemporaryDirectory() as temp:
            folder = export_results(result, self.corpus, temp)
            exported = pd.read_csv(folder / "group_profiles.csv", keep_default_na=False)
            self.assertEqual(exported["代表文档ID"].tolist(), table["代表文档ID"].tolist())
            parameters = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
            self.assertEqual(parameters["representative_count"], 1)
        old = replace(result, config={"method": "nmf"})
        self.assertTrue(all(profile.representative_count == 3 for profile in build_profiles(old)))


if __name__ == "__main__":
    unittest.main()
