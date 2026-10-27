from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from semantic_toolkit.analysis import AnalysisConfig, run_analysis
from semantic_toolkit.data_io import Corpus, Document
from semantic_toolkit.exploration import build_profiles, filter_rows, page_rows, profile_table
from semantic_toolkit.runtime import (corpus_to_dict, create_project_from_caches, load_project_full,
                                      save_project, save_result_cache)
from semantic_toolkit.worker import run_worker


class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus([
            Document("phone.[1]", "苹果 手机 电池 摄影 清晰", "test.csv", 2),
            Document("phone-2", "安卓 手机 电池 续航 清晰", "test.csv", 3),
            Document("sport-1", "足球 球队 训练 比赛 得分", "test.csv", 4),
            Document("sport-2", "篮球 球队 训练 比赛 得分", "test.csv", 5),
            Document("park-1", "城市 公园 天气 散步 温暖", "test.csv", 6),
            Document("park-2", "公园 城市 天气 步行 温暖", "test.csv", 7),
        ])
        cls.result = run_analysis(cls.corpus, AnalysisConfig(method="tfidf-word", clusters=3))

    def test_literal_id_filter_either_endpoint_and_source_cluster(self):
        frame = pd.DataFrame({"文档ID": ["A", "phone.[1]", "AB"],
                              "近邻ID": ["phone.[1]", "B", "C"], "余弦相似度": [.8, .4, -.1]})
        filtered = filter_rows(frame, query=".[1]", cluster_ids={"A"}, minimum=.8, maximum=.8)
        self.assertEqual(filtered["文档ID"].tolist(), ["A"])
        self.assertTrue(filter_rows(frame, query="missing").empty)
        self.assertTrue(filter_rows(frame, cluster_ids=set()).empty)
        for lo, hi in [(float("nan"), None), (None, float("inf")), (1.1, None), (.5, .1)]:
            with self.assertRaises(ValueError):
                filter_rows(frame, minimum=lo, maximum=hi)

    def test_pagination_covers_every_row_and_clamps_after_filter(self):
        frame = pd.DataFrame({"文档ID": [str(i) for i in range(731)]})
        all_ids = []
        for page in range(15):
            chunk, actual, count = page_rows(frame, page, 50)
            self.assertEqual(count, 15)
            self.assertEqual(actual, page)
            all_ids.extend(chunk["文档ID"])
        self.assertEqual(all_ids, frame["文档ID"].tolist())
        self.assertEqual(page_rows(frame.iloc[:2], 14, 50)[1:], (0, 1))
        self.assertTrue(page_rows(frame.iloc[:0], 14, 50)[0].empty)

    def test_cluster_representatives_match_direct_distance_oracle(self):
        for matrix in (self.result.doc_matrix, self.result.doc_matrix.toarray()):
            result = replace(self.result, doc_matrix=matrix)
            profiles = build_profiles(result)
            self.assertEqual(sum(len(profile.member_ids) for profile in profiles), len(result.doc_ids))
            for profile in profiles:
                ids = self.result.cluster_labels.loc[self.result.cluster_labels["簇"] == int(profile.key.split(":")[1]), "文档ID"].tolist()
                indices = [result.doc_ids.index(doc_id) for doc_id in ids]
                dense = matrix[indices].toarray() if sparse.issparse(matrix) else matrix[indices]
                distances = np.linalg.norm(dense - dense.mean(axis=0), axis=1)
                observed = dict(zip(profile.member_ids, profile.scores))
                np.testing.assert_allclose([observed[doc_id] for doc_id in ids], distances, atol=1e-5)
                self.assertEqual(profile.scores, sorted(profile.scores))
                self.assertTrue(profile.keywords)

    def test_nmf_assignment_empty_topic_and_zero_exclusion(self):
        weights = np.array([[2, 1, 0], [1, 3, 0], [0, 0, 0], [1, 1, 0], [0, 5, 0], [7, 0, 0]], dtype=float)
        result = replace(self.result, method="nmf", topic_document_matrix=weights,
                         topic_weight_matrix=np.ones((3, len(self.result.word_vocabulary))),
                         zero_document_ids=["park-2"])
        profiles = [p for p in build_profiles(result) if p.kind == "主题"]
        self.assertEqual(profiles[0].member_ids, ["phone.[1]", "sport-2"])
        self.assertEqual(profiles[1].member_ids, ["park-1", "phone-2"])
        self.assertEqual(profiles[2].member_ids, [])

    def test_lsa_signed_profiles_do_not_imply_probabilities(self):
        weights = np.array([[2, -1], [-3, 1], [0, 0], [1, 2], [-1, -2], [0, 0]], dtype=float)
        terms = np.zeros((2, len(self.result.word_vocabulary)))
        terms[0, :2] = [2, -4]
        profiles = build_profiles(replace(self.result, method="lsa", topic_document_matrix=weights,
                                         topic_weight_matrix=terms, cluster_labels=self.result.cluster_labels.iloc[:0]))
        self.assertEqual(profiles[0].member_ids, ["phone.[1]", "sport-2"])
        self.assertEqual(profiles[1].member_ids, ["phone-2", "park-1"])
        self.assertEqual(profiles[1].scores, [-3, -1])
        self.assertEqual(profiles[1].keywords, [self.result.word_vocabulary[1]])
        self.assertIn("不是互斥主题或概率", profiles[0].explanation)

    def test_cache_project_worker_export_and_backward_compatibility(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cache = save_result_cache(root / "cache", self.corpus, self.result)
            cache_corpus, _, _, _ = load_project_full(cache)
            self.assertTrue(all(not doc.text for doc in cache_corpus.documents))
            metadata = {"tfidf-word": {"cluster_names": {"cluster:1": "手机与电池"}}}
            saved = create_project_from_caches(root / "project", corpus_to_dict(self.corpus), [cache], runtime_metadata=metadata)
            corpus, results, _, restored = load_project_full(saved)
            self.assertEqual(corpus.texts, self.corpus.texts)
            self.assertEqual(restored, metadata)
            self.assertIn("手机与电池", profile_table(build_profiles(results["tfidf-word"]), restored["tfidf-word"]["cluster_names"])["名称"].tolist())
            old = save_project(root / "old", self.corpus, {"tfidf-word": self.result})
            self.assertEqual(load_project_full(old)[3], {})
            task = root / "worker"
            task.mkdir()
            request = {"run_id": "export-test", "operation": "export", "method": "tfidf-word",
                       "result_directory": str(saved), "parent_directory": str(root / "exports"),
                       "staging_directory": str(root / "exports" / ".pending")}
            (task / "request.json").write_text(json.dumps(request), encoding="utf-8")
            self.assertEqual(run_worker(task), 0)
            output = Path(json.loads((task / "output.json").read_text(encoding="utf-8"))["path"])
            frame = pd.read_csv(output / "group_profiles.csv")
            self.assertIn("手机与电池", frame["名称"].tolist())
            self.assertEqual(len(pd.read_csv(output / "group_members.csv")), len(self.corpus.documents))


if __name__ == "__main__":
    unittest.main()
