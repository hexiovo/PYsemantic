from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from semantic_toolkit.analysis import AnalysisConfig, _create_bge_session, run_analysis
from semantic_toolkit.data_io import Corpus, Document


class _FakeSessionOptions:
    def __init__(self):
        self.execution_mode = "parallel"
        self.enable_mem_pattern = True
        self.intra_op_num_threads = 0


class _FakeSession:
    def __init__(self, providers):
        self.providers = list(providers)

    def get_providers(self):
        return list(self.providers)

    def get_inputs(self):
        return [SimpleNamespace(name=name, type="tensor(int64)") for name in
                ("input_ids", "attention_mask", "token_type_ids")]

    def get_outputs(self):
        return [SimpleNamespace(name="last_hidden_state")]

    def run(self, output_names, feed):
        sequence_length = feed["input_ids"].shape[1]
        return [np.ones((1, sequence_length, 3), dtype=np.float32)]


class _FakeOrt:
    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    SessionOptions = _FakeSessionOptions

    def __init__(self, available, active_by_request=None):
        self.available = list(available)
        self.active_by_request = active_by_request or {}
        self.calls = []

    def get_available_providers(self):
        return list(self.available)

    def InferenceSession(self, model_path, *, sess_options, providers):
        self.calls.append({"model_path": model_path, "options": sess_options, "providers": list(providers)})
        active = self.active_by_request.get(providers[0], providers)
        return _FakeSession(active)


class AccelerationTests(unittest.TestCase):
    def test_acceleration_config_defaults_and_boundaries(self):
        cfg = AnalysisConfig().normalized()
        self.assertFalse(cfg.parallel_enabled)
        self.assertEqual(cfg.cpu_workers, 4)
        self.assertEqual(cfg.bge_device, "cpu")
        for change in ({"parallel_enabled": "false"}, {"parallel_enabled": 1},
                       {"cpu_workers": 0}, {"cpu_workers": 33}, {"cpu_workers": True},
                       {"cpu_workers": 1.5}, {"bge_device": "gpu"}, {"bge_device": "CUDA"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(cfg, **change).normalized()
        for workers in (1, 32):
            replace(cfg, parallel_enabled=True, cpu_workers=workers).normalized()
        for device in ("cpu", "auto", "cuda", "directml"):
            replace(cfg, bge_device=device).normalized()

    def test_directml_session_uses_required_options_and_reports_active_provider(self):
        ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
        cfg = AnalysisConfig(parallel_enabled=True, cpu_workers=3, bge_device="directml").normalized()

        session, info = _create_bge_session(ort, "model.onnx", cfg)

        self.assertEqual(session.get_providers(), ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(info["selected_provider"], "DmlExecutionProvider")
        self.assertEqual(info["active_providers"], ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(ort.calls[0]["providers"], ["DmlExecutionProvider", "CPUExecutionProvider"])
        options = ort.calls[0]["options"]
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.execution_mode, _FakeOrt.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(options.intra_op_num_threads, 3)

    def test_explicit_unavailable_or_fallen_back_gpu_provider_fails(self):
        unavailable = _FakeOrt(["CPUExecutionProvider"])
        with self.assertRaisesRegex(RuntimeError, "未提供 DmlExecutionProvider"):
            _create_bge_session(unavailable, "model.onnx", AnalysisConfig(bge_device="directml").normalized())
        self.assertEqual(unavailable.calls, [])

        silently_falling_back = _FakeOrt(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            {"CUDAExecutionProvider": ["CPUExecutionProvider"]},
        )
        with self.assertRaisesRegex(RuntimeError, "避免静默回退"):
            _create_bge_session(silently_falling_back, "model.onnx", AnalysisConfig(bge_device="cuda").normalized())

    def test_auto_mode_falls_back_to_cpu_and_preserves_reason(self):
        ort = _FakeOrt(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            {"CUDAExecutionProvider": ["CPUExecutionProvider"]},
        )

        session, info = _create_bge_session(ort, "model.onnx", AnalysisConfig(bge_device="auto").normalized())

        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        self.assertEqual(info["selected_provider"], "CPUExecutionProvider")
        self.assertIn("CUDAExecutionProvider", info["fallback_reason"])
        self.assertEqual([call["providers"] for call in ort.calls], [
            ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]
        ])

    def test_bge_provider_is_reported_in_warning_result_and_stage_event(self):
        class FakeTokenizer:
            @classmethod
            def from_file(cls, _path):
                return cls()

            def no_truncation(self):
                pass

            def encode(self, text, add_special_tokens=True):
                if not text and add_special_tokens:
                    return SimpleNamespace(ids=[101, 102])
                if not text:
                    return SimpleNamespace(ids=[])
                return SimpleNamespace(ids=[11, 12])

        ort = _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"])
        events = []
        corpus = Corpus([Document("a", "手机电池", "synthetic"), Document("b", "足球比赛", "synthetic")])
        with TemporaryDirectory() as temp:
            model_dir = Path(temp)
            (model_dir / "onnx").mkdir()
            (model_dir / "onnx" / "model.onnx").touch()
            (model_dir / "tokenizer.json").touch()
            with patch("semantic_toolkit.analysis._bge_model_dir", return_value=model_dir), \
                    patch.dict("sys.modules", {
                        "onnxruntime": ort,
                        "tokenizers": SimpleNamespace(Tokenizer=FakeTokenizer),
                    }):
                result = run_analysis(
                    corpus,
                    AnalysisConfig(method="bge", bge_device="directml", stopwords="无",
                                   compute_clusters=False, compute_similarity=False),
                    event_callback=events.append,
                )

        self.assertTrue(any("DmlExecutionProvider" in warning and "不保证模型中的每个算子" in warning
                            for warning in result.warnings))
        provider_events = [event for event in events if event.get("stage_id") == "bge-model-load"
                           and event.get("selected_provider")]
        self.assertEqual(len(provider_events), 1)
        self.assertEqual(provider_events[0]["selected_provider"], "DmlExecutionProvider")
        self.assertIn("DmlExecutionProvider", provider_events[0]["active_providers"])

    def test_lda_single_and_two_worker_outputs_are_numerically_close(self):
        corpus = Corpus([
            Document("a", "手机 电池 拍照 屏幕 清晰", "synthetic"),
            Document("b", "手机 电池 续航 屏幕 耐用", "synthetic"),
            Document("c", "手机 相机 拍照 清晰 轻便", "synthetic"),
            Document("d", "电池 续航 充电 耐用 快速", "synthetic"),
            Document("e", "足球 球队 比赛 训练 进球", "synthetic"),
            Document("f", "足球 球队 训练 得分 比赛", "synthetic"),
            Document("g", "比赛 球员 进球 得分 球队", "synthetic"),
            Document("h", "球员 足球 训练 进球 快速", "synthetic"),
        ])
        base = AnalysisConfig(method="lda", stopwords="无", topic_count=2, clusters=2,
                              compute_similarity=False, lda_max_iter=4, random_state=31)

        one = run_analysis(corpus, replace(base, parallel_enabled=True, cpu_workers=1))
        two = run_analysis(corpus, replace(base, parallel_enabled=True, cpu_workers=2))

        np.testing.assert_allclose(one.topic_document_matrix, two.topic_document_matrix, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(one.topic_weight_matrix, two.topic_weight_matrix, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
