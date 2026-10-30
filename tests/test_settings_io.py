"""Boundary tests for portable JSON settings and preset imports."""
from pathlib import Path
import tempfile
import unittest

from semantic_toolkit.analysis import AnalysisConfig, METHODS
from semantic_toolkit.settings_io import (
    FORMAT_NAME,
    decode_settings_document,
    presets_document,
    read_settings_file,
    settings_document,
    write_settings_file,
)


class SettingsIOTests(unittest.TestCase):
    def test_config_and_versioned_settings_round_trip(self):
        config = AnalysisConfig(method="tfidf-word", segmentation="字符", stopwords="标准内置")
        document = settings_document({**config.__dict__, "case_mode": "统一小写", "split_mode": "每段"})
        decoded = decode_settings_document({**document, "future_optional": "ignored"})
        self.assertEqual(decoded["settings"]["method"], "tfidf-word")
        self.assertEqual(decoded["settings"]["segmentation"], "字符")
        self.assertEqual(decoded["settings"]["case_mode"], "统一小写")

    def test_legacy_raw_settings_and_localized_method_label(self):
        decoded = decode_settings_document({
            "method": METHODS["nmf"], "segmentation": "jieba搜索模式",
            "stopwords": "自定义文件", "word_ngram_min": "1", "word_ngram_max": "2",
        })
        self.assertEqual(decoded["settings"]["method"], "nmf")
        self.assertEqual(decoded["settings"]["word_ngram_max"], 2)

    def test_legacy_presets_allow_settings_field_names_as_preset_names(self):
        settings = {"method": "bge", "segmentation": "jieba精确模式"}
        decoded = decode_settings_document({"method": settings, "settings": settings, "format": settings})
        self.assertEqual(set(decoded["presets"]), {"method", "settings", "format"})

    def test_preset_export_import_file_round_trip(self):
        document = presets_document({"基准": {"method": "bge"}, "词项": {"method": "tfidf-word"}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "presets.json"
            write_settings_file(path, document)
            decoded = read_settings_file(path)
        self.assertEqual(set(decoded["presets"]), {"基准", "词项"})

    def test_explicit_format_and_version_are_required_and_checked(self):
        with self.assertRaisesRegex(ValueError, "未知文件格式"):
            decode_settings_document({"format": "other", "version": 1, "settings": {}})
        with self.assertRaisesRegex(ValueError, "版本"):
            decode_settings_document({"format": FORMAT_NAME, "version": 2, "settings": {}})

    def test_invalid_boundary_values_are_rejected(self):
        invalid = [
            {"method": "made-up"},
            {"compute_clusters": "false"},
            {"max_df": float("nan")},
            {"neighbors_k": 101},
            {"word_ngram_min": 3, "word_ngram_max": 2},
            {"unknown_option": True},
            {"selected_paths": ["ok.txt", 1]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_settings_document(value)


if __name__ == "__main__":
    unittest.main()
