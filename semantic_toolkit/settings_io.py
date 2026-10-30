"""Portable, validated JSON import/export for application settings."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
from typing import Any

from .analysis import AnalysisConfig, METHODS


FORMAT_NAME = "pysemantic.settings"
FORMAT_VERSION = 1

_UI_FIELDS = {
    "split_mode", "text_column", "id_column", "deduplicate", "selected_paths",
    "batch_methods", "xlsx",
}
_SETTINGS_FIELDS = set(asdict(AnalysisConfig())) | _UI_FIELDS
_BOOL_FIELDS = {
    "word2vec_tfidf_weight", "compute_similarity", "compute_clusters", "compute_neighbors",
    "compute_silhouette", "remove_numbers", "remove_urls", "tfidf_sublinear_tf",
    "tfidf_smooth_idf", "deduplicate", "xlsx", "parallel_enabled",
}
_STRING_FIELDS = {
    "custom_stopwords_path", "user_dictionary_path", "word2vec_path", "text_column", "id_column",
}
_INT_RANGES: dict[str, tuple[int, int | None]] = {
    "max_features": (1, 20_000), "min_frequency": (1, None),
    "topic_count": (2, 100), "cooc_window": (2, 20), "cooc_vocab_size": (20, 1_000),
    "clusters": (2, 100), "neighbors_k": (1, 100), "word_ngram_min": (1, 3),
    "word_ngram_max": (1, 3), "char_ngram_min": (1, 5), "char_ngram_max": (1, 5),
    "nmf_max_iter": (1, 1_000), "lda_max_iter": (1, 1_000),
    "random_state": (0, 2**32 - 1), "min_token_length": (1, 20),
    "topic_top_words": (1, 100), "profile_top_words": (1, 100),
    "representative_count": (1, 50), "cpu_workers": (1, 32),
}
_FLOAT_FIELDS = {
    "max_df", "nmf_tol", "kmeans_tol", "lda_mean_change_tol", "lda_perp_tol",
    "lda_doc_topic_prior", "lda_topic_word_prior",
}
_OPTIONAL_FLOAT_FIELDS = {"lda_doc_topic_prior", "lda_topic_word_prior"}
_METHODS = set(METHODS)
_METHOD_LABELS = {label: method for method, label in METHODS.items()}
_SEGMENTATIONS = {"jieba精确模式", "jieba搜索模式", "字符"}
_STOPWORDS = {"无", "保守内置", "标准内置", "自定义文件"}
_CASE_MODES = {"方法默认", "统一小写", "保持原样"}
_SPLIT_MODES = {"整篇/每文件", "每行", "每段"}
_INT_TEXT = re.compile(r"^[+-]?\d+$")


def _bad(message: str) -> ValueError:
    return ValueError(f"设置 JSON 无效：{message}")


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _bad(f"“{key}”必须是数字。")
    if isinstance(value, str) and not value.strip():
        raise _bad(f"“{key}”不能为空。")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise _bad(f"“{key}”必须是数字。") from exc
    if not math.isfinite(number):
        raise _bad(f"“{key}”必须是有限数字。")
    return number


def validate_settings(value: Any) -> dict[str, Any]:
    """Validate a legacy snapshot or AnalysisConfig-shaped dictionary before use."""
    if not isinstance(value, dict):
        raise _bad("settings 必须是 JSON 对象。")
    unknown = sorted(set(value) - _SETTINGS_FIELDS)
    if unknown:
        raise _bad(f"包含不支持的设置项：{', '.join(map(str, unknown[:8]))}。")
    result = dict(value)

    enum_values = {
        "method": (_METHODS | set(_METHOD_LABELS), "受支持的分析方法"),
        "segmentation": (_SEGMENTATIONS, "受支持的分词选项"),
        "stopwords": (_STOPWORDS, "受支持的停词选项"),
        "case_mode": (_CASE_MODES, "受支持的大小写选项"),
        "split_mode": (_SPLIT_MODES, "受支持的切分选项"),
        "bge_device": ({"cpu", "auto", "cuda", "directml"}, "受支持的 BGE 运行设备"),
    }
    for key, (allowed, description) in enum_values.items():
        if key not in result:
            continue
        if not isinstance(result[key], str) or result[key] not in allowed:
            raise _bad(f"{key} 不是{description}。")
    if result.get("method") in _METHOD_LABELS:
        result["method"] = _METHOD_LABELS[result["method"]]

    for key in _BOOL_FIELDS & result.keys():
        if type(result[key]) is not bool:
            raise _bad(f"“{key}”必须是 true 或 false。")
    for key in _STRING_FIELDS & result.keys():
        if not isinstance(result[key], str):
            raise _bad(f"“{key}”必须是字符串。")

    if "selected_paths" in result:
        paths = result["selected_paths"]
        if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
            raise _bad("selected_paths 必须是字符串数组。")
    if "batch_methods" in result:
        methods = result["batch_methods"]
        if (not isinstance(methods, list) or any(not isinstance(method, str) or method not in _METHODS
                                                  for method in methods)):
            raise _bad("batch_methods 必须只包含受支持的方法 ID。")
        if len(set(methods)) != len(methods):
            raise _bad("batch_methods 不能包含重复方法。")

    for key, (low, high) in _INT_RANGES.items():
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, bool):
            raise _bad(f"“{key}”必须是整数。")
        if isinstance(value, str):
            if not _INT_TEXT.fullmatch(value.strip()):
                raise _bad(f"“{key}”必须是整数。")
            value = int(value)
        elif not isinstance(value, int):
            raise _bad(f"“{key}”必须是整数。")
        if value < low or high is not None and value > high:
            range_text = f"{low} 以上" if high is None else f"{low}–{high}"
            raise _bad(f"“{key}”必须在 {range_text}范围内。")
        result[key] = value

    for key in _FLOAT_FIELDS & result.keys():
        raw = result[key]
        if raw is None and key in _OPTIONAL_FLOAT_FIELDS:
            continue
        if raw == "" and key in _OPTIONAL_FLOAT_FIELDS:
            continue
        number = _number(raw, key)
        if key == "max_df":
            if not 0 < number <= 1:
                raise _bad("“max_df”必须大于 0 且不超过 1。")
        elif number <= 0:
            raise _bad(f"“{key}”必须是正数。")
        result[key] = number

    if "lda_evaluate_every" in result:
        value = result["lda_evaluate_every"]
        if isinstance(value, bool):
            raise _bad("“lda_evaluate_every”必须是 -1 或 1–100 的整数。")
        if isinstance(value, str):
            if not _INT_TEXT.fullmatch(value.strip()):
                raise _bad("“lda_evaluate_every”必须是 -1 或 1–100 的整数。")
            value = int(value)
        if not isinstance(value, int) or value != -1 and not 1 <= value <= 100:
            raise _bad("“lda_evaluate_every”必须是 -1 或 1–100 的整数。")
        result["lda_evaluate_every"] = value

    for lower, upper in (("word_ngram_min", "word_ngram_max"), ("char_ngram_min", "char_ngram_max")):
        if lower in result and upper in result and result[lower] > result[upper]:
            raise _bad(f"“{lower}”不能大于“{upper}”。")
    return result


def _validate_presets(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise _bad("presets 必须是包含至少一个项目的对象。")
    validated: dict[str, dict[str, Any]] = {}
    for raw_name, settings in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip() or len(raw_name.strip()) > 80:
            raise _bad("预设名称须为 1–80 个字符。")
        name = raw_name.strip()
        if "\n" in name or "\r" in name:
            raise _bad("预设名称不能包含换行。")
        if name in validated:
            raise _bad(f"预设名称规范化后重复：{name!r}。")
        validated[name] = validate_settings(settings)
    return validated


def decode_settings_document(value: Any) -> dict[str, Any]:
    """Return validated ``settings`` and/or ``presets`` from supported JSON forms.

    Unknown keys are ignored only in this explicitly versioned envelope. Legacy
    raw settings and raw preset maps reject unknown setting fields.
    """
    if not isinstance(value, dict) or not value:
        raise _bad("文件顶层必须是非空 JSON 对象。")

    envelope_signature = (value.get("format") == FORMAT_NAME
                          or ("format" in value and "version" in value
                              and not all(isinstance(item, dict) for item in value.values())))
    if envelope_signature:
        if value.get("format") != FORMAT_NAME:
            raise _bad(f"未知文件格式；预期 {FORMAT_NAME!r}。")
        version = value.get("version")
        if type(version) is not int or version != FORMAT_VERSION:
            raise _bad(f"不支持的设置文件版本：{version!r}。")
        result: dict[str, Any] = {}
        if "settings" in value:
            result["settings"] = validate_settings(value["settings"])
        if "presets" in value:
            result["presets"] = _validate_presets(value["presets"])
        if not result:
            raise _bad("文件中没有 settings 或 presets。")
        return result

    # Legacy preset files are plain name -> settings objects, and names such as
    # "method" or "settings" are legal. Detect this shape before interpreting
    # keys as settings fields.
    if all(isinstance(settings, dict) for settings in value.values()):
        return {"presets": _validate_presets(value)}

    if set(value) & _SETTINGS_FIELDS:
        return {"settings": validate_settings(value)}
    if "settings" in value or "presets" in value:
        raise _bad("带有 settings 或 presets 的文件必须声明 format 和 version。")
    raise _bad("文件格式无法识别。")


def settings_document(settings: dict[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT_NAME, "version": FORMAT_VERSION,
            "settings": validate_settings(settings)}


def presets_document(presets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"format": FORMAT_NAME, "version": FORMAT_VERSION,
            "presets": _validate_presets(presets)}


def read_settings_file(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _bad(f"无法读取 JSON 文件：{exc}") from exc
    return decode_settings_document(value)


def write_settings_file(path: str | Path, document: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
