"""Chinese text analysis methods and bounded result computation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import monotonic
from typing import Any, Callable
import json
import math
import re
import threading

import jieba
import numpy as np
import pandas as pd
from scipy import sparse
from joblib import parallel_backend
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .data_io import Corpus
from .resources import user_data_path

METHODS = {
    "bge": "BGE-small-zh-v1.5（本地 ONNX）",
    "tfidf-word": "词级 TF-IDF",
    "tfidf-char": "字/字符 TF-IDF",
    "lsa": "LSA 潜在语义分析",
    "nmf": "NMF 主题分解",
    "lda": "LDA 主题模型",
    "cooc-ppmi": "窗口共现与 PPMI",
    "word2vec": "外部 word2vec 平均向量",
}
MAX_ANALYSIS_DOCS = 20_000
MAX_TOTAL_TOKENS = 2_000_000
MAX_TOKENS_PER_DOCUMENT = 100_000
MAX_FULL_SIMILARITY_DOCS = 1_500
MAX_FEATURES = 20_000
MAX_COOCCURRENCE_VOCAB = 1_000
SILHOUETTE_SAMPLE_LIMIT = 2_000

_CONSERVATIVE_STOPWORDS = set("的 了 和 与 或 是 在 对 把 被 为 于 及 并 但 而 也 都 很 更 最 还 又 就 才 只 若 如 及其 这个 那个 这些 那些 一个 一种 进行 以及 其中 通过 根据 关于 对于 可以 需要 可能 已经 目前 同时 因为 所以 如果 由于 因此 我们 你们 他们 她们 它们 我 你 他 她 它 该 其 此 该项 相关 本次 本文 本研究 本项目 研究 结果 说明 表明 表示 指出 认为 发现 采用 使用 包括 等 等等 例如 主要 一般 其他 其他的 一个 一些 多个 该 该等 之 中 上 下 内 外 后 前 时 次 年 月 日 方面 情况 问题 内容 数据 方法 进行 方面 时候".split())
_STANDARD_EXTRA_STOPWORDS = set("啊 呢 吧 嘛 哦 嗯 呀 啦 哇 么 时候 这里 那里 怎么 什么 为什么 哪些 哪个 谁 何时 多少 如何 可以 能够 应当 应该 必须 可能 似乎 大约 其中 此外 另外 而且 并且 然而 不过 但是 因而 因此 所以 于是 以便 以免 除非 即使 尽管 虽然 无论 或者 还是 以及 及其 及与 同样 该项 此项 上述 下述 前述 后者 前者 各个 每个 所有 某些 一切 一点 一些 一下 一般 通常 进一步 逐步 继续 仍然 仍旧 已经 正在 将会 将要 曾经 总是 一直 往往 有关 相关 方面 情形 状况 进行 实现 产生 完成 发生 存在 成为 作为 通过 基于 按照 依据 由于 因为 而言 来说 来看 可以说 总体而言 总之 此时 此刻 当前 现在 之前 之后 期间 之间 以上 以下 左右 以内 以后 以前 之内 之外 里面 外面".split())


@dataclass
class AnalysisConfig:
    method: str = "bge"
    segmentation: str = "jieba精确模式"
    stopwords: str = "保守内置"
    custom_stopwords_path: str = ""
    user_dictionary_path: str = ""
    max_features: int = 4_000
    min_frequency: int = 1
    topic_count: int = 8
    cooc_window: int = 5
    cooc_vocab_size: int = 400
    clusters: int = 5
    word2vec_path: str = ""
    word2vec_tfidf_weight: bool = True
    compute_similarity: bool = True
    max_df: float = 1.0
    word_ngram_min: int = 1
    word_ngram_max: int = 1
    char_ngram_min: int = 1
    char_ngram_max: int = 2
    neighbors_k: int = 10
    compute_clusters: bool = True
    nmf_max_iter: int = 300
    lda_max_iter: int = 20
    random_state: int = 42
    min_token_length: int = 1
    remove_numbers: bool = False
    remove_urls: bool = False
    case_mode: str = "方法默认"
    topic_top_words: int = 12
    profile_top_words: int = 10
    representative_count: int = 3
    compute_neighbors: bool = True
    compute_silhouette: bool = True
    tfidf_sublinear_tf: bool = False
    tfidf_smooth_idf: bool = True
    nmf_tol: float = 1e-4
    kmeans_tol: float = 1e-4
    lda_mean_change_tol: float = 1e-3
    lda_perp_tol: float = .1
    lda_evaluate_every: int = -1
    lda_doc_topic_prior: float | None = None
    lda_topic_word_prior: float | None = None
    parallel_enabled: bool = False
    cpu_workers: int = 4
    bge_device: str = "cpu"

    def normalized(self) -> "AnalysisConfig":
        cfg = AnalysisConfig(**asdict(self))
        if cfg.method not in METHODS:
            raise ValueError(f"未知分析方法：{cfg.method}")
        if cfg.segmentation not in {"jieba精确模式", "jieba搜索模式", "字符"}:
            raise ValueError(f"未知分词方法：{cfg.segmentation}")
        if cfg.stopwords not in {"无", "保守内置", "标准内置", "自定义文件"}:
            raise ValueError(f"未知停用词选项：{cfg.stopwords}")
        cfg.max_features = int(np.clip(cfg.max_features, 1, MAX_FEATURES))
        cfg.min_frequency = max(1, int(cfg.min_frequency))
        cfg.topic_count = max(2, min(100, int(cfg.topic_count)))
        cfg.cooc_window = max(2, min(20, int(cfg.cooc_window)))
        cfg.cooc_vocab_size = max(20, min(MAX_COOCCURRENCE_VOCAB, int(cfg.cooc_vocab_size)))
        cfg.clusters = max(2, min(100, int(cfg.clusters)))
        cfg.max_df = float(cfg.max_df)
        if not math.isfinite(cfg.max_df) or not 0 < cfg.max_df <= 1:
            raise ValueError("最大文档比例必须大于 0 且不超过 1；1 表示不过滤高频词。")
        for prefix, limit, label in [("word", 3, "词级"), ("char", 5, "字符级")]:
            lower, upper = getattr(cfg, f"{prefix}_ngram_min"), getattr(cfg, f"{prefix}_ngram_max")
            if not isinstance(lower, int) or not isinstance(upper, int) or not 1 <= lower <= upper <= limit:
                raise ValueError(f"{label} n-gram 范围须满足 1 ≤ 最小值 ≤ 最大值 ≤ {limit}。")
        for name, low, high, label in [("neighbors_k", 1, 100, "近邻 Top-K"),
                                       ("nmf_max_iter", 1, 1000, "NMF 最大迭代次数"),
                                       ("lda_max_iter", 1, 1000, "LDA 最大迭代次数"),
                                       ("min_token_length", 1, 20, "最小词长"),
                                       ("topic_top_words", 1, 100, "主题词数量"),
                                       ("profile_top_words", 1, 100, "画像关键词数量"),
                                       ("representative_count", 1, 50, "代表文本数量"),
                                       ("random_state", 0, 2**32 - 1, "随机种子")]:
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(f"{label}须为 {low}–{high} 范围内的整数。")
        for name in ["compute_clusters", "compute_neighbors", "compute_silhouette", "remove_numbers",
                     "remove_urls", "tfidf_sublinear_tf", "tfidf_smooth_idf", "parallel_enabled"]:
            if not isinstance(getattr(cfg, name), bool):
                raise ValueError(f"{name} 须为布尔值。")
        if (not isinstance(cfg.cpu_workers, int) or isinstance(cfg.cpu_workers, bool)
                or not 1 <= cfg.cpu_workers <= 32):
            raise ValueError("CPU 工作线程数须为 1–32 的整数。")
        if cfg.bge_device not in {"cpu", "auto", "cuda", "directml"}:
            raise ValueError("BGE 设备须选择 cpu、auto、cuda 或 directml。")
        if cfg.case_mode not in {"方法默认", "统一小写", "保持原样"}:
            raise ValueError("英文大小写须选择方法默认、统一小写或保持原样。")
        for name, label in [("nmf_tol", "NMF 收敛阈值"), ("kmeans_tol", "KMeans 收敛阈值"),
                            ("lda_mean_change_tol", "LDA 文档更新阈值"), ("lda_perp_tol", "LDA 困惑度阈值"),
                            ("lda_doc_topic_prior", "LDA 文档—主题先验"), ("lda_topic_word_prior", "LDA 主题—词先验")]:
            raw_value = getattr(cfg, name)
            if raw_value is None and name.endswith("prior"):
                continue
            if raw_value is None or isinstance(raw_value, bool):
                raise ValueError(f"{label}须为有限正数。")
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label}须为有限正数。")
            setattr(cfg, name, value)
        if (not isinstance(cfg.lda_evaluate_every, int) or isinstance(cfg.lda_evaluate_every, bool)
                or cfg.lda_evaluate_every not in {-1, *range(1, 101)}):
            raise ValueError("LDA 评估间隔须为 -1（关闭）或 1–100 的整数。")
        return cfg


@dataclass
class AnalysisResult:
    doc_ids: list[str]
    method: str
    method_label: str
    config: dict[str, Any]
    word_count_matrix: sparse.csr_matrix
    word_tfidf_matrix: sparse.csr_matrix
    word_vocabulary: list[str]
    doc_matrix: Any
    doc_feature_labels: list[str]
    doc_matrix_name: str
    document_metrics: pd.DataFrame
    word_frequencies: pd.DataFrame
    nearest_neighbors: pd.DataFrame
    similarity_matrix: np.ndarray | None
    topic_document_matrix: np.ndarray | None
    topic_weight_matrix: np.ndarray | None
    topic_weight_kind: str
    topic_terms: pd.DataFrame
    cooccurrence_counts: sparse.csr_matrix | None
    ppmi_matrix: sparse.csr_matrix | None
    cooccurrence_vocabulary: list[str]
    cluster_labels: pd.DataFrame
    clustering_summary: dict[str, Any]
    warnings: list[str]
    zero_document_ids: list[str]


def stopword_set(config: AnalysisConfig) -> set[str]:
    if config.stopwords == "无":
        return set()
    if config.stopwords == "保守内置":
        return set(_CONSERVATIVE_STOPWORDS)
    if config.stopwords == "标准内置":
        return set(_CONSERVATIVE_STOPWORDS | _STANDARD_EXTRA_STOPWORDS)
    if not config.custom_stopwords_path:
        raise ValueError("已选择自定义停用词，但还没有选择词表文件。")
    path = Path(config.custom_stopwords_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到停用词文件：{path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return {line.strip() for line in text.splitlines() if line.strip()}


def _check_cancel(event: Any) -> None:
    if event is not None and event.is_set():
        raise InterruptedError("分析已取消")


def _notify(progress: Callable[[float, str], None] | None, value: float, message: str) -> None:
    if progress:
        progress(value, message)


@contextmanager
def _stage(event_callback: Callable[[dict[str, Any]], None] | None, stage_id: str, label: str,
           method: str, *, work_total: float | None = None, unit: str = "工作量",
           completion_total: Callable[[], float | None] | None = None,
           completion_done: Callable[[], float | None] | None = None,
           completion_success: Callable[[], bool] | None = None):
    """Emit explicit boundaries around a real blocking call."""
    started = monotonic()
    if event_callback:
        event_callback({"type": "stage-start", "stage_id": stage_id, "label": label, "method": method,
                        "initial_seconds": 1.0, "work_total": work_total, "unit": unit})
    try:
        yield
    except BaseException:
        if event_callback:
            event_callback({"type": "stage-failed", "stage_id": stage_id, "method": method,
                            "elapsed_seconds": max(0.0, monotonic() - started)})
        raise
    else:
        final_total = completion_total() if completion_total else work_total
        final_done = completion_done() if completion_done else final_total
        if event_callback:
            successful = completion_success() if completion_success else True
            event_callback({"type": "stage-complete" if successful else "stage-failed",
                            "stage_id": stage_id, "method": method,
                            "work_done": final_done, "work_total": final_total, "unit": unit,
                            "elapsed_seconds": max(0.0, monotonic() - started)})


def _stage_progress(event_callback: Callable[[dict[str, Any]], None] | None, stage_id: str,
                    method: str, work_done: float, work_total: float | None, unit: str,
                    **details: Any) -> None:
    if event_callback:
        event_callback({"type": "stage-progress", "stage_id": stage_id, "method": method,
                        "work_done": work_done, "work_total": work_total, "unit": unit, **details})


_URL_PATTERN = re.compile(r"(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)%?")


def _prepare_text(text: str, cfg: AnalysisConfig) -> str:
    if cfg.remove_urls:
        text = _URL_PATTERN.sub(" ", text)
        if not text.strip():
            text = ""
    return text.lower() if cfg.case_mode == "统一小写" else text


def _tokens_for(text: str, cfg: AnalysisConfig, stopwords: set[str], tokenizer: jieba.Tokenizer) -> list[str]:
    if cfg.segmentation == "字符":
        tokens = iter(text)
    elif cfg.segmentation == "jieba搜索模式":
        tokens = tokenizer.cut_for_search(text)
    else:
        tokens = tokenizer.cut(text, cut_all=False)
    result = []
    for token in tokens:
        cleaned = token.strip()
        if not cleaned or cleaned in stopwords or re.fullmatch(r"[\W_]+", cleaned, flags=re.UNICODE):
            continue
        if len(cleaned) < cfg.min_token_length or (cfg.remove_numbers and _NUMBER_PATTERN.fullmatch(cleaned)):
            continue
        result.append(cleaned)
        if len(result) > MAX_TOKENS_PER_DOCUMENT:
            raise ValueError(f"单篇文档分词数超过 {MAX_TOKENS_PER_DOCUMENT:,} 的内存保护上限。请先切分过长文本。")
    return result


def _configure_user_dictionary(path: str) -> jieba.Tokenizer:
    """Make a per-run jieba tokenizer so custom words cannot leak to later runs."""
    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = str(user_data_path("cache"))
    if path:
        user_path = Path(path)
        if not user_path.is_file():
            raise FileNotFoundError(f"找不到自定义词典：{user_path}")
        tokenizer.load_userdict(str(user_path))
    return tokenizer


def _fit_word_matrices(token_docs: list[list[str]], max_features: int, min_frequency: int, *,
                       max_df: float = 1.0, ngram_range: tuple[int, int] = (1, 1),
                       sublinear_tf: bool = False, smooth_idf: bool = True) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str]]:
    if token_docs and min_frequency > len(token_docs):
        raise ValueError(f"最小文档频次 {min_frequency} 大于文档总数 {len(token_docs)}。")
    if token_docs and max_df * len(token_docs) < min_frequency:
        raise ValueError("最大文档比例对应的文档数小于最小文档频次，请调大比例或调小最小频次。")
    text_docs = [" ".join(tokens) for tokens in token_docs]
    vectorizer = CountVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False,
                                 max_features=max_features, min_df=min_frequency, max_df=max_df,
                                 ngram_range=ngram_range, dtype=np.int32)
    try:
        counts = vectorizer.fit_transform(text_docs).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower() or "no terms remain" in str(exc).lower():
            return sparse.csr_matrix((len(text_docs), 0), dtype=np.int32), sparse.csr_matrix((len(text_docs), 0), dtype=np.float32), []
        raise
    vocab = [str(x) for x in vectorizer.get_feature_names_out()]
    tfidf = TfidfTransformer(sublinear_tf=sublinear_tf, smooth_idf=smooth_idf).fit_transform(counts).tocsr().astype(np.float32)
    return counts, tfidf, vocab


def _bge_model_dir() -> Path:
    from .resources import resource_path
    return resource_path("assets/bge-small-zh-v1.5")


_BGE_PROVIDER_BY_DEVICE = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}


def _create_bge_session(ort: Any, model_path: str | Path, cfg: AnalysisConfig) -> tuple[Any, dict[str, Any]]:
    """Create a BGE session and verify the requested provider is active.

    ONNX Runtime can keep a CPU provider in the session as a fallback for graph
    nodes an accelerator cannot execute. The returned provider list therefore
    describes session priority, not a guarantee that every graph operation is
    running on the accelerator.
    """
    available = list(ort.get_available_providers())
    requested = cfg.bge_device
    cpu_provider = "CPUExecutionProvider"
    fallback_reasons: list[str] = []

    if requested == "cpu":
        candidates = [cpu_provider]
    elif requested == "auto":
        candidates = [provider for provider in (
            _BGE_PROVIDER_BY_DEVICE["cuda"], _BGE_PROVIDER_BY_DEVICE["directml"]
        ) if provider in available]
        if not candidates:
            fallback_reasons.append("当前 ONNX Runtime 未提供 CUDA 或 DirectML execution provider")
        candidates.append(cpu_provider)
    else:
        provider = _BGE_PROVIDER_BY_DEVICE[requested]
        if provider not in available:
            raise RuntimeError(
                f"已选择 BGE {requested.upper()}，但当前 ONNX Runtime 未提供 {provider}。"
                "请安装包含该 execution provider 的运行时并确认设备驱动可用。"
            )
        candidates = [provider]

    for provider in candidates:
        if provider not in available:
            reason = f"{provider} 不在当前 ONNX Runtime 的可用 provider 列表中"
            if requested == "auto":
                fallback_reasons.append(reason)
                continue
            raise RuntimeError(reason)

        session_options = ort.SessionOptions()
        if cfg.parallel_enabled:
            session_options.intra_op_num_threads = cfg.cpu_workers
        if provider == _BGE_PROVIDER_BY_DEVICE["directml"]:
            # DirectML requires sequential graph execution and disabled memory patterns.
            session_options.enable_mem_pattern = False
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        requested_providers = [provider]
        if provider != cpu_provider and cpu_provider in available:
            requested_providers.append(cpu_provider)
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=session_options, providers=requested_providers
            )
            actual_providers = list(session.get_providers())
        except Exception as exc:
            if requested != "auto":
                label = requested.upper()
                raise RuntimeError(f"无法初始化明确选择的 BGE {label} execution provider：{exc}") from exc
            fallback_reasons.append(f"{provider} 初始化失败（{type(exc).__name__}: {exc}）")
            continue

        # Provider order is significant: the first entry has highest priority.
        # Explicit accelerator requests must not silently become CPU sessions.
        if not actual_providers or actual_providers[0] != provider:
            reason = f"请求 {provider}，会话实际 provider 顺序为 {actual_providers or '空'}"
            if requested == "auto":
                fallback_reasons.append(reason)
                continue
            raise RuntimeError(
                f"已选择 BGE {requested.upper()}，但 ONNX Runtime 未将 {provider} 设为活动 provider；{reason}。"
                "为避免静默回退，分析已停止。"
            )

        return session, {
            "requested_device": requested,
            "selected_provider": provider,
            "active_providers": actual_providers,
            "fallback_reason": "; ".join(fallback_reasons),
        }

    # Auto mode always tries CPU last. Reaching this branch means even CPU
    # session creation failed; preserve the concrete initialization detail.
    reason = "; ".join(fallback_reasons) or "没有可用 execution provider"
    raise RuntimeError(f"BGE 自动设备选择失败：{reason}。")


def _encode_bge(texts: list[str], progress: Callable[[float, str], None] | None, cancel_event: Any,
                event_callback: Callable[[dict[str, Any]], None] | None = None, *,
                config: AnalysisConfig | None = None,
                device_info: dict[str, Any] | None = None) -> tuple[np.ndarray, int]:
    with _stage(event_callback, "bge-model-load", "加载本地 BGE 模型", "bge"):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # run_analysis passes its already-normalized config; keep the no-config
        # private-call path compatible without repeating boundary validation.
        cfg = config if config is not None else AnalysisConfig().normalized()
        model_dir = _bge_model_dir()
        model_path = model_dir / "onnx" / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError("本地 BGE 模型资源缺失。请重新安装完整程序；分析过程不会联网下载模型。")
        session, selected_device = _create_bge_session(ort, model_path, cfg)
        if device_info is not None:
            device_info.update(selected_device)
        if event_callback:
            provider_summary = (
                f"requested={selected_device['requested_device']}; "
                f"selected={selected_device['selected_provider']}; "
                f"active={','.join(selected_device['active_providers'])}"
            )
            event_callback({
                "type": "stage-progress", "stage_id": "bge-model-load", "method": "bge",
                "work_done": 1, "work_total": 1, "unit": "ONNX 会话",
                "requested_device": selected_device["requested_device"],
                "selected_provider": selected_device["selected_provider"],
                "active_providers": selected_device["active_providers"],
                "details": provider_summary,
            })
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_truncation()
        input_meta = {item.name: item for item in session.get_inputs()}
        output_names = [item.name for item in session.get_outputs()]
        empty_specials = tokenizer.encode("", add_special_tokens=True).ids
        body_empty = tokenizer.encode("", add_special_tokens=False).ids
        special_ids = [i for i in empty_specials if i not in body_empty]
        # BERT-style classifiers use [CLS] + body + [SEP]; derive the sequence boundary IDs locally.
        if len(special_ids) < 2:
            special_ids = empty_specials[:2]
        prefix, suffix = special_ids[:1], special_ids[-1:]
        if not prefix or not suffix:
            raise RuntimeError("无法从本地分词器读取 CLS/SEP 特殊标记。")

    char_lengths = [len(text) for text in texts]
    suffix_chars = [0] * (len(texts) + 1)
    for index in range(len(texts) - 1, -1, -1):
        suffix_chars[index] = suffix_chars[index + 1] + char_lengths[index]
    estimated_chunks = max(len(texts), sum(max(1, math.ceil(size / 4 / 510)) for size in char_lengths))
    chunk_count = 0
    tokens_encoded = 0
    tokens_discovered = 0
    chars_discovered = 0
    with _stage(event_callback, "bge-encode", "BGE 分词与分块编码", "bge", work_total=estimated_chunks,
                unit="估算分块", completion_total=lambda: chunk_count):
        vectors: list[np.ndarray] = []
        long_documents = 0
        for i, text in enumerate(texts):
            _check_cancel(cancel_event)
            body_ids = tokenizer.encode(text, add_special_tokens=False).ids
            tokens_discovered += len(body_ids)
            chars_discovered += char_lengths[i]
            if not body_ids:
                vectors.append(np.zeros(512, dtype=np.float32))
            else:
                chunks = [body_ids[start:start + 510] for start in range(0, len(body_ids), 510)]
                if len(chunks) > 1:
                    long_documents += 1
                pooled: np.ndarray | None = None
                weight_sum = 0.0
                for chunk_index, chunk in enumerate(chunks, 1):
                    _check_cancel(cancel_event)
                    ids = prefix + chunk + suffix
                    arr_ids = np.asarray([ids], dtype=np.int64)
                    feed: dict[str, np.ndarray] = {}
                    for name, meta in input_meta.items():
                        dtype = np.int32 if "int32" in meta.type else np.int64
                        if name == "input_ids":
                            values = arr_ids.astype(dtype, copy=False)
                        elif name == "attention_mask":
                            values = np.ones_like(arr_ids, dtype=dtype)
                        elif name == "token_type_ids":
                            values = np.zeros_like(arr_ids, dtype=dtype)
                        else:
                            raise RuntimeError(f"模型需要不支持的输入：{name}")
                        feed[name] = values
                    raw_outputs = session.run(output_names, feed)
                    output = None
                    for candidate in raw_outputs:
                        if getattr(candidate, "ndim", 0) == 3:
                            output = candidate
                            break
                    if output is None:
                        raise RuntimeError("ONNX 输出中没有 token 隐层，无法按 CLS 方法生成句向量。")
                    cls_vector = np.asarray(output[0, 0, :], dtype=np.float32)
                    cls_norm = float(np.linalg.norm(cls_vector))
                    if cls_norm > 0:
                        cls_vector = cls_vector / cls_norm
                    weight = float(len(chunk))
                    pooled = cls_vector * weight if pooled is None else pooled + cls_vector * weight
                    weight_sum += weight
                    chunk_count += 1
                    tokens_encoded += len(chunk)
                    remaining_in_document = len(chunks) - chunk_index
                    remaining_documents = len(texts) - i - 1
                    token_chars_ratio = tokens_discovered / max(chars_discovered, 1)
                    estimated_future_chunks = max(remaining_documents,
                                                  int(math.ceil(suffix_chars[i + 1] * token_chars_ratio / 510)))
                    dynamic_total = max(chunk_count,
                                       chunk_count + remaining_in_document + estimated_future_chunks)
                    if chunk_count <= 2 or chunk_count % 8 == 0 or chunk_index == len(chunks):
                        doc_complete = chunk_index == len(chunks)
                        _stage_progress(event_callback, "bge-encode", "bge", chunk_count, dynamic_total,
                                        "已编码chunk / 总chunk估计", documents_done=i + 1 if doc_complete else i,
                                        documents_total=len(texts), token_count_done=tokens_encoded,
                                        document_index=i + 1, current_document_complete=doc_complete)
                vector = (pooled / max(weight_sum, 1.0)).astype(np.float32)
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector /= norm
                vectors.append(vector)
            if (i + 1) % 4 == 0 or i + 1 == len(texts) or not body_ids:
                token_chars_ratio = tokens_discovered / max(chars_discovered, 1)
                future_chunks = max(len(texts) - i - 1,
                                    int(math.ceil(suffix_chars[i + 1] * token_chars_ratio / 510)))
                dynamic_total = max(chunk_count, chunk_count + future_chunks)
                _stage_progress(event_callback, "bge-encode", "bge", chunk_count, dynamic_total,
                                "已编码chunk / 总chunk估计", documents_done=i + 1,
                                documents_total=len(texts), token_count_done=tokens_encoded)
                _notify(progress, 0.18 + 0.42 * (i + 1) / max(len(texts), 1), f"BGE 本地编码：{i + 1}/{len(texts)}")
        return np.vstack(vectors).astype(np.float32, copy=False), long_documents


def _parse_word2vec(
    path: str,
    needed_words: set[str],
    *,
    progress: Callable[[float, str], None] | None = None,
    cancel_event: Any = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    max_dimension: int = 1_024,
) -> tuple[dict[str, np.ndarray], int, int, int, int]:
    if not path:
        raise ValueError("word2vec 方法需要选择 UTF-8/本地编码的文本向量文件。")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到 word2vec 文件：{file_path}")
    vectors: dict[str, np.ndarray] = {}
    dimension: int | None = None
    line_no = 0
    invalid_values = 0
    file_size = max(file_path.stat().st_size, 1)
    encoding = "utf-8"
    retried_encoding = False
    position = 0
    with file_path.open("rb") as handle:
        while True:
            _check_cancel(cancel_event)
            raw = handle.readline()
            position = handle.tell()
            if not raw:
                break
            line_no += 1
            if line_no <= 2:
                _stage_progress(event_callback, "word2vec-scan", "word2vec", position, file_size,
                                "字节", bytes_done=position, bytes_total=file_size, lines_done=line_no)
            try:
                line = raw.decode(encoding).lstrip("\ufeff").strip()
            except UnicodeDecodeError as exc:
                if encoding == "utf-8" and not retried_encoding:
                    encoding = "gb18030"
                    retried_encoding = True
                    vectors.clear()
                    dimension = None
                    invalid_values = 0
                    line_no = 0
                    handle.seek(0)
                    continue
                raise ValueError("word2vec 文本向量需使用 UTF-8 或 GB18030 编码；请另存为 UTF-8 后重试。") from exc
            if not line:
                continue
            pieces = line.split()
            if line_no == 1 and len(pieces) == 2 and all(part.isdigit() for part in pieces):
                dimension = int(pieces[1])
                if dimension < 1 or dimension > max_dimension:
                    raise ValueError(f"word2vec 向量维数须在 1–{max_dimension} 之间。")
                continue
            if len(pieces) < 3:
                continue
            word, raw_values = pieces[0], pieces[1:]
            if dimension is None:
                dimension = len(raw_values)
                if dimension > max_dimension:
                    raise ValueError(f"word2vec 向量维数超过 {max_dimension} 的资源保护上限。")
            if len(raw_values) != dimension or word not in needed_words:
                if line_no % 2_000 == 0:
                    _stage_progress(event_callback, "word2vec-scan", "word2vec", position, file_size,
                                    "字节", bytes_done=position, bytes_total=file_size, lines_done=line_no)
                    _notify(progress, 0.20 + 0.25 * min(position / file_size, 1.0),
                            f"扫描外部向量：{position:,}/{file_size:,} 字节")
                continue
            try:
                vector = np.asarray(raw_values, dtype=np.float32)
            except ValueError:
                continue
            if not np.all(np.isfinite(vector)):
                invalid_values += 1
                continue
            vectors[word] = vector
            if line_no % 2_000 == 0:
                position = handle.tell()
                _stage_progress(event_callback, "word2vec-scan", "word2vec", position, file_size,
                                "字节", bytes_done=position, bytes_total=file_size, lines_done=line_no)
                _notify(progress, 0.20 + 0.25 * min(position / file_size, 1.0),
                        f"扫描外部向量：{position:,}/{file_size:,} 字节")
            if len(vectors) == len(needed_words):
                break
    _stage_progress(event_callback, "word2vec-scan", "word2vec", position,
                    file_size, "字节", bytes_total=file_size, lines_done=line_no)
    return vectors, int(dimension or 0), line_no, invalid_values, position


def _cooccurrence(tokens_by_doc: list[list[str]], vocab_limit: int, window: int, *,
                  event_callback: Callable[[dict[str, Any]], None] | None = None,
                  cancel_event: Any = None) -> tuple[sparse.csr_matrix, sparse.csr_matrix, list[str], sparse.csr_matrix]:
    from collections import Counter
    counts = Counter(token for doc in tokens_by_doc for token in doc)
    vocab = [token for token, _ in counts.most_common(vocab_limit)]
    index = {token: i for i, token in enumerate(vocab)}
    pair_counts: dict[tuple[int, int], int] = {}
    per_doc_terms: list[list[str]] = []
    total_positions = max(1, sum(len(tokens) for tokens in tokens_by_doc))
    processed_positions = 0
    report_every = max(1, len(tokens_by_doc) // 40)
    for doc_index, tokens in enumerate(tokens_by_doc, 1):
        _check_cancel(cancel_event)
        per_doc_terms.append(tokens)
        # Window is a token-position radius. Out-of-vocabulary items keep their
        # positions and are ignored; removing them cannot pull distant words closer.
        for i, term in enumerate(tokens):
            left = index.get(term)
            if left is None:
                continue
            for other in tokens[i + 1:i + window + 1]:
                right = index.get(other)
                if right is None or left == right:
                    continue
                a, b = (left, right) if left < right else (right, left)
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
                pair_counts[(b, a)] = pair_counts.get((b, a), 0) + 1
        processed_positions += len(tokens)
        if doc_index <= 2 or doc_index % report_every == 0 or doc_index == len(tokens_by_doc):
            _stage_progress(event_callback, "cooc-ppmi", "cooc-ppmi", processed_positions,
                            total_positions, "token 位置", documents_done=doc_index,
                            documents_total=len(tokens_by_doc))
    if pair_counts:
        rows, cols, vals = zip(*((r, c, v) for (r, c), v in pair_counts.items()))
        cooc = sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(vocab), len(vocab)))
    else:
        cooc = sparse.csr_matrix((len(vocab), len(vocab)), dtype=np.float32)
    total = float(cooc.sum())
    if total > 0:
        row_sums = np.asarray(cooc.sum(axis=1)).ravel()
        col_sums = np.asarray(cooc.sum(axis=0)).ravel()
        coo = cooc.tocoo()
        denom = row_sums[coo.row] * col_sums[coo.col]
        values = np.log(np.maximum(coo.data * total / np.maximum(denom, 1e-20), 1e-20))
        keep = values > 0
        ppmi = sparse.csr_matrix((values[keep].astype(np.float32), (coo.row[keep], coo.col[keep])), shape=cooc.shape)
    else:
        ppmi = sparse.csr_matrix(cooc.shape, dtype=np.float32)
    if vocab:
        docs = [" ".join(x) for x in per_doc_terms]
        # Fix column order to match the co-occurrence vocabulary.
        doc_counts = sparse.csr_matrix((len(docs), len(vocab)), dtype=np.float32)
        vlookup = {term: i for i, term in enumerate(vocab)}
        rr, cc, vv = [], [], []
        for r, terms in enumerate(per_doc_terms):
            local: dict[int, int] = {}
            for term in terms:
                k = vlookup.get(term)
                if k is None:
                    continue
                local[k] = local.get(k, 0) + 1
            for k, val in local.items():
                rr.append(r); cc.append(k); vv.append(val)
        if rr:
            doc_counts = sparse.csr_matrix((vv, (rr, cc)), shape=(len(docs), len(vocab)), dtype=np.float32)
        doc_repr = (doc_counts @ ppmi).tocsr()
    else:
        doc_repr = sparse.csr_matrix((len(tokens_by_doc), 0), dtype=np.float32)
    return cooc, ppmi, vocab, doc_repr


def _topics(components: np.ndarray, feature_names: list[str], limit: int = 12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic_idx, weights in enumerate(components):
        if not len(feature_names):
            continue
        indices = np.argsort(weights)[::-1][:min(limit, len(feature_names))]
        for rank, idx in enumerate(indices, 1):
            rows.append({"主题": f"主题 {topic_idx + 1}", "排名": rank, "词语": feature_names[int(idx)], "权重": float(weights[int(idx)])})
    return pd.DataFrame(rows, columns=["主题", "排名", "词语", "权重"])


def _nearest_neighbors(matrix: Any, ids: list[str], zero_mask: np.ndarray, k: int = 10, *,
                       cancel_event: Any = None,
                       event_callback: Callable[[dict[str, Any]], None] | None = None) -> pd.DataFrame:
    normalized = normalize(matrix, norm="l2", copy=True)
    rows: list[dict[str, Any]] = []
    n = len(ids)
    if n < 2:
        return pd.DataFrame(columns=["文档ID", "排名", "近邻ID", "余弦相似度"])
    block_size = max(1, min(64, 8_000_000 // max(n, 1)))
    report_every = max(block_size, n // 20)
    last_report = 0
    for start in range(0, n, block_size):
        _check_cancel(cancel_event)
        stop = min(n, start + block_size)
        scores = (normalized[start:stop] @ normalized.T)
        scores = scores.toarray() if sparse.issparse(scores) else np.asarray(scores)
        for local, row in enumerate(scores):
            i = start + local
            if zero_mask[i]:
                continue
            row[i] = -np.inf
            row[zero_mask] = -np.inf
            take = min(k, n - int(zero_mask.sum()) - 1)
            if take <= 0:
                continue
            indices = np.argpartition(row, -take)[-take:]
            indices = indices[np.argsort(row[indices])[::-1]]
            for rank, j in enumerate(indices, 1):
                if not np.isfinite(row[j]):
                    continue
                rows.append({"文档ID": ids[i], "排名": rank, "近邻ID": ids[int(j)], "余弦相似度": float(row[j])})
        if stop >= n or stop - last_report >= report_every:
            _stage_progress(event_callback, "neighbors", "", stop * n, n * n, "文档对",
                            rows_done=stop, rows_total=n)
            last_report = stop
    return pd.DataFrame(rows, columns=["文档ID", "排名", "近邻ID", "余弦相似度"])


def run_analysis(
    corpus: Corpus,
    config: AnalysisConfig | None = None,
    *,
    cancel_event: Any = None,
    progress: Callable[[float, str], None] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AnalysisResult:
    cfg = (config or AnalysisConfig()).normalized()
    thread_context = threadpool_limits(limits=cfg.cpu_workers) if cfg.parallel_enabled else nullcontext()
    with thread_context:
        return _run_analysis_normalized(
            corpus, cfg, cancel_event=cancel_event, progress=progress, event_callback=event_callback
        )


def _run_analysis_normalized(
    corpus: Corpus,
    config: AnalysisConfig,
    *,
    cancel_event: Any = None,
    progress: Callable[[float, str], None] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AnalysisResult:
    cfg = config
    docs = corpus.documents
    n_docs = len(docs)
    if n_docs > MAX_ANALYSIS_DOCS:
        raise ValueError(f"当前分析最多处理 {MAX_ANALYSIS_DOCS:,} 篇文档；本次有 {n_docs:,} 篇。请先筛选或分批处理。")
    _check_cancel(cancel_event)
    raw_texts = [d.text for d in docs]
    analysis_texts = [_prepare_text(text, cfg) for text in raw_texts]
    token_docs: list[list[str]] = []
    total_tokens = 0
    with _stage(event_callback, "tokenize", "中文分词与停词处理", cfg.method, work_total=n_docs, unit="文档"):
        tokenizer = _configure_user_dictionary(cfg.user_dictionary_path)
        stops = stopword_set(cfg)
        if cfg.case_mode == "统一小写":
            stops = {word.lower() for word in stops}
        for index, text in enumerate(analysis_texts, 1):
            _check_cancel(cancel_event)
            tokens = _tokens_for(text, cfg, stops, tokenizer)
            total_tokens += len(tokens)
            if total_tokens > MAX_TOTAL_TOKENS:
                raise ValueError(f"累计分词数超过 {MAX_TOTAL_TOKENS:,} 的内存保护上限；请减少文档数量或分批分析。")
            token_docs.append(tokens)
            if index <= 2 or index % max(1, n_docs // 40) == 0 or index == n_docs:
                _stage_progress(event_callback, "tokenize", cfg.method, index, n_docs, "文档",
                                documents_done=index, documents_total=n_docs, tokens_done=total_tokens)
                _notify(progress, 0.08 * index / max(n_docs, 1), f"中文分词：{index:,}/{n_docs:,}")
    _notify(progress, 0.08, "中文分词与停词处理完成")
    _check_cancel(cancel_event)

    with _stage(event_callback, "term-matrix", "构建词项矩阵", cfg.method, work_total=n_docs, unit="文档"):
        word_ngrams = (cfg.word_ngram_min, cfg.word_ngram_max) if cfg.method in {"tfidf-word", "lsa", "nmf", "lda"} else (1, 1)
        counts, word_tfidf, word_vocab = _fit_word_matrices(
            token_docs, cfg.max_features, cfg.min_frequency, max_df=cfg.max_df, ngram_range=word_ngrams,
            sublinear_tf=cfg.tfidf_sublinear_tf, smooth_idf=cfg.tfidf_smooth_idf)
        if not word_vocab and cfg.method in {"tfidf-word", "lsa", "nmf", "lda", "word2vec"}:
            raise ValueError("筛选后没有可用词项，请降低最小文档频次、提高最大文档比例、缩小 n-gram 范围或调整停词。")
        word_freq_values = np.asarray(counts.sum(axis=0)).ravel() if counts.shape[1] else np.zeros(0)
        word_freq = pd.DataFrame({"词语": word_vocab, "词频": word_freq_values.astype(np.int64)}).sort_values("词频", ascending=False, kind="stable").reset_index(drop=True)
    _notify(progress, 0.18, f"词项矩阵完成：{counts.shape[0]} × {counts.shape[1]}")

    raw_char_tfidf = None
    cooc_counts = ppmi = None
    cooc_vocab: list[str] = []
    topic_doc: np.ndarray | None = None
    topic_weight_matrix: np.ndarray | None = None
    topic_weight_kind = "未生成"
    topic_terms = pd.DataFrame(columns=["主题", "排名", "词语", "权重"])
    warnings: list[str] = []
    doc_features: list[str] = []
    matrix_name = ""

    if cfg.method == "bge":
        device_info: dict[str, Any] = {}
        doc_matrix, long_document_count = _encode_bge(
            analysis_texts, progress, cancel_event, event_callback, config=cfg, device_info=device_info
        )
        selected_provider = device_info.get("selected_provider")
        if selected_provider in {_BGE_PROVIDER_BY_DEVICE["cuda"], _BGE_PROVIDER_BY_DEVICE["directml"]}:
            fallback_note = (
                f"ONNX Runtime 会话 provider 顺序为 {' → '.join(device_info.get('active_providers', []))}。"
                "不保证模型中的每个算子都由 GPU 执行，ONNX Runtime 仍可能将不支持的算子分配到 CPU。"
            )
            warnings.append(f"BGE GPU provider 已启用（{selected_provider}）；{fallback_note}")
        elif cfg.bge_device == "auto":
            reason = device_info.get("fallback_reason") or "未激活 GPU provider"
            warnings.append(
                f"BGE 设备设为自动选择，但本次使用 CPUExecutionProvider（{reason}）。"
            )
        doc_features = [f"维度{i + 1}" for i in range(doc_matrix.shape[1])]
        matrix_name = "BGE CLS + L2 向量"
        if long_document_count:
            warnings.append(f"{long_document_count} 篇超长文本按真实 BERT token ID 数量分块（每块最多 510 个正文 token）；每块 CLS 先 L2 归一化，再按正文 token 数加权平均并再次 L2 归一化。这是长文本处理近似。")
    elif cfg.method == "tfidf-word":
        doc_matrix, doc_features, matrix_name = word_tfidf, word_vocab, "词级 TF-IDF 表示"
    elif cfg.method == "tfidf-char":
        char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max), max_features=cfg.max_features,
                                          min_df=cfg.min_frequency, max_df=cfg.max_df, dtype=np.float32,
                                          lowercase=cfg.case_mode != "保持原样", sublinear_tf=cfg.tfidf_sublinear_tf,
                                          smooth_idf=cfg.tfidf_smooth_idf)
        with _stage(event_callback, "char-fit", "拟合字符 TF-IDF", cfg.method, work_total=n_docs, unit="文档"):
            try:
                raw_char_tfidf = char_vectorizer.fit_transform(analysis_texts).tocsr().astype(np.float32)
                doc_matrix, doc_features, matrix_name = raw_char_tfidf, [str(x) for x in char_vectorizer.get_feature_names_out()], "字/字符 n-gram TF-IDF 表示"
            except ValueError as exc:
                if "empty vocabulary" in str(exc).lower() or "no terms remain" in str(exc).lower():
                    raise ValueError("字符筛选后没有可用特征，请调整文档频次、最大文档比例或字符 n-gram 范围。") from exc
                else:
                    raise
    elif cfg.method == "lsa":
        if n_docs < 2 or counts.shape[1] < 2:
            raise ValueError("LSA 至少需要 2 篇文档和 2 个词项；当前数据无法形成有意义的潜在维度。")
        n_components = min(cfg.topic_count, n_docs - 1, counts.shape[1] - 1)
        model = TruncatedSVD(n_components=n_components, random_state=cfg.random_state)
        with _stage(event_callback, "lsa-fit", "拟合 LSA 降维", cfg.method, work_total=1, unit="拟合"):
            doc_matrix = model.fit_transform(word_tfidf).astype(np.float32)
        topic_doc = doc_matrix
        doc_features = [f"LSA维度{i + 1}" for i in range(n_components)]
        matrix_name = f"LSA 降维表示（{n_components} 维）"
        topic_weight_matrix = np.asarray(model.components_, dtype=np.float32)
        topic_weight_kind = "LSA 词项载荷（原始分解权重，可正可负）"
        topic_terms = _topics(topic_weight_matrix, word_vocab, limit=cfg.topic_top_words)
    elif cfg.method in {"nmf", "lda"}:
        if n_docs < 2 or counts.shape[1] < 2:
            raise ValueError(f"{METHODS[cfg.method]} 至少需要 2 篇文档和 2 个词项。")
        n_components = min(cfg.topic_count, n_docs, counts.shape[1])
        if cfg.method == "nmf":
            model = NMF(n_components=n_components, init="nndsvda", random_state=cfg.random_state,
                        max_iter=cfg.nmf_max_iter, tol=cfg.nmf_tol)
            with _stage(event_callback, "nmf-fit", "拟合 NMF 主题模型", cfg.method, work_total=1, unit="拟合"):
                topic_doc = model.fit_transform(word_tfidf).astype(np.float32)
            components = model.components_
            topic_weight_kind = "NMF 非负主题词权重（原始分解权重）"
            matrix_name = f"NMF 文档主题权重（{n_components} 个主题）"
        else:
            model = LatentDirichletAllocation(n_components=n_components, learning_method="batch", random_state=cfg.random_state,
                        max_iter=cfg.lda_max_iter, mean_change_tol=cfg.lda_mean_change_tol, perp_tol=cfg.lda_perp_tol,
                        evaluate_every=cfg.lda_evaluate_every, doc_topic_prior=cfg.lda_doc_topic_prior,
                        topic_word_prior=cfg.lda_topic_word_prior,
                        n_jobs=cfg.cpu_workers if cfg.parallel_enabled else None)
            with _stage(event_callback, "lda-fit", "拟合 LDA 主题模型", cfg.method, work_total=1, unit="拟合"):
                if cfg.parallel_enabled and cfg.cpu_workers > 1:
                    # LDA's joblib work runs in threads so frozen Windows builds
                    # do not need to spawn child processes. Keep native BLAS/OpenMP
                    # work single-threaded inside those workers to avoid nested pools.
                    with parallel_backend("threading", n_jobs=cfg.cpu_workers), threadpool_limits(limits=1):
                        topic_doc = model.fit_transform(counts).astype(np.float32)
                else:
                    topic_doc = model.fit_transform(counts).astype(np.float32)
            components = model.components_ / np.maximum(model.components_.sum(axis=1, keepdims=True), 1e-20)
            topic_weight_kind = "LDA p(词语 | 主题)（逐主题归一化概率）"
            matrix_name = f"LDA 文档主题概率（{n_components} 个主题）"
        doc_matrix = topic_doc
        doc_features = [f"主题{i + 1}" for i in range(n_components)]
        topic_weight_matrix = np.asarray(components, dtype=np.float32)
        topic_terms = _topics(topic_weight_matrix, word_vocab, limit=cfg.topic_top_words)
    elif cfg.method == "cooc-ppmi":
        with _stage(event_callback, "cooc-ppmi", "统计窗口共现并计算 PPMI", cfg.method,
                    work_total=total_tokens, unit="token 位置"):
            cooc_counts, ppmi, cooc_vocab, doc_matrix = _cooccurrence(token_docs, cfg.cooc_vocab_size, cfg.cooc_window,
                                                                     event_callback=event_callback, cancel_event=cancel_event)
        doc_features = cooc_vocab
        matrix_name = "文档 × PPMI 词向量聚合表示"
        if cooc_counts.nnz == 0:
            warnings.append("当前窗口内没有不同词的共现对，PPMI 矩阵为空。")
    elif cfg.method == "word2vec":
        needed = set(word_vocab)
        if not needed:
            raise ValueError("word2vec 需要至少一个非停词的有效词项；请减少停词或检查文本。")
        file_bytes = max(Path(cfg.word2vec_path).stat().st_size, 1)
        scanned_bytes = 0
        with _stage(event_callback, "word2vec-scan", "扫描 word2vec 文件", cfg.method,
                    work_total=file_bytes, unit="字节", completion_done=lambda: scanned_bytes):
            vectors, dimension, lines_scanned, invalid_values, scanned_bytes = _parse_word2vec(
                cfg.word2vec_path, needed, progress=progress, cancel_event=cancel_event, event_callback=event_callback)
        if not vectors:
            if not dimension:
                raise ValueError("没有读取到有效的文本格式 word2vec 向量；二进制 .bin 文件不支持。")
            raise ValueError(f"外部 word2vec 文件与当前词表完全没有交集；扫描 {lines_scanned:,} 行。请检查分词和向量文件词表。")
        idf = None
        if cfg.word2vec_tfidf_weight:
            transformer = TfidfTransformer(smooth_idf=cfg.tfidf_smooth_idf).fit(counts)
            idf = {word: float(value) for word, value in zip(word_vocab, transformer.idf_)}
        doc_matrix = np.zeros((n_docs, dimension), dtype=np.float32)
        oov_counts = np.zeros(n_docs, dtype=np.int64)
        omitted_by_vocab = 0
        with _stage(event_callback, "word2vec-average", "生成 word2vec 文档表示", cfg.method,
                    work_total=n_docs, unit="文档"):
            for i, tokens in enumerate(token_docs):
                _check_cancel(cancel_event)
                token_frequencies = Counter(tokens) if cfg.word2vec_tfidf_weight and cfg.tfidf_sublinear_tf else None
                vec_sum = np.zeros(dimension, dtype=np.float64)
                total_weight = 0.0
                for token in tokens:
                    if token not in needed:
                        omitted_by_vocab += 1
                        continue
                    vector = vectors.get(token)
                    if vector is None:
                        oov_counts[i] += 1
                        continue
                    weight = (idf.get(token, 1.0) if idf is not None else 1.0)
                    if token_frequencies is not None:
                        frequency = token_frequencies[token]
                        weight *= (1 + math.log(frequency)) / frequency
                    vec_sum += vector * weight
                    total_weight += weight
                if total_weight > 0:
                    doc_matrix[i] = (vec_sum / total_weight).astype(np.float32)
                if i + 1 <= 2 or (i + 1) % max(1, n_docs // 40) == 0 or i + 1 == n_docs:
                    _stage_progress(event_callback, "word2vec-average", cfg.method, i + 1, n_docs, "文档",
                                    documents_done=i + 1, documents_total=n_docs)
        doc_features = [f"维度{i + 1}" for i in range(dimension)]
        matrix_name = "外部 word2vec 词向量加权平均"
        all_token_count = int(sum(len(row) for row in token_docs))
        total_oov = int(oov_counts.sum())
        weight_note = "TF-IDF 权重" if cfg.word2vec_tfidf_weight else "等权"
        warnings.append(f"外部文本向量扫描 {lines_scanned:,} 行；命中词表 {len(vectors):,}/{len(needed):,} 个不同词。按{weight_note}平均。词表上限外 token {omitted_by_vocab:,} 个；表内未命中 token {total_oov:,}/{all_token_count:,}（{total_oov / max(all_token_count, 1):.1%}）；命中但含 NaN/Inf 的向量行 {invalid_values:,} 行已跳过。")
    else:
        raise ValueError(f"没有实现的方法：{cfg.method}")

    _check_cancel(cancel_event)
    with _stage(event_callback, "document-metrics", "计算文档指标", cfg.method, work_total=n_docs, unit="文档"):
        if sparse.issparse(doc_matrix):
            matrix_norms = np.sqrt(np.asarray(doc_matrix.multiply(doc_matrix).sum(axis=1)).ravel())
        else:
            matrix_norms = np.linalg.norm(np.asarray(doc_matrix), axis=1)
        zero_mask = matrix_norms <= 1e-12
        zero_ids = [docs[i].doc_id for i, is_zero in enumerate(zero_mask) if is_zero]
        if zero_ids:
            warnings.append(f"{len(zero_ids)} 篇文档的表示为全零，已标记并排除于近邻/聚类；相似度设为空值，不视为完美自相似。")
        _notify(progress, 0.62, "文档表示完成，计算基础指标")

        lengths_chars = [len(text.strip()) for text in raw_texts]
        token_counts = np.asarray([len(tokens) for tokens in token_docs], dtype=np.int64)
        unique_counts = np.asarray([len(set(tokens)) for tokens in token_docs], dtype=np.int64)
        ttr = np.divide(unique_counts, token_counts, out=np.zeros(n_docs, dtype=np.float64), where=token_counts > 0)
        doc_metrics = pd.DataFrame({
            "文档ID": [d.doc_id for d in docs], "来源": [d.source for d in docs], "来源行": [d.source_row for d in docs],
            "字数": lengths_chars, "词数": token_counts, "不同词数": unique_counts, "TTR": ttr,
            "向量范数": matrix_norms.astype(np.float32), "全零表示": zero_mask,
        })

    _check_cancel(cancel_event)
    similar: np.ndarray | None = None
    if cfg.compute_similarity and n_docs <= MAX_FULL_SIMILARITY_DOCS:
        with _stage(event_callback, "similarity", "计算完整相似度矩阵", cfg.method,
                    work_total=n_docs * n_docs, unit="文档对"):
            normalized = normalize(doc_matrix, norm="l2", copy=True)
            similar = (normalized @ normalized.T)
            similar = similar.toarray() if sparse.issparse(similar) else np.asarray(similar, dtype=np.float32)
            if similar.dtype != np.float32:
                similar = similar.astype(np.float32)
            similar[zero_mask, :] = np.nan
            similar[:, zero_mask] = np.nan
            _stage_progress(event_callback, "similarity", cfg.method, n_docs * n_docs, n_docs * n_docs,
                            "文档对", documents_done=n_docs, documents_total=n_docs)
        _notify(progress, 0.70, f"完整余弦相似度矩阵完成：{n_docs} × {n_docs}")
    else:
        if event_callback:
            event_callback({"type": "stage-skipped", "stage_id": "similarity", "method": cfg.method})
        if cfg.compute_similarity and n_docs > MAX_FULL_SIMILARITY_DOCS:
            warnings.append(f"文档数超过 {MAX_FULL_SIMILARITY_DOCS:,} 篇，未生成完整相似度矩阵以限制内存；近邻表仍使用分块计算。")

    if cfg.compute_neighbors:
        with _stage(event_callback, "neighbors", "分块搜索近邻", cfg.method,
                    work_total=n_docs * n_docs, unit="文档对"):
            neighbors = _nearest_neighbors(doc_matrix, [d.doc_id for d in docs], zero_mask, k=cfg.neighbors_k,
                                           cancel_event=cancel_event, event_callback=event_callback)
    else:
        neighbors = pd.DataFrame(columns=["文档ID", "排名", "近邻ID", "余弦相似度"])
        warnings.append("已按设置跳过近邻计算。")
        if event_callback:
            event_callback({"type": "stage-skipped", "stage_id": "neighbors", "method": cfg.method})
    _notify(progress, 0.80, "近邻结果完成，计算聚类")

    cluster_rows: list[dict[str, Any]] = []
    cluster_summary: dict[str, Any] = {"方法": f"KMeans(random_state={cfg.random_state})", "聚类数": cfg.clusters, "轮廓系数": None, "轮廓系数说明": "未计算", "状态": "跳过"}
    valid_indices = np.flatnonzero(~zero_mask)
    if not cfg.compute_clusters or len(valid_indices) < cfg.clusters:
        cluster_summary["轮廓系数说明"] = ("用户已关闭聚类，未计算 KMeans 与轮廓系数。" if not cfg.compute_clusters
                                              else f"有效表示 {len(valid_indices)} 篇少于所设簇数 {cfg.clusters}。")
        if not cfg.compute_clusters:
            cluster_summary["状态"] = "已关闭聚类"
        if event_callback:
            event_callback({"type": "stage-skipped", "stage_id": "kmeans", "method": cfg.method})
            event_callback({"type": "stage-skipped", "stage_id": "silhouette", "method": cfg.method})
    else:
        _check_cancel(cancel_event)
        model = KMeans(n_clusters=cfg.clusters, random_state=cfg.random_state, n_init=10, tol=cfg.kmeans_tol)
        labels = None
        fit_ok = [True]
        with _stage(event_callback, "kmeans", "拟合 KMeans 聚类", cfg.method, work_total=1, unit="拟合",
                    completion_success=lambda: fit_ok[0]):
            try:
                labels = model.fit_predict(doc_matrix[valid_indices])
                for idx, label in zip(valid_indices, labels):
                    cluster_rows.append({"文档ID": docs[int(idx)].doc_id, "簇": int(label) + 1})
            except (ValueError, MemoryError) as exc:
                fit_ok[0] = False
                cluster_summary["轮廓系数说明"] = f"聚类未完成：{exc}"
        if labels is None:
            if event_callback:
                event_callback({"type": "stage-skipped", "stage_id": "silhouette", "method": cfg.method})
        else:
            distinct = np.unique(labels)
            if not cfg.compute_silhouette:
                if event_callback:
                    event_callback({"type": "stage-skipped", "stage_id": "silhouette", "method": cfg.method})
                cluster_summary.update({"轮廓系数说明": "用户已关闭轮廓系数，聚类标签仍保留。", "状态": "已聚类，未计算轮廓系数"})
            elif 1 < len(distinct) < len(labels):
                silhouette_ok = [True]
                with _stage(event_callback, "silhouette", "计算轮廓系数", cfg.method, work_total=1, unit="计算",
                            completion_success=lambda: silhouette_ok[0]):
                    try:
                        if len(labels) > SILHOUETTE_SAMPLE_LIMIT:
                            score = float(silhouette_score(doc_matrix[valid_indices], labels, metric="cosine", sample_size=SILHOUETTE_SAMPLE_LIMIT, random_state=cfg.random_state))
                            note = f"基于随机种子 {cfg.random_state} 抽样 {SILHOUETTE_SAMPLE_LIMIT:,}/{len(labels):,} 篇，cosine 距离。"
                        else:
                            score = float(silhouette_score(doc_matrix[valid_indices], labels, metric="cosine"))
                            note = "使用全部非零表示，cosine 距离。"
                        cluster_summary.update({"轮廓系数": score, "轮廓系数说明": note, "状态": "已计算"})
                    except (ValueError, MemoryError) as exc:
                        silhouette_ok[0] = False
                        cluster_summary.update({"轮廓系数说明": f"轮廓系数未计算：{exc}", "状态": "已聚类，未计算轮廓系数"})
            else:
                if event_callback:
                    event_callback({"type": "stage-skipped", "stage_id": "silhouette", "method": cfg.method})
                cluster_summary.update({"轮廓系数说明": "KMeans 只得到一个有效簇，轮廓系数不定义。", "状态": "已聚类，未计算轮廓系数"})
    cluster_labels = pd.DataFrame(cluster_rows, columns=["文档ID", "簇"])
    _notify(progress, 0.95, "分析完成，整理可导出结果")

    with _stage(event_callback, "result-assembly", "整理分析结果", cfg.method, work_total=n_docs, unit="文档"):
        result = AnalysisResult(
            doc_ids=[d.doc_id for d in docs], method=cfg.method, method_label=METHODS[cfg.method], config=asdict(cfg),
            word_count_matrix=counts, word_tfidf_matrix=word_tfidf, word_vocabulary=word_vocab,
            doc_matrix=doc_matrix, doc_feature_labels=doc_features, doc_matrix_name=matrix_name,
            document_metrics=doc_metrics, word_frequencies=word_freq, nearest_neighbors=neighbors,
            similarity_matrix=similar, topic_document_matrix=topic_doc,
            topic_weight_matrix=topic_weight_matrix, topic_weight_kind=topic_weight_kind, topic_terms=topic_terms,
            cooccurrence_counts=cooc_counts, ppmi_matrix=ppmi, cooccurrence_vocabulary=cooc_vocab,
            cluster_labels=cluster_labels, clustering_summary=cluster_summary, warnings=warnings, zero_document_ids=zero_ids,
        )
    return result

