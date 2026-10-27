"""Read-only result exploration; all row identities come from document IDs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .analysis import AnalysisResult


def filter_rows(frame: pd.DataFrame, *, query: str = "", cluster_ids: set[str] | None = None,
                minimum: float | None = None, maximum: float | None = None) -> pd.DataFrame:
    """ID query matches either end of a neighbor pair; cluster matches the source."""
    if any(value is not None and (not np.isfinite(value) or not -1 <= value <= 1)
           for value in (minimum, maximum)):
        raise ValueError("相似度上下限必须是 -1 到 1 之间的有限数值。")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("相似度下限不能大于上限。")
    mask = pd.Series(True, index=frame.index)
    query = query.strip()
    if query:
        columns = [key for key in ("文档ID", "近邻ID") if key in frame]
        if not columns:
            columns = [key for key in ("词语", "名称", "关键词", "组别") if key in frame]
        matches = pd.Series(False, index=frame.index)
        for column in columns:
            matches |= frame[column].astype(str).str.contains(query, regex=False, case=False, na=False)
        mask &= matches
    if cluster_ids is not None:
        mask &= frame["文档ID"].astype(str).isin(cluster_ids) if "文档ID" in frame else False
    if "余弦相似度" in frame:
        if minimum is not None:
            mask &= frame["余弦相似度"].ge(minimum)
        if maximum is not None:
            mask &= frame["余弦相似度"].le(maximum)
    return frame.loc[mask].reset_index(drop=True)


def page_rows(frame: pd.DataFrame, page: int, size: int) -> tuple[pd.DataFrame, int, int]:
    if size <= 0:
        raise ValueError("Page size must be positive")
    pages = max(1, (len(frame) + size - 1) // size)
    page = min(max(0, page), pages - 1)
    return frame.iloc[page * size:(page + 1) * size], page, pages


@dataclass
class GroupProfile:
    key: str
    kind: str
    label: str
    keywords: list[str]
    member_ids: list[str]
    scores: list[float]
    score_label: str
    explanation: str
    representative_count: int = 3


def _top_words(values: Any, vocabulary: list[str], limit: int = 10) -> list[str]:
    weights = np.asarray(values).ravel()
    indices = np.flatnonzero(np.isfinite(weights) & (weights > 0))
    indices = indices[np.argsort(-weights[indices], kind="stable")[:limit]]
    return [vocabulary[int(index)] for index in indices]


def build_profiles(result: AnalysisResult) -> list[GroupProfile]:
    """Centroid distances use the original KMeans feature space, not a projection."""
    profiles: list[GroupProfile] = []
    keyword_limit = int(result.config.get("profile_top_words", 10))
    representative_count = int(result.config.get("representative_count", 3))
    positions = {doc_id: i for i, doc_id in enumerate(result.doc_ids)}
    zero = set(result.zero_document_ids)
    clusters = result.cluster_labels
    if not clusters.empty:
        for cluster, rows in clusters.groupby("簇", sort=True):
            ids = [str(value) for value in rows["文档ID"] if str(value) in positions and str(value) not in zero]
            if not ids:
                continue
            indices = np.asarray([positions[doc_id] for doc_id in ids], dtype=int)
            matrix = result.doc_matrix[indices]
            centroid = np.asarray(matrix.mean(axis=0)).ravel()
            if sparse.issparse(matrix):
                squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
            else:
                squared = np.einsum("ij,ij->i", matrix, matrix)
            distances = np.sqrt(np.maximum(0, squared - 2 * np.asarray(matrix @ centroid).ravel()
                                           + float(centroid @ centroid)))
            order = np.argsort(distances, kind="stable")
            keywords = _top_words(result.word_tfidf_matrix[indices].mean(axis=0), result.word_vocabulary, keyword_limit)
            profiles.append(GroupProfile(
                f"cluster:{int(cluster)}", "聚类", f"簇 {int(cluster)}", keywords,
                [ids[int(i)] for i in order], [float(distances[i]) for i in order], "距簇中心",
                "关键词按簇内平均词级 TF-IDF 排序；代表文本按原始表示空间中距簇中心的欧氏距离从小到大排列。", representative_count))
    if result.topic_document_matrix is None or result.topic_weight_matrix is None:
        return profiles
    weights = np.asarray(result.topic_document_matrix)
    terms = np.asarray(result.topic_weight_matrix)
    valid = np.asarray([doc_id not in zero for doc_id in result.doc_ids])
    finite = np.all(np.isfinite(weights), axis=1)
    if result.method == "lsa":
        for index in range(weights.shape[1]):
            for sign, pole in ((1, "正向"), (-1, "负向")):
                values = sign * weights[:, index]
                members = np.flatnonzero(valid & finite & (values > 1e-12))
                order = members[np.argsort(-values[members], kind="stable")]
                profiles.append(GroupProfile(
                    f"lsa:{index + 1}:{sign}", "LSA 维度", f"维度 {index + 1} · {pole}",
                    _top_words(sign * terms[index], result.word_vocabulary, keyword_limit),
                    [result.doc_ids[int(i)] for i in order], [float(weights[i, index]) for i in order], "维度载荷",
                    "LSA 按载荷正负分开展示；各维度成员可重叠，不是互斥主题或概率。关键词与文档均按该方向载荷强度排序。", representative_count))
    else:
        dominant = np.argmax(weights, axis=1)
        nonzero = np.max(weights, axis=1) > 1e-12
        for index in range(weights.shape[1]):
            members = np.flatnonzero(valid & finite & nonzero & (dominant == index))
            order = members[np.argsort(-weights[members, index], kind="stable")]
            profiles.append(GroupProfile(
                f"topic:{index + 1}", "主题", f"主题 {index + 1}",
                _top_words(terms[index], result.word_vocabulary, keyword_limit),
                [result.doc_ids[int(i)] for i in order], [float(weights[i, index]) for i in order],
                "主题概率" if result.method == "lda" else "主题权重",
                "成员按最大主题权重归属（并列时取编号较小者），代表文本按本主题权重降序；这是浏览分组，不改变原始主题矩阵。"
                + ("LDA 权重为主题概率。" if result.method == "lda" else "NMF 权重不是概率。"), representative_count))
    return profiles


def profile_table(profiles: list[GroupProfile], names: dict[str, str] | None = None) -> pd.DataFrame:
    names = names or {}
    return pd.DataFrame([{"组别": item.label, "名称": names.get(item.key, "") or item.label,
                          "文档数": len(item.member_ids), "关键词": "、".join(item.keywords),
                          "代表文档ID": "、".join(item.member_ids[:item.representative_count])} for item in profiles],
                        columns=["组别", "名称", "文档数", "关键词", "代表文档ID"])
