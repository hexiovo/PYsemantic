"""Offline export bundle writer. Sparse matrices remain exact in NPZ files."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime
import html
import json
import math
import re
import shutil
import uuid
from time import monotonic

import numpy as np
import pandas as pd
from scipy import sparse

from .analysis import AnalysisResult
from .data_io import Corpus
from .exploration import build_profiles, profile_table
from .resources import resource_path

MAX_TRIPLET_CSV_ENTRIES = 2_000_000
EXCEL_MAX_CELLS_PER_SHEET = 1_000_000


def _unique_output_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("中文语义分析_%Y%m%d_%H%M%S")
    candidate = parent / stem
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("结果导出已取消")


def _event(progress: Any, value: dict[str, Any]) -> None:
    if progress:
        progress(value)


def _stage(progress: Any, stage_id: str, label: str, method: str, *, work_total: float | None = None,
           unit: str = "工作量"):
    from contextlib import contextmanager

    @contextmanager
    def scoped():
        started = monotonic()
        _event(progress, {"type": "stage-start", "stage_id": stage_id, "label": label, "method": method,
                         "work_total": work_total, "unit": unit, "initial_seconds": 1.0})
        try:
            yield
        except BaseException:
            _event(progress, {"type": "stage-failed", "stage_id": stage_id, "method": method,
                              "elapsed_seconds": max(0.0, monotonic() - started)})
            raise
        else:
            _event(progress, {"type": "stage-complete", "stage_id": stage_id, "method": method,
                              "work_done": work_total, "work_total": work_total, "unit": unit,
                              "elapsed_seconds": max(0.0, monotonic() - started)})
    return scoped()


def _csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _labels_file(path: Path, rows: list[str], columns: list[str]) -> None:
    path.write_text(json.dumps({"rows": rows, "columns": columns}, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_sparse(matrix: sparse.spmatrix, stem: str, out: Path, row_labels: list[str], col_labels: list[str], skipped: list[str]) -> None:
    filename = f"{stem}.npz"
    sparse.save_npz(out / filename, matrix.tocsr(), compressed=True)
    _labels_file(out / f"{stem}.labels.json", row_labels, col_labels)
    csr = matrix.tocsr()
    if csr.nnz <= MAX_TRIPLET_CSV_ENTRIES:
        coo = csr.tocoo()
        frame = pd.DataFrame({"行标签": [row_labels[i] for i in coo.row], "列标签": [col_labels[i] for i in coo.col], "数值": coo.data})
        _csv(frame, out / f"{stem}.sparse.csv")
    else:
        skipped.append(f"{stem}.sparse.csv：非零项 {csr.nnz:,} 超过 CSV 安全阈值 {MAX_TRIPLET_CSV_ENTRIES:,}；完整矩阵保存在 NPZ。")


def _matrix_to_frame(matrix: Any, row_labels: list[str], col_labels: list[str]) -> pd.DataFrame:
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return pd.DataFrame(values, index=row_labels, columns=col_labels).rename_axis("文档ID").reset_index()


def _write_xlsx(result: AnalysisResult, corpus: Corpus, out: Path, skipped: list[str]) -> None:
    path = out / "analysis.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result.document_metrics.to_excel(writer, index=False, sheet_name="文档指标")
        result.word_frequencies.to_excel(writer, index=False, sheet_name="词频")
        result.nearest_neighbors.to_excel(writer, index=False, sheet_name="近邻")
        result.cluster_labels.to_excel(writer, index=False, sheet_name="聚类")
        result.topic_terms.to_excel(writer, index=False, sheet_name="主题词")
        pd.DataFrame(corpus.cleaning_log).to_excel(writer, index=False, sheet_name="来源与清洗")
        matrices = [
            {"矩阵": "文档词项计数", "文件": "document_term_counts.npz", "说明": "完整稀疏矩阵与标签见 NPZ / labels JSON"},
            {"矩阵": "文档词项 TF-IDF", "文件": "document_tfidf.npz", "说明": "完整稀疏矩阵与标签见 NPZ / labels JSON"},
            {"矩阵": "文档表示", "文件": "document_representations.npz", "说明": "完整表示见 NPZ"},
            {"矩阵": "主题 × 词项权重", "文件": "topic_term_matrix.npz", "说明": result.topic_weight_kind},
            {"矩阵": "相似度矩阵", "文件": "similarity_matrix.csv / similarity_matrix.npz", "说明": "完整矩阵见 CSV / NPZ；仅在文档数上限内生成" if result.similarity_matrix is not None else "未生成或超过完整矩阵文档数上限"},
            {"矩阵": "共现 / PPMI", "文件": "cooccurrence_*.npz", "说明": "完整稀疏矩阵与标签见 NPZ / labels JSON"},
        ]
        pd.DataFrame(matrices).to_excel(writer, index=False, sheet_name="矩阵文件")
        if result.similarity_matrix is not None:
            n = len(result.doc_ids)
            rows_per_sheet = max(1, min(900_000 // max(n, 1), 1_048_575))
            for part_no, start in enumerate(range(0, n, rows_per_sheet), 1):
                stop = min(n, start + rows_per_sheet)
                frame = pd.DataFrame(result.similarity_matrix[start:stop], columns=result.doc_ids)
                frame.insert(0, "文档ID", result.doc_ids[start:stop])
                frame.to_excel(writer, index=False, sheet_name=f"相似度{part_no}")


def export_results(result: AnalysisResult, corpus: Corpus, parent_directory: str | Path, *, include_xlsx: bool = False,
                   staging_directory: str | Path | None = None, progress: Any = None,
                   cancel_event: Any = None, method_index: int | None = None, method_total: int | None = None,
                   cluster_names: dict[str, str] | None = None) -> Path:
    """Write into a task-owned sibling directory and atomically publish on success."""
    parent = Path(parent_directory).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(staging_directory).resolve() if staging_directory else parent / f".cst-pending-{uuid.uuid4().hex}"
    if out.parent != parent:
        raise ValueError("導出暫存目錄必須位於所選輸出父目錄中，才能原子提交。")
    if out.exists():
        raise FileExistsError(f"導出暫存目錄已存在：{out}")
    out.mkdir()
    skipped: list[str] = []
    try:
        with _stage(progress, "export-tables", "导出结果表格与矩阵", result.method,
                    work_total=14, unit="导出项目"):
            _check_cancel(cancel_event)
            config_payload = dict(result.config)
            config_payload.update({"method_label": result.method_label, "method": result.method})
            (out / "parameters.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tables = [(result.document_metrics, "document_metrics.csv"), (result.word_frequencies, "word_frequencies.csv"),
                      (result.nearest_neighbors, "nearest_neighbors.csv"), (result.cluster_labels, "cluster_labels.csv"),
                      (result.topic_terms, "topic_terms.csv"), (pd.DataFrame(corpus.cleaning_log), "source_and_cleaning.csv"),
                      (pd.DataFrame([corpus.stats]), "corpus_summary.csv")]
            for index, (frame, name) in enumerate(tables, 1):
                _check_cancel(cancel_event)
                _csv(frame, out / name)
                _event(progress, {"type": "stage-progress", "stage_id": "export-tables", "method": result.method,
                                  "work_done": index, "work_total": 14, "unit": "导出项目"})
            profiles = build_profiles(result)
            _check_cancel(cancel_event)
            _csv(profile_table(profiles, cluster_names), out / "group_profiles.csv")
            member_rows = [{"组别": profile.label, "文档ID": doc_id, "排名": rank,
                            "指标": profile.score_label, "数值": score}
                           for profile in profiles for rank, (doc_id, score) in
                           enumerate(zip(profile.member_ids, profile.scores), 1)]
            _csv(pd.DataFrame(member_rows, columns=["组别", "文档ID", "排名", "指标", "数值"]),
                 out / "group_members.csv")
            (out / "group_profiles.json").write_text(json.dumps({
                "cluster_names": cluster_names or {},
                "rules": {profile.label: profile.explanation for profile in profiles}}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            _save_sparse(result.word_count_matrix, "document_term_counts", out, result.doc_ids, result.word_vocabulary, skipped)
            _save_sparse(result.word_tfidf_matrix, "document_tfidf", out, result.doc_ids, result.word_vocabulary, skipped)
            if sparse.issparse(result.doc_matrix):
                _save_sparse(result.doc_matrix, "document_representations", out, result.doc_ids, result.doc_feature_labels, skipped)
            else:
                np.savez_compressed(out / "document_representations.npz", matrix=np.asarray(result.doc_matrix, dtype=np.float32),
                                    doc_ids=np.asarray(result.doc_ids, dtype=str), feature_labels=np.asarray(result.doc_feature_labels, dtype=str))
            if result.topic_document_matrix is not None:
                np.savez_compressed(out / "topic_document_matrix.npz", matrix=result.topic_document_matrix,
                                    doc_ids=np.asarray(result.doc_ids, dtype=str), topic_labels=np.asarray([f"主题{i + 1}" for i in range(result.topic_document_matrix.shape[1])], dtype=str))
            if result.topic_weight_matrix is not None:
                topic_labels = [f"主题{i + 1}" for i in range(result.topic_weight_matrix.shape[0])]
                np.savez_compressed(out / "topic_term_matrix.npz", matrix=result.topic_weight_matrix,
                                    topic_labels=np.asarray(topic_labels, dtype=str), term_labels=np.asarray(result.word_vocabulary, dtype=str),
                                    weight_kind=np.asarray([result.topic_weight_kind], dtype=str))
                (out / "topic_term_matrix.labels.json").write_text(
                    json.dumps({"rows": topic_labels, "columns": result.word_vocabulary, "weight_kind": result.topic_weight_kind}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                if result.topic_weight_matrix.size <= MAX_TRIPLET_CSV_ENTRIES:
                    rows, cols = np.nonzero(result.topic_weight_matrix)
                    frame = pd.DataFrame({"主题": [topic_labels[i] for i in rows], "词语": [result.word_vocabulary[i] for i in cols],
                                          "权重": result.topic_weight_matrix[rows, cols]})
                    _csv(frame, out / "topic_term_matrix.sparse.csv")
                else:
                    skipped.append(f"topic_term_matrix.sparse.csv：矩阵单元 {result.topic_weight_matrix.size:,} 超过 CSV 安全阈值；完整矩阵保存在 NPZ。")
            if result.similarity_matrix is not None:
                np.savez_compressed(out / "similarity_matrix.npz", matrix=result.similarity_matrix, doc_ids=np.asarray(result.doc_ids, dtype=str))
                _csv(_matrix_to_frame(result.similarity_matrix, result.doc_ids, result.doc_ids), out / "similarity_matrix.csv")
            if result.cooccurrence_counts is not None:
                _save_sparse(result.cooccurrence_counts, "word_cooccurrence_counts", out, result.cooccurrence_vocabulary, result.cooccurrence_vocabulary, skipped)
            if result.ppmi_matrix is not None:
                _save_sparse(result.ppmi_matrix, "word_ppmi", out, result.cooccurrence_vocabulary, result.cooccurrence_vocabulary, skipped)
        with _stage(progress, "export-report", "生成离线报告", result.method, work_total=4, unit="文件"):
            _check_cancel(cancel_event)
            references_path = resource_path("references.json")
            if references_path.is_file():
                shutil.copy2(references_path, out / "references.json")
            notes_path = resource_path("METHOD_NOTES.md")
            if notes_path.is_file():
                shutil.copy2(notes_path, out / "METHOD_NOTES.md")
            notices_path = resource_path("THIRD_PARTY_NOTICES.txt")
            if notices_path.is_file():
                shutil.copy2(notices_path, out / "THIRD_PARTY_NOTICES.txt")
            model_manifest = resource_path("assets/bge-small-zh-v1.5/model-manifest.json")
            if model_manifest.is_file():
                shutil.copy2(model_manifest, out / "model-manifest.json")
            manifest = {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "method": result.method_label,
                "documents": len(result.doc_ids),
                "warnings": result.warnings,
                "clustering": result.clustering_summary,
                "zero_document_ids": result.zero_document_ids,
                "csv_threshold_notes": skipped,
                "matrix_files": sorted(p.name for p in out.glob("*.npz")),
                "full_similarity_generated": result.similarity_matrix is not None,
                "method_index": method_index,
                "method_total": method_total,
            }
            (out / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_html_report(result, corpus, out, skipped)
        if include_xlsx:
            with _stage(progress, "export-xlsx", "生成 XLSX 工作簿", result.method, work_total=1, unit="工作簿"):
                _check_cancel(cancel_event)
                _write_xlsx(result, corpus, out, skipped)
                manifest["csv_threshold_notes"] = skipped
                (out / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _check_cancel(cancel_event)
        final = _unique_output_dir(parent)
        suffix = 2
        while final.exists():
            final = parent / f"{final.name}_{suffix}"
            suffix += 1
        out.replace(final)
        return final
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _table_html(frame: pd.DataFrame, rows: int = 12) -> str:
    if frame.empty:
        return "<p>无结果</p>"
    return frame.head(rows).to_html(index=False, border=0, classes="data", escape=True)


def _write_html_report(result: AnalysisResult, corpus: Corpus, out: Path, skipped: list[str]) -> None:
    config_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in result.config.items())
    warning_html = "".join(f"<li>{html.escape(item)}</li>" for item in result.warnings) or "<li>无</li>"
    skipped_html = "".join(f"<li>{html.escape(item)}</li>" for item in skipped) or "<li>完整导出，无 CSV 阈值省略。</li>"
    clustering = "；".join(f"{html.escape(str(k))}：{html.escape(str(v))}" for k, v in result.clustering_summary.items())
    title = html.escape(result.method_label)
    source_names = "、".join(html.escape(Path(row.get("来源", "")).name or str(row.get("来源", ""))) for row in corpus.cleaning_log)
    body = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>中文语义分析报告</title>
<style>
body{{font-family:'Microsoft YaHei UI','Segoe UI',sans-serif;max-width:1080px;margin:36px auto;padding:0 20px;color:#27324a;background:#fcf8f2;line-height:1.55}}
h1,h2{{border:2px solid #27324a;padding:12px 16px;background:#e6f7fb}} h1{{background:#e9fbf1}} section{{border:2px solid #27324a;padding:16px;margin:16px 0;background:#fff}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}}th,td{{border-bottom:1px solid #d9dfeb;padding:7px 9px;text-align:left;white-space:nowrap}}th{{background:#e6e6fa}}
.callout{{padding:10px;background:#f1f9d8;border-left:5px solid #27324a}}small{{color:#555}}
</style><body>
<h1>中文语义分析报告</h1><p><b>方法：</b>{title}　<b>文档数：</b>{len(result.doc_ids)}　<b>来源：</b>{source_names}</p>
<p class="callout">语义相似度是模型或向量空间中的余弦值，不是概率、准确率或人工标注结论。</p>
<section><h2>计算设置</h2><table>{config_rows}</table><p><b>聚类：</b>{clustering}</p><p><b>零向量文档：</b>{len(result.zero_document_ids)}</p></section>
<section><h2>文档指标（前 100 篇）</h2>{_table_html(result.document_metrics, 100)}</section>
<section><h2>高频词（前 100）</h2>{_table_html(result.word_frequencies, 100)}</section>
<section><h2>近邻（前 100 行）</h2>{_table_html(result.nearest_neighbors, 100)}</section>
<section><h2>主题词</h2>{_table_html(result.topic_terms, 200)}</section>
<section><h2>注意事项</h2><ul>{warning_html}</ul></section>
    <section><h2>导出说明</h2><p>完整矩阵数据保存在 NPZ 文件中，标签位于对应 labels JSON。稀疏矩阵同时提供三元组 CSV。主题词权重口径：{html.escape(result.topic_weight_kind)}。</p><ul>{skipped_html}</ul>
<p>参数保存在 parameters.json；文献来源及适用范围见 references.json。</p></section>
<footer><small>离线 HTML，无外部脚本或网络资源。导出时间：{html.escape(datetime.now().astimezone().isoformat(timespec='seconds'))}</small></footer>
</body></html>"""
    (out / "report.html").write_text(body, encoding="utf-8")

