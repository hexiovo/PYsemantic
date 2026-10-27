"""Small integration exercise for a packaged offline executable."""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import numpy as np
import sys

from .analysis import AnalysisConfig, METHODS, run_analysis
from .data_io import Corpus, Document
from .exports import export_results


def run_self_test(output_directory: str | Path) -> Path:
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "self-test.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        if sys.stdout is not None:
            print(message, flush=True)
    # These records are deliberately artificial and are never presented as research data.
    texts = [
        "SYNTHETIC_ONLY 人工构造文本：苹果手机可以拍照和播放音乐。",
        "SYNTHETIC_ONLY 人工构造文本：智能手机的相机能够拍摄照片，也能听音乐。",
        "SYNTHETIC_ONLY 人工构造文本：医生在医院为患者诊断疾病并开具药物。",
        "SYNTHETIC_ONLY 人工构造文本：病人在诊所接受医生治疗和健康检查。",
        "SYNTHETIC_ONLY 人工构造文本：今天天气晴朗，公园里的人们散步。",
        "SYNTHETIC_ONLY 人工构造文本：明天可能下雨，气象预报提醒大家带伞。",
    ]
    ids = [f"SYNTHETIC_ONLY-{i + 1:02d}" for i in range(len(texts))]
    corpus = Corpus(
        documents=[Document(doc_id=doc_id, text=text, source="SYNTHETIC_ONLY", source_row=i + 1)
                   for i, (doc_id, text) in enumerate(zip(ids, texts))],
        cleaning_log=[{"来源": "SYNTHETIC_ONLY", "读入行/段": len(texts), "空值排除": 0, "去重排除": 0}],
        stats={"读入记录": len(texts), "空值排除": 0, "重复排除": 0, "保留文档": len(texts)},
    )
    vector_path = out / "SYNTHETIC_ONLY_word2vec.txt"
    alphabet = sorted({character for text in texts for character in text if not character.isspace() and character.isalnum()})
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(len(alphabet), 12)).astype(np.float32)
    with vector_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(alphabet)} 12\n")
        for word, vector in zip(alphabet, vectors):
            handle.write(word + " " + " ".join(f"{x:.7f}" for x in vector) + "\n")

    records: list[dict[str, object]] = []
    for method, label in METHODS.items():
        config = AnalysisConfig(
            method=method,
            segmentation="字符" if method == "word2vec" else "jieba精确模式",
            stopwords="无",
            max_features=300,
            min_frequency=1,
            topic_count=3,
            cooc_window=4,
            cooc_vocab_size=100,
            clusters=2,
            word2vec_path=str(vector_path) if method == "word2vec" else "",
            compute_similarity=True,
        )
        result = run_analysis(corpus, config)
        if len(result.doc_ids) != len(texts):
            raise RuntimeError(f"{method}: 文档数不匹配")
        if result.doc_matrix.shape[0] != len(texts):
            raise RuntimeError(f"{method}: 文档表示行数错误")
        if result.similarity_matrix is None or result.similarity_matrix.shape != (len(texts), len(texts)):
            raise RuntimeError(f"{method}: 相似度矩阵未按小语料生成")
        export_dir = export_results(result, corpus, out / method, include_xlsx=(method == "tfidf-word"))
        required = ["document_metrics.csv", "parameters.json", "report.html", "document_representations.npz", "export_manifest.json"]
        missing = [name for name in required if not (export_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{method}: 导出文件缺失：{', '.join(missing)}")
        records.append({
            "method": method,
            "label": label,
            "documents": len(result.doc_ids),
            "representation_shape": list(result.doc_matrix.shape),
            "topic_matrix_shape": None if result.topic_weight_matrix is None else list(result.topic_weight_matrix.shape),
            "similarity_shape": list(result.similarity_matrix.shape),
            "export_directory": str(export_dir),
        })
        log(f"OK {method}: representation={result.doc_matrix.shape}; export={export_dir}")

    manifest = {
        "dataset": "SYNTHETIC_ONLY",
        "note": "All documents and the test word2vec vectors in this smoke run are artificial.",
        "methods": records,
    }
    manifest_path = out / "self-test-results.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    log(f"SELF-TEST OK: {len(records)} methods; manifest sha256={digest}")
    return manifest_path

