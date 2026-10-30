"""Verify real BGE GPU execution and compare it with the CPU runtime."""
from pathlib import Path
import argparse
import json
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from semantic_toolkit.analysis import AnalysisConfig, _create_bge_session


def run(output: Path, device: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / "assets/bge-small-zh-v1.5"
    tokens = Tokenizer.from_file(str(model / "tokenizer.json")).encode("这是一项中文语义分析与显卡加速测试。")
    ids = np.asarray([tokens.ids], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
    original_options = ort.SessionOptions

    def profiled_options():
        options = original_options()
        options.enable_profiling = True
        options.profile_file_prefix = str(output / "gpu-profile")
        return options

    with patch.object(ort, "SessionOptions", side_effect=profiled_options):
        gpu, info = _create_bge_session(ort, model / "onnx/model.onnx", AnalysisConfig(bge_device=device))
    # Adapt to the model's declared inputs once at the input boundary.
    gpu_input = {item.name: feed[item.name].astype(np.int32 if "int32" in item.type else np.int64)
                 for item in gpu.get_inputs()}
    gpu_outputs = gpu.run(None, gpu_input)
    profile_path = Path(gpu.end_profiling())
    counts = {}
    for event in json.loads(profile_path.read_text(encoding="utf-8")):
        provider = event.get("args", {}).get("provider")
        if provider:
            counts[provider] = counts.get(provider, 0) + 1
    if counts.get(info["selected_provider"], 0) == 0:
        raise AssertionError(f"No actual GPU kernel events: {counts}")
    del gpu
    cpu, _ = _create_bge_session(ort, model / "onnx/model.onnx", AnalysisConfig(bge_device="cpu"))
    cpu_outputs = cpu.run(None, gpu_input)
    gpu_cls, cpu_cls = gpu_outputs[0][0, 0].astype(float), cpu_outputs[0][0, 0].astype(float)
    cosine = float(gpu_cls @ cpu_cls / (np.linalg.norm(gpu_cls) * np.linalg.norm(cpu_cls)))
    max_abs = float(np.max(np.abs(gpu_outputs[0] - cpu_outputs[0])))
    if cosine < .9999:
        raise AssertionError(f"GPU/CPU CLS direction differs: cosine={cosine}")
    result = {"status": "PASS", "runtime": ort.__version__, "device": info,
              "profile_node_counts": counts, "cls_cosine_cpu_gpu": cosine,
              "max_abs_hidden_state_difference": max_abs, "profile": str(profile_path)}
    (output / "gpu-runtime-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "directml"], default="directml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.device)
