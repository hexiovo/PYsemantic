"""Download pinned public BGE ONNX files for offline packaging (never called by the app)."""
from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
import os
import sys


REPO = "Xenova/bge-small-zh-v1.5"
REVISION = "75c43b069aac4d136ba6bc1122f995fedcfd2781"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def download(relative: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{relative}"
    temp = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "ChineseSemanticToolkit/1.0 model-packager"})
    digest = hashlib.sha256()
    total = 0
    with urlopen(request, timeout=90) as response, temp.open("wb") as output:
        expected = int(response.headers.get("Content-Length", "0") or 0)
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)
            total += len(block)
            if expected:
                print(f"{relative}: {total:,}/{expected:,} bytes ({total / expected:.0%})", flush=True)
    if temp.stat().st_size != total or total == 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"download size check failed: {relative}")
    os.replace(temp, destination)
    return {"path": relative, "url": url, "sha256": digest.hexdigest(), "bytes": total}


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "assets" / "bge-small-zh-v1.5"
    entries = []
    for relative in FILES:
        local = target / relative
        if local.is_file() and local.stat().st_size > 0:
            file_hash = hashlib.sha256(local.read_bytes()).hexdigest()
            entries.append({"path": relative, "url": f"{BASE_URL}/{relative}", "sha256": file_hash, "bytes": local.stat().st_size})
            print(f"Already present: {relative} ({local.stat().st_size:,} bytes)", flush=True)
        else:
            entries.append(download(relative, local))
    manifest = {
        "model": "BAAI/bge-small-zh-v1.5",
        "format": "Xenova ONNX conversion; original upstream weights, no fine-tuning",
        "repository": REPO,
        "revision": REVISION,
        "source_url": "https://huggingface.co/BAAI/bge-small-zh-v1.5",
        "onnx_source_url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/LICENSE",
        "files": entries,
    }
    (target / "model-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved pinned model resources under {target}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Model download failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)

