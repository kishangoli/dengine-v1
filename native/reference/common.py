"""Pinned model location and integrity checks for Stage 1 reference tools."""

import hashlib
import json
from pathlib import Path


REFERENCE_DIR = Path(__file__).resolve().parent
MODEL_DIR = REFERENCE_DIR.parent / "models" / "smollm2-135m-instruct"
MODEL_INFO = json.loads((REFERENCE_DIR / "model_info.json").read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_files() -> None:
    for name, expected_hash in MODEL_INFO["files"].items():
        path = MODEL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run download_model.py")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash}")
