"""Download the trained litter detection model if missing."""

from __future__ import annotations

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "litter_inference_graph", "frozen_inference_graph.pb")
EXPECTED_SIZE = 415_611_168


def model_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) == EXPECTED_SIZE


def download_model() -> str:
    if model_ready():
        return MODEL_PATH

    script = os.path.join(PROJECT_ROOT, "download_model.sh")
    if not os.path.exists(script):
        raise FileNotFoundError(f"Missing download script: {script}")

    subprocess.run(["bash", script], check=True, cwd=PROJECT_ROOT)

    if not model_ready():
        raise RuntimeError("Model download failed. Run: bash download_model.sh")

    return MODEL_PATH


if __name__ == "__main__":
    path = download_model()
    print(f"Model ready: {path} ({os.path.getsize(path) // 1_000_000} MB)")
