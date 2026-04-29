#!/usr/bin/env python3
"""Download HXQ GGUF models from HuggingFace for the capsule runtime.

Usage:
    python3 download_models.py smollm3           # Download one model
    python3 download_models.py all               # Download all three
    python3 download_models.py --list             # Show available models
"""
import json
import os
import sys
from pathlib import Path

# Model registry: name → (HF repo, GGUF filename)
MODELS = {
    "smollm3": (
        "EchoLabs33/smollm3-3b-hxq",
        "smollm3-3b-hxq-affine6.gguf",
    ),
    "qwen2.5-coder": (
        "EchoLabs33/qwen2.5-coder-3b-hxq",
        "qwen2.5-coder-3b-instruct-hxq-affine6.gguf",
    ),
    "qwen2.5-sentinel": (
        # Sentinel uses the same coder base with LoRA merged.
        # For container use, the coder GGUF serves as sentinel too
        # (system prompt differentiation). Custom sentinel GGUF
        # can be substituted by mounting a different file.
        "EchoLabs33/qwen2.5-coder-3b-hxq",
        "qwen2.5-coder-3b-instruct-hxq-affine6.gguf",
    ),
}

# Where config expects the files
CONFIG_PATHS = {
    "smollm3": "/models/smollm3-3b-hxq-affine6.gguf",
    "qwen2.5-coder": "/models/qwen2.5-coder-3b-hxq-affine6.gguf",
    "qwen2.5-sentinel": "/models/qwen2.5-sentinel-hxq-affine6.gguf",
}


def download_model(name: str, models_dir: str = "/models"):
    """Download a single model GGUF from HuggingFace."""
    if name not in MODELS:
        print(f"Unknown model: {name}. Available: {', '.join(MODELS.keys())}")
        return False

    repo, filename = MODELS[name]
    target_path = CONFIG_PATHS.get(name, f"{models_dir}/{filename}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path):
        size_gb = os.path.getsize(target_path) / (1024**3)
        print(f"  {name}: already exists ({size_gb:.1f} GB) → {target_path}")
        return True

    print(f"  {name}: downloading {repo}/{filename}...")
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=models_dir,
        )
        # If config expects a different name (e.g., sentinel), symlink
        if target_path != f"{models_dir}/{filename}" and not os.path.exists(target_path):
            os.symlink(f"{models_dir}/{filename}", target_path)
        size_gb = os.path.getsize(local) / (1024**3)
        print(f"  {name}: done ({size_gb:.1f} GB) → {target_path}")
        return True
    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        return False


def main():
    models_dir = os.environ.get("CAPSULE_MODELS_DIR", "/models")

    if len(sys.argv) < 2 or "--help" in sys.argv:
        print(__doc__)
        return

    if "--list" in sys.argv:
        print("Available models:")
        for name, (repo, fname) in MODELS.items():
            exists = os.path.exists(CONFIG_PATHS.get(name, ""))
            status = "PRESENT" if exists else "not downloaded"
            print(f"  {name:20s} → {repo}/{fname}  [{status}]")
        return

    targets = sys.argv[1:]
    if "all" in targets:
        targets = list(MODELS.keys())

    ok = 0
    for name in targets:
        if download_model(name, models_dir):
            ok += 1
    print(f"\n{ok}/{len(targets)} models ready.")


if __name__ == "__main__":
    main()
