#!/usr/bin/env python3
"""Build an addressable weight-page manifest from a GGUF model file.

Each tensor in the GGUF becomes a page entry with:
  - tensor identity (name, layer, role)
  - physical location (offset, length, shape, dtype)
  - verification (sha256)
  - residency tracking (disk/ram/gpu)

Usage:
    python3 tools/build_weight_page_manifest.py ~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
    python3 tools/build_weight_page_manifest.py ~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf --verify 10
"""
import argparse
import hashlib
import json
import mmap
import os
import re
import sys
import time
from pathlib import Path

import gguf


def parse_tensor_role(name: str) -> tuple[int | None, str]:
    """Extract layer_id and tensor role from GGUF tensor name.

    Returns (layer_id, role).
    layer_id is None for non-layer tensors (embeddings, output, norms).
    """
    # Layer tensors: blk.N.xxx
    m = re.match(r'blk\.(\d+)\.(.+)', name)
    if m:
        layer_id = int(m.group(1))
        suffix = m.group(2)
        role_map = {
            'attn_q.weight': 'attn_q',
            'attn_k.weight': 'attn_k',
            'attn_v.weight': 'attn_v',
            'attn_output.weight': 'attn_o',
            'ffn_gate.weight': 'ffn_gate',
            'ffn_up.weight': 'ffn_up',
            'ffn_down.weight': 'ffn_down',
            'ffn_gate_inp.weight': 'moe_gate',
            'attn_norm.weight': 'attn_norm',
            'ffn_norm.weight': 'ffn_norm',
            # SSM/Mamba tensors
            'ssm_in.weight': 'ssm_in',
            'ssm_out.weight': 'ssm_out',
            'ssm_conv1d.weight': 'ssm_conv1d',
            'ssm_dt.weight': 'ssm_dt',
            'ssm_a': 'ssm_a',
            'ssm_d': 'ssm_d',
        }
        role = role_map.get(suffix, suffix.replace('.weight', ''))
        return layer_id, role

    # Non-layer tensors
    if name == 'token_embd.weight':
        return None, 'embed'
    if name == 'output.weight':
        return None, 'output_head'
    if name == 'output_norm.weight':
        return None, 'output_norm'
    return None, name.replace('.weight', '')


def sha256_region(filepath: str, offset: int, length: int) -> str:
    """SHA256 hash of a byte region in a file using mmap."""
    with open(filepath, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            h = hashlib.sha256()
            h.update(mm[offset:offset + length])
            return h.hexdigest()


def build_manifest(gguf_path: str, verify_n: int = 0) -> dict:
    """Build weight page manifest from a GGUF file."""
    gguf_path = os.path.expanduser(gguf_path)
    reader = gguf.GGUFReader(gguf_path)

    file_size = os.path.getsize(gguf_path)
    model_id = Path(gguf_path).stem

    pages = []
    layers_seen = set()

    for t in reader.tensors:
        name = t.name
        layer_id, role = parse_tensor_role(name)
        if layer_id is not None:
            layers_seen.add(layer_id)

        try:
            dtype_name = gguf.GGMLQuantizationType(t.tensor_type).name
        except (ValueError, KeyError):
            dtype_name = f"type_{t.tensor_type}"

        page = {
            "tensor_name": name,
            "layer_id": layer_id,
            "tensor_role": role,
            "byte_offset": int(t.data_offset),
            "byte_length": int(t.n_bytes),
            "shape": [int(s) for s in t.shape],
            "dtype_or_quant": dtype_name,
            "residency": "disk",
            "hotness_score": 0.0,
            "last_used": None,
            "sha256": None,
            "receipt_id": None,
        }
        pages.append(page)

    # Verify N sampled pages if requested
    verified = 0
    if verify_n > 0:
        import random
        sample = random.sample(pages, min(verify_n, len(pages)))
        for p in sample:
            p["sha256"] = sha256_region(
                gguf_path, p["byte_offset"], p["byte_length"])
            verified += 1

    manifest = {
        "model_id": model_id,
        "gguf_path": gguf_path,
        "file_size_bytes": file_size,
        "total_tensors": len(pages),
        "total_layers": len(layers_seen),
        "layer_range": [min(layers_seen), max(layers_seen)] if layers_seen else [],
        "pages": pages,
        "verified_count": verified,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build weight-page manifest from GGUF")
    parser.add_argument("gguf_path", help="Path to GGUF model file")
    parser.add_argument("--verify", type=int, default=0,
                        help="Number of pages to verify (SHA256)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: stdout)")
    args = parser.parse_args()

    t0 = time.time()
    manifest = build_manifest(args.gguf_path, verify_n=args.verify)
    elapsed = time.time() - t0

    manifest["build_time_s"] = round(elapsed, 3)

    output = json.dumps(manifest, indent=2)
    if args.output:
        out_path = os.path.expanduser(args.output)
        with open(out_path, 'w') as f:
            f.write(output)
        print(f"Manifest written to {out_path}")
        print(f"  {manifest['total_tensors']} tensors, "
              f"{manifest['total_layers']} layers, "
              f"{manifest['verified_count']} verified, "
              f"{elapsed:.1f}s")
    else:
        print(output)


if __name__ == "__main__":
    main()
