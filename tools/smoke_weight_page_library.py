#!/usr/bin/env python3
"""Smoke test for weight page library.

1. Parse GGUF tensor directory → build manifest
2. Verify offsets and hashes for N sampled pages
3. Load sampled pages into CPU RAM
4. If CUDA available, copy one page to GPU
5. Evict it
6. Write receipt

Usage:
    PYTHONPATH=src python3 tools/smoke_weight_page_library.py
    PYTHONPATH=src python3 tools/smoke_weight_page_library.py --gguf ~/models/some-model.gguf
"""
import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_GGUF = os.path.expanduser(
    "~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf")
MANIFEST_DIR = os.path.expanduser("~/receipts/cell/weight_pages/")
VERIFY_SAMPLE = 10


def main():
    parser = argparse.ArgumentParser(description="Smoke test weight page library")
    parser.add_argument("--gguf", default=DEFAULT_GGUF, help="GGUF model path")
    args = parser.parse_args()

    gguf_path = os.path.expanduser(args.gguf)
    if not os.path.exists(gguf_path):
        print(f"GGUF not found: {gguf_path}")
        sys.exit(1)

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Step 1: Build manifest ──
    print(f"Step 1: Building manifest from {Path(gguf_path).name}...")
    sys.path.insert(0, str(Path(__file__).parent))
    from build_weight_page_manifest import build_manifest

    manifest = build_manifest(gguf_path, verify_n=VERIFY_SAMPLE)
    manifest_path = os.path.join(
        MANIFEST_DIR, f"{manifest['model_id']}_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  {manifest['total_tensors']} tensors, "
          f"{manifest['total_layers']} layers")
    print(f"  {manifest['verified_count']} pages verified (SHA256)")
    print(f"  Manifest: {manifest_path}")
    print()

    # ── Step 2: Load library from manifest ──
    print("Step 2: Loading weight page library...")
    from cell.weight_pages import WeightPageLibrary

    lib = WeightPageLibrary(manifest_path)
    summary = lib.summary()
    print(f"  Model: {summary['model_id']}")
    print(f"  Tensors: {summary['total_tensors']}, Layers: {summary['total_layers']}")
    print(f"  All on disk: {summary['on_disk']}")
    print()

    # ── Step 3: Load sample pages into RAM ──
    print("Step 3: Loading sample pages into RAM...")
    import random
    sample_names = random.sample(list(lib.pages.keys()),
                                 min(5, len(lib.pages)))

    for name in sample_names:
        page = lib.pages[name]
        data = lib.load_page(name, verify=page.sha256 is not None)
        size_kb = len(data) / 1024
        print(f"  {name}: {size_kb:.0f} KB → RAM "
              f"(layer={page.layer_id}, role={page.tensor_role})")

    summary_after_ram = lib.summary()
    print(f"  RAM usage: {summary_after_ram['ram_bytes'] / 1024 / 1024:.1f} MB "
          f"({summary_after_ram['in_ram']} pages)")
    print()

    # ── Step 4: GPU copy if available ──
    gpu_tested = False
    gpu_copy_ms = None
    try:
        import torch
        if torch.cuda.is_available():
            test_page = sample_names[0]
            page = lib.pages[test_page]
            print(f"Step 4: Copying {test_page} to GPU...")
            t0 = time.time()
            gpu_tensor = lib.copy_to_gpu(test_page)
            gpu_copy_ms = round((time.time() - t0) * 1000, 2)
            print(f"  Copied {page.byte_length / 1024:.0f} KB to GPU "
                  f"in {gpu_copy_ms:.1f}ms")
            print(f"  GPU tensor shape: {gpu_tensor.shape}, "
                  f"device: {gpu_tensor.device}")
            gpu_tested = True

            # ── Step 5: Evict ──
            print("Step 5: Evicting from GPU...")
            lib.evict_page(test_page, from_gpu=True, from_ram=False)
            print(f"  {test_page} evicted from GPU, still in RAM")
            print()
        else:
            print("Step 4: CUDA not available, skipping GPU test")
            print()
    except ImportError:
        print("Step 4: torch not installed, skipping GPU test")
        print()

    # ── Step 6: Evict all from RAM ──
    print("Step 6: Evicting all pages from RAM...")
    for name in sample_names:
        lib.evict_page(name)
    summary_final = lib.summary()
    print(f"  All back on disk: {summary_final['on_disk']} pages")
    print()

    # ── Ops log ──
    ops = lib.get_ops_log()
    print(f"Operations log: {len(ops)} entries")
    for op in ops[:10]:
        print(f"  [{op['op']}] {op['tensor'][:40]}")
    if len(ops) > 10:
        print(f"  ... and {len(ops) - 10} more")
    print()

    # ── Layer structure analysis ──
    print("Layer structure:")
    if manifest['total_layers'] > 0:
        roles_per_layer = {}
        for p in manifest['pages']:
            lid = p.get('layer_id')
            if lid is not None:
                roles_per_layer.setdefault(lid, []).append(p['tensor_role'])
        first_layer = min(roles_per_layer.keys())
        print(f"  Layer {first_layer} roles: {sorted(roles_per_layer[first_layer])}")
        total_bytes_per_role = {}
        for p in manifest['pages']:
            role = p['tensor_role']
            total_bytes_per_role[role] = (
                total_bytes_per_role.get(role, 0) + p['byte_length'])
        print("  Total bytes by role:")
        for role, nbytes in sorted(total_bytes_per_role.items(),
                                    key=lambda x: -x[1]):
            print(f"    {role}: {nbytes / 1024 / 1024:.1f} MB")
    print()

    # ── Write receipt ──
    wall_time = round(time.time() - t_start, 3)
    cpu_time = round(time.process_time() - cpu_start, 3)

    receipt = {
        "receipt_id": f"weight_page_library_v0_{time.strftime('%Y%m%dT%H%M%SZ')}",
        "title": "Weight Page Library v0 Smoke Test",
        "status": "PASS",
        "model_id": manifest["model_id"],
        "gguf_path": gguf_path,
        "total_tensors": manifest["total_tensors"],
        "total_layers": manifest["total_layers"],
        "verified_pages": manifest["verified_count"],
        "pages_loaded_to_ram": len(sample_names),
        "gpu_tested": gpu_tested,
        "gpu_copy_ms": gpu_copy_ms,
        "manifest_path": manifest_path,
        "ops_log": ops,
        "cost": {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    receipt_path = os.path.expanduser(
        f"~/receipts/weight_page_library_v0_{time.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)

    print(f"RESULT: PASS")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Receipt: {receipt_path}")

    lib.close()


if __name__ == "__main__":
    main()
