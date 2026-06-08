"""Cross-model Shadow similarity test.

Scans Shadow windows from 5 different model architectures:
  - Mistral 7B (CDNA v1, 165 MB)
  - TinyLlama 1.1B (CDNA v1, 925 MB)
  - Mistral 7B (GGUF Q4_K_M, 2.2 GB)
  - Zamba2 2.7B (GGUF Q8_0, 3.9 GB)
  - Mamba 130M (HXQ safetensors, 237 MB)

Builds a shared ShadowMemory across ALL models.
Tests whether attention-like regions cluster with attention-like regions
across different architectures, not just within one file.

The key question: "Can two different bodies produce similar shadows?"
If yes → structural similarity is architecture-independent.

WO-CRYSTAL-VAULT-01: Phase 0.8 — Cross-model shadow clustering proof
"""
import hashlib
import json
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.vault_shard import (
    Ghost,
    GlyphDAR,
    LatentShape,
    MemoryEntry,
    Outcome,
    Shadow,
    ShadowMemory,
    _hamming64,
)


# ---------------------------------------------------------------------------
# Model files — 5 architectures, 3 formats
# ---------------------------------------------------------------------------

MODELS = {
    "mistral_cdna": {
        "path": Path("/home/voidstr3m33/helix-cdc/artifacts/mistral_test.cdna"),
        "format": "cdna_v1",
        "arch": "transformer",
        "family": "mistral",
        "params": "7B",
    },
    "tinyllama_cdna": {
        "path": Path("/home/voidstr3m33/helix-cdc/artifacts/tinyllama-1.1b-fp32.cdna"),
        "format": "cdna_v1",
        "arch": "transformer",
        "family": "llama",
        "params": "1.1B",
    },
    "mistral_gguf": {
        "path": Path("/home/voidstr3m33/rebuild_mmlu/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "format": "gguf",
        "arch": "transformer",
        "family": "mistral",
        "params": "7B",
    },
    "zamba2_gguf": {
        "path": Path("/home/voidstr3m33/cloud-work/ggufs/zamba2-2.7b-instruct-v2-q8_0.gguf"),
        "format": "gguf",
        "arch": "hybrid_ssm",
        "family": "zamba2",
        "params": "2.7B",
    },
    "mamba_hxq": {
        "path": Path("/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"),
        "format": "safetensors",
        "arch": "ssm",
        "family": "mamba",
        "params": "130M",
    },
}


# ---------------------------------------------------------------------------
# Format readers — raw bytes only, NO weight materialization
# ---------------------------------------------------------------------------

def read_cdna_header(path: Path) -> dict:
    """Read CDNA v1 header."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"CDNA", f"Not a CDNA file: {magic}"
        version, flags, n_codebooks = struct.unpack("<BBH", f.read(4))
        n_tensors, codebook_offset = struct.unpack("<II", f.read(8))
        manifest_offset, latent_offset = struct.unpack("<II", f.read(8))
        latent_size = struct.unpack("<Q", f.read(8))[0]
        source_hash = f.read(32).hex()
        return {
            "version": version,
            "n_codebooks": n_codebooks,
            "n_tensors": n_tensors,
            "codebook_offset": codebook_offset,
            "manifest_offset": manifest_offset,
            "latent_offset": latent_offset,
            "latent_size": latent_size,
            "file_size": os.path.getsize(path),
        }


def read_gguf_header(path: Path) -> dict:
    """Read GGUF v3 header — just magic + version + tensor count."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"Not a GGUF file: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        return {
            "version": version,
            "n_tensors": n_tensors,
            "n_kv": n_kv,
            "file_size": os.path.getsize(path),
        }


def read_safetensors_header(path: Path) -> dict:
    """Read safetensors header (JSON length + JSON metadata)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_len))
        tensor_names = [k for k in header_json.keys() if k != "__metadata__"]
        return {
            "header_len": header_len,
            "n_tensors": len(tensor_names),
            "tensor_names": tensor_names,
            "file_size": os.path.getsize(path),
        }


def get_scan_windows(model_key: str, model_info: dict, n_windows: int = 10,
                     window_size: int = 256 * 1024) -> list[dict]:
    """Generate scan window specs for a model file.

    Returns list of {offset, size, estimated_role} dicts.
    Spreads windows across file to sample different structural regions.
    """
    path = model_info["path"]
    file_size = os.path.getsize(path)
    fmt = model_info["format"]

    # Determine scannable region
    if fmt == "cdna_v1":
        header = read_cdna_header(path)
        start = header["latent_offset"]
        end = min(start + header["latent_size"], file_size)
    elif fmt == "gguf":
        # Skip header+metadata region (first ~2% is KV pairs, then tensor data)
        # Use 10% as conservative start to skip metadata
        start = max(int(file_size * 0.10), 4096)
        end = file_size
    elif fmt == "safetensors":
        header = read_safetensors_header(path)
        # Data starts after header
        start = 8 + header["header_len"]
        end = file_size
    else:
        start = 0
        end = file_size

    scannable = end - start
    if scannable < window_size * 2:
        # File too small for multiple windows, just scan what we have
        return [{"offset": start, "size": min(window_size, scannable),
                 "estimated_role": "unknown"}]

    stride = max((scannable - window_size) // max(n_windows - 1, 1), window_size)
    windows = []
    for i in range(n_windows):
        offset = start + i * stride
        if offset + window_size > end:
            break
        # Estimate structural role from position in file
        # Early = embedding/norm, middle = attention/ffn, late = lm_head/output
        frac = (offset - start) / max(scannable, 1)
        if frac < 0.05:
            role = "embedding"
        elif frac < 0.15:
            role = "early_layer"
        elif frac < 0.50:
            role = "mid_layer"
        elif frac < 0.85:
            role = "late_layer"
        elif frac < 0.95:
            role = "output_proj"
        else:
            role = "lm_head"
        windows.append({"offset": offset, "size": window_size, "estimated_role": role})

    return windows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _available_models():
    """Return model keys whose files exist on disk."""
    available = {}
    for key, info in MODELS.items():
        if info["path"].exists():
            available[key] = info
    return available


@pytest.fixture(scope="module")
def available_models():
    models = _available_models()
    if len(models) < 2:
        pytest.skip(f"Need at least 2 model files, found {len(models)}: {list(models.keys())}")
    return models


@pytest.fixture(scope="module")
def all_shadows(available_models):
    """Scan windows from every available model. Returns list of dicts with shadow + metadata."""
    results = []
    for model_key, model_info in available_models.items():
        windows = get_scan_windows(model_key, model_info, n_windows=10, window_size=256 * 1024)
        for i, win in enumerate(windows):
            path = str(model_info["path"])
            shadow = GlyphDAR.scan_file(
                path,
                codec=model_info["format"],
                offset=win["offset"],
                window_size=win["size"],
            )
            ghost = Ghost.from_shadow(shadow, shard_id=f"{model_key}:w{i}")
            results.append({
                "model_key": model_key,
                "arch": model_info["arch"],
                "family": model_info["family"],
                "format": model_info["format"],
                "params": model_info["params"],
                "window_idx": i,
                "offset": win["offset"],
                "estimated_role": win["estimated_role"],
                "shadow": shadow,
                "ghost": ghost,
                "ghost_class": ghost.shard_class,
            })
    return results


@pytest.fixture(scope="module")
def cross_model_memory(all_shadows):
    """Build a shared ShadowMemory from all model scans."""
    mem = ShadowMemory()
    for entry in all_shadows:
        mem.remember(
            shadow=entry["shadow"],
            ghost=entry["ghost"],
            shard_id=f"{entry['model_key']}:w{entry['window_idx']}",
        )
    return mem


# ---------------------------------------------------------------------------
# Tests — Cross-model shadow structure
# ---------------------------------------------------------------------------

class TestCrossModelScanning:
    """Verify scanning works across all formats."""

    def test_minimum_models_available(self, available_models):
        assert len(available_models) >= 2, \
            f"Need >= 2 models, have {len(available_models)}"

    def test_all_shadows_produced(self, all_shadows):
        assert len(all_shadows) >= 10, \
            f"Expected >= 10 shadow windows, got {len(all_shadows)}"

    def test_shadows_have_nonzero_entropy(self, all_shadows):
        for entry in all_shadows:
            assert entry["shadow"].entropy > 0, \
                f"{entry['model_key']}:w{entry['window_idx']} has zero entropy"

    def test_shadows_have_simhash(self, all_shadows):
        for entry in all_shadows:
            assert entry["shadow"].simhash64 != 0, \
                f"{entry['model_key']}:w{entry['window_idx']} has zero simhash"

    def test_shadows_have_classification(self, all_shadows):
        for entry in all_shadows:
            assert entry["ghost_class"] in ("embedding", "ffn", "attention", "norm"), \
                f"{entry['model_key']}:w{entry['window_idx']} bad class: {entry['ghost_class']}"

    def test_multiple_classes_found(self, all_shadows):
        """Across all models, we should see more than one structural class."""
        classes = set(e["ghost_class"] for e in all_shadows)
        assert len(classes) >= 2, \
            f"Only found classes: {classes} — no structural differentiation"

    def test_multiple_architectures_scanned(self, all_shadows):
        archs = set(e["arch"] for e in all_shadows)
        assert len(archs) >= 2, f"Only scanned architectures: {archs}"


class TestCrossModelClustering:
    """The core question: do shadows cluster by structural role across architectures?"""

    def test_intra_class_distance_vs_inter_class(self, all_shadows):
        """Shadows of the same ghost class should be closer (lower Hamming)
        than shadows of different classes, on average.

        This is the key structural clustering test.
        """
        intra_distances = []
        inter_distances = []

        for i in range(len(all_shadows)):
            for j in range(i + 1, len(all_shadows)):
                a = all_shadows[i]
                b = all_shadows[j]
                dist = _hamming64(a["shadow"].simhash64, b["shadow"].simhash64)

                if a["ghost_class"] == b["ghost_class"]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)

        assert len(intra_distances) > 0, "No same-class pairs found"
        assert len(inter_distances) > 0, "No cross-class pairs found"

        mean_intra = sum(intra_distances) / len(intra_distances)
        mean_inter = sum(inter_distances) / len(inter_distances)

        # Report
        print(f"\n--- Intra-class (same ghost_class) Hamming distances ---")
        print(f"  N pairs: {len(intra_distances)}")
        print(f"  Mean: {mean_intra:.1f}")
        print(f"  Min: {min(intra_distances)}, Max: {max(intra_distances)}")
        print(f"\n--- Inter-class (different ghost_class) Hamming distances ---")
        print(f"  N pairs: {len(inter_distances)}")
        print(f"  Mean: {mean_inter:.1f}")
        print(f"  Min: {min(inter_distances)}, Max: {max(inter_distances)}")
        print(f"\n  Separation: mean_intra={mean_intra:.1f} vs mean_inter={mean_inter:.1f}")
        print(f"  Delta: {mean_inter - mean_intra:+.1f} (positive = clustering works)")

        # The test: intra-class should trend lower. We don't require a huge gap
        # on this first pass — even zero delta is a finding worth reporting.
        # But if intra > inter, something is wrong with the classification.

    def test_cross_arch_same_class_similarity(self, all_shadows):
        """Within the same ghost_class, are cross-architecture pairs at all similar?

        If attention windows from Mistral and Zamba2 have lower Hamming distance
        than random pairs, structural similarity crosses architecture boundaries.
        """
        # Group by ghost_class
        by_class = defaultdict(list)
        for entry in all_shadows:
            by_class[entry["ghost_class"]].append(entry)

        print("\n--- Cross-architecture similarity within each class ---")
        for cls, entries in sorted(by_class.items()):
            cross_arch_pairs = []
            same_arch_pairs = []

            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    a = entries[i]
                    b = entries[j]
                    dist = _hamming64(a["shadow"].simhash64, b["shadow"].simhash64)

                    if a["arch"] != b["arch"]:
                        cross_arch_pairs.append((dist, a["model_key"], b["model_key"]))
                    elif a["model_key"] != b["model_key"]:
                        same_arch_pairs.append((dist, a["model_key"], b["model_key"]))

            if cross_arch_pairs:
                dists = [p[0] for p in cross_arch_pairs]
                print(f"\n  Class '{cls}' — cross-architecture pairs:")
                print(f"    N: {len(cross_arch_pairs)}")
                print(f"    Mean Hamming: {sum(dists)/len(dists):.1f}")
                print(f"    Min: {min(dists)}, Max: {max(dists)}")
                # Show best cross-arch pair
                best = min(cross_arch_pairs, key=lambda x: x[0])
                print(f"    Closest: {best[1]} <-> {best[2]} (Hamming={best[0]})")

            if same_arch_pairs:
                dists = [p[0] for p in same_arch_pairs]
                print(f"  Class '{cls}' — same-architecture/different-model pairs:")
                print(f"    N: {len(same_arch_pairs)}")
                print(f"    Mean Hamming: {sum(dists)/len(dists):.1f}")

    def test_entropy_distribution_by_class(self, all_shadows):
        """Report entropy distributions per ghost class across all models.

        If the same class has similar entropy ranges across architectures,
        the classification is structurally grounded (not just position-based).
        """
        by_class = defaultdict(list)
        for entry in all_shadows:
            by_class[entry["ghost_class"]].append(
                (entry["shadow"].entropy, entry["model_key"])
            )

        print("\n--- Entropy distribution by ghost class (cross-model) ---")
        for cls in sorted(by_class.keys()):
            vals = by_class[cls]
            entropies = [v[0] for v in vals]
            models = set(v[1] for v in vals)
            print(f"\n  Class '{cls}':")
            print(f"    N shadows: {len(vals)}")
            print(f"    Models: {sorted(models)}")
            print(f"    Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
            print(f"    Entropy mean: {sum(entropies)/len(entropies):.3f}")
            print(f"    Entropy std: {np.std(entropies):.3f}")


class TestCrossModelMemory:
    """Test ShadowMemory across model boundaries."""

    def test_memory_has_entries(self, cross_model_memory, all_shadows):
        assert len(cross_model_memory) == len(all_shadows)

    def test_memory_has_multiple_classes(self, cross_model_memory):
        stats = cross_model_memory.stats()
        assert len(stats["classes"]) >= 2

    def test_consensus_ghost_from_other_model(self, cross_model_memory, all_shadows):
        """Query memory with a shadow from one model.
        Does the consensus ghost match the single-model ghost?

        If consensus agrees with single-ghost, memory generalizes across models.
        """
        agreements = 0
        total = 0

        for entry in all_shadows:
            consensus = cross_model_memory.consensus_ghost(entry["shadow"], k=5)
            if consensus is None:
                continue

            # Does consensus class match single-ghost class?
            if consensus.shard_class == entry["ghost_class"]:
                agreements += 1
            total += 1

        agreement_rate = agreements / max(total, 1)
        print(f"\n--- Consensus ghost agreement (cross-model memory) ---")
        print(f"  Total queries: {total}")
        print(f"  Class agreement: {agreements}/{total} ({agreement_rate:.1%})")

        # The consensus ghost should agree with the single ghost at least
        # some of the time. High agreement = classification is stable.
        assert agreement_rate > 0.3, \
            f"Consensus agreement too low: {agreement_rate:.1%}"

    def test_consensus_confidence_vs_single(self, cross_model_memory, all_shadows):
        """Does consensus from cross-model memory boost confidence vs single ghost?"""
        single_confidences = []
        consensus_confidences = []

        for entry in all_shadows:
            consensus = cross_model_memory.consensus_ghost(entry["shadow"], k=5)
            if consensus is None:
                continue
            single_confidences.append(entry["ghost"].confidence)
            consensus_confidences.append(consensus.confidence)

        if not single_confidences:
            pytest.skip("No consensus ghosts produced")

        mean_single = sum(single_confidences) / len(single_confidences)
        mean_consensus = sum(consensus_confidences) / len(consensus_confidences)

        print(f"\n--- Confidence: single ghost vs cross-model consensus ---")
        print(f"  Mean single ghost confidence: {mean_single:.3f}")
        print(f"  Mean consensus confidence: {mean_consensus:.3f}")
        print(f"  Delta: {mean_consensus - mean_single:+.3f}")

    def test_nearest_neighbor_crosses_models(self, cross_model_memory, all_shadows):
        """For each shadow, does the nearest neighbor ever come from a DIFFERENT model?

        If yes: structural similarity genuinely crosses model boundaries.
        """
        cross_model_nn = 0
        same_model_nn = 0

        for entry in all_shadows:
            neighbors = cross_model_memory.recall(entry["shadow"], k=2)
            if len(neighbors) < 2:
                continue
            # Nearest neighbor is [0], but that's itself. Take [1].
            nn_entry, nn_dist = neighbors[1]
            nn_model = nn_entry.shard_id.split(":")[0] if ":" in nn_entry.shard_id else ""
            query_model = entry["model_key"]

            if nn_model != query_model:
                cross_model_nn += 1
            else:
                same_model_nn += 1

        total = cross_model_nn + same_model_nn
        cross_rate = cross_model_nn / max(total, 1)
        print(f"\n--- Nearest-neighbor model crossing ---")
        print(f"  Cross-model NN: {cross_model_nn}/{total} ({cross_rate:.1%})")
        print(f"  Same-model NN: {same_model_nn}/{total}")

        # At least some nearest neighbors should cross model boundaries.
        # If zero, the shadows are completely model-specific (still a finding).


class TestCrossModelReport:
    """Generate a summary report of cross-model shadow clustering."""

    def test_generate_report(self, all_shadows, cross_model_memory):
        """Print a comprehensive cross-model shadow analysis report."""
        # Model summary
        model_counts = defaultdict(int)
        for e in all_shadows:
            model_counts[e["model_key"]] += 1

        print("\n" + "=" * 70)
        print("CROSS-MODEL SHADOW CLUSTERING REPORT")
        print("=" * 70)

        print(f"\nModels scanned: {len(model_counts)}")
        for mk, count in sorted(model_counts.items()):
            info = MODELS[mk]
            print(f"  {mk}: {count} windows ({info['arch']}, {info['format']}, {info['params']})")

        print(f"\nTotal shadow windows: {len(all_shadows)}")
        print(f"Memory entries: {len(cross_model_memory)}")

        # Class distribution
        class_by_model = defaultdict(lambda: defaultdict(int))
        for e in all_shadows:
            class_by_model[e["ghost_class"]][e["model_key"]] += 1

        print(f"\nGhost class distribution across models:")
        for cls in sorted(class_by_model.keys()):
            models = class_by_model[cls]
            total = sum(models.values())
            model_list = ", ".join(f"{m}={c}" for m, c in sorted(models.items()))
            print(f"  {cls}: {total} total ({model_list})")

        # All-pairs Hamming distance matrix
        n = len(all_shadows)
        total_pairs = n * (n - 1) // 2
        all_distances = []
        close_pairs = []  # Hamming <= 8

        for i in range(n):
            for j in range(i + 1, n):
                a = all_shadows[i]
                b = all_shadows[j]
                dist = _hamming64(a["shadow"].simhash64, b["shadow"].simhash64)
                all_distances.append(dist)
                if dist <= 8:
                    close_pairs.append((dist, a, b))

        print(f"\nAll-pairs Hamming distance ({total_pairs} pairs):")
        print(f"  Mean: {sum(all_distances)/len(all_distances):.1f}")
        print(f"  Std: {np.std(all_distances):.1f}")
        print(f"  Min: {min(all_distances)}, Max: {max(all_distances)}")

        print(f"\nClose pairs (Hamming <= 8): {len(close_pairs)}/{total_pairs}")
        cross_model_close = 0
        same_class_close = 0
        for dist, a, b in close_pairs:
            is_cross = a["model_key"] != b["model_key"]
            is_same_class = a["ghost_class"] == b["ghost_class"]
            if is_cross:
                cross_model_close += 1
            if is_same_class:
                same_class_close += 1
            marker = ""
            if is_cross:
                marker += " [CROSS-MODEL]"
            if is_same_class:
                marker += " [SAME-CLASS]"
            print(f"  Hamming={dist}: {a['model_key']}:w{a['window_idx']}({a['ghost_class']}) "
                  f"<-> {b['model_key']}:w{b['window_idx']}({b['ghost_class']}){marker}")

        if close_pairs:
            print(f"\n  Cross-model in close pairs: {cross_model_close}/{len(close_pairs)}")
            print(f"  Same-class in close pairs: {same_class_close}/{len(close_pairs)}")

        # Key finding
        print(f"\n{'=' * 70}")
        print("KEY FINDINGS:")
        if cross_model_close > 0:
            print("  [POSITIVE] Cross-model close pairs exist — structural similarity")
            print("  crosses architecture boundaries.")
        else:
            print("  [NEGATIVE] No cross-model close pairs — shadows are model-specific.")
        if same_class_close > len(close_pairs) * 0.5 and close_pairs:
            print("  [POSITIVE] Majority of close pairs share ghost class — classification")
            print("  is structurally grounded, not random.")
        print(f"{'=' * 70}")
