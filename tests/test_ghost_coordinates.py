"""Phase 0.12: Ghost Coordinate Space — do tensor types form natural clusters?

Phase 0.11 proved:
  Route: 100% from transition graph alone
  Class: 22% with fixed thresholds (but 88% cluster dominance)

The advisor's insight: Ghost is a coordinate system, not a classifier.
If tensor types naturally cluster in (complexity, predictability, locality)
space, then labels are post-processing, not hard-coded thresholds.

Experiment:
  1. Compute ghost coordinates for EVERY scannable window across 5 models
  2. DO NOT classify. Just plot coordinates.
  3. Ask: do attention/FFN/embedding/norm form natural regions?
  4. Measure: silhouette score, inter-cluster distance, cluster purity

The three ghost dimensions:
  complexity     = mean(TE, TR)    [TE and TR collapse, r=0.90]
  predictability = MO               [independent from complexity]
  locality       = AC               [independent from both]

Ground truth labels come from the FULL Shadow (entropy-based classification).
Ghost coordinates use ONLY transition features. The test is whether the
coordinate space separates the classes that entropy labels assign.

WO-CRYSTAL-VAULT-01: Phase 0.12 — Ghost Coordinate Space
"""
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.vault_shard import GlyphDAR, Ghost


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------

MODELS = {
    "mistral_cdna": {
        "path": Path("/home/voidstr3m33/helix-cdc/artifacts/mistral_test.cdna"),
        "format": "cdna_v1",
    },
    "tinyllama_cdna": {
        "path": Path("/home/voidstr3m33/helix-cdc/artifacts/tinyllama-1.1b-fp32.cdna"),
        "format": "cdna_v1",
    },
    "mistral_gguf": {
        "path": Path("/home/voidstr3m33/rebuild_mmlu/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "format": "gguf",
    },
    "zamba2_gguf": {
        "path": Path("/home/voidstr3m33/cloud-work/ggufs/zamba2-2.7b-instruct-v2-q8_0.gguf"),
        "format": "gguf",
    },
    "mamba_hxq": {
        "path": Path("/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"),
        "format": "safetensors",
    },
}


# ---------------------------------------------------------------------------
# Scanning (from Phase 0.11)
# ---------------------------------------------------------------------------

def _get_scan_start(path: Path, fmt: str) -> int:
    file_size = os.path.getsize(path)
    if fmt == "cdna_v1":
        with open(path, "rb") as f:
            f.read(4)
            f.read(4)
            f.read(8)
            _, latent_offset = struct.unpack("<II", f.read(8))
        return latent_offset
    elif fmt == "gguf":
        return max(int(file_size * 0.10), 4096)
    elif fmt == "safetensors":
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
        return 8 + hlen
    return 0


def scan_dense(path: Path, fmt: str, n_windows: int = 20,
               window_size: int = 256 * 1024) -> list[tuple[bytes, int]]:
    """Scan many windows for dense coverage of the file."""
    file_size = os.path.getsize(path)
    start = _get_scan_start(path, fmt)
    end = file_size
    scannable = end - start

    if scannable < window_size:
        with open(path, "rb") as f:
            f.seek(start)
            return [(f.read(min(window_size, scannable)), start)]

    stride = max((scannable - window_size) // max(n_windows - 1, 1), window_size)
    windows = []
    with open(path, "rb") as f:
        for i in range(n_windows):
            offset = start + i * stride
            if offset + window_size > end:
                break
            f.seek(offset)
            windows.append((f.read(window_size), offset))
    return windows


# ---------------------------------------------------------------------------
# Ghost coordinates
# ---------------------------------------------------------------------------

def ghost_coordinates(raw_bytes: bytes) -> dict:
    """Compute the 3D ghost coordinate from raw encoded bytes.

    Returns: {complexity, predictability, locality}
    Plus the raw features for analysis.

    body_opened = false
    labels_used = none
    """
    n = len(raw_bytes)
    if n < 64:
        return {"complexity": 0.0, "predictability": 0.0, "locality": 0.0,
                "te": 0.0, "tr": 0.0, "mo": 0.0, "ac": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Bigram transition matrix
    bigram_counts = np.zeros((256, 256), dtype=np.int64)
    for i in range(n - 1):
        bigram_counts[arr[i], arr[i + 1]] += 1
    total_bigrams = n - 1

    # TE: transition entropy
    bigram_probs = bigram_counts[bigram_counts > 0] / total_bigrams
    bigram_h = -float(np.sum(bigram_probs * np.log2(bigram_probs)))
    max_bigram_h = 2.0 * np.log2(256)
    te = bigram_h / max_bigram_h if max_bigram_h > 0 else 0.0

    # TR: transition rank
    row_sums = bigram_counts.sum(axis=1)
    row_entropies = []
    for r in range(256):
        if row_sums[r] > 0:
            rp = bigram_counts[r][bigram_counts[r] > 0] / row_sums[r]
            row_entropies.append(-float(np.sum(rp * np.log2(rp))))
    mean_row_h = np.mean(row_entropies) if row_entropies else 0.0
    tr = mean_row_h / np.log2(256)

    # MO: markov order
    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 0.0

    # AC: index autocorrelation
    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    ac = float(np.mean((a - ma) * (b - mb)) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0

    # Collapse to 3D coordinates
    complexity = (te + tr) / 2.0       # TE and TR collapse (r=0.90)
    predictability = mo                 # Independent dimension
    locality = ac                       # Independent dimension

    return {
        "complexity": round(complexity, 6),
        "predictability": round(predictability, 6),
        "locality": round(locality, 6),
        "te": round(te, 6),
        "tr": round(tr, 6),
        "mo": round(mo, 6),
        "ac": round(ac, 6),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_points():
    """Collect ghost coordinates + ground truth labels for all windows."""
    points = []
    for model_key, info in MODELS.items():
        if not info["path"].exists():
            continue
        windows = scan_dense(info["path"], info["format"], n_windows=20)
        for raw_bytes, offset in windows:
            # Ghost coordinates (transition features only)
            coords = ghost_coordinates(raw_bytes)

            # Ground truth label from full shadow
            full_shadow = GlyphDAR.scan(raw_bytes, codec=info["format"])
            label = full_shadow.latent_shape.cluster if full_shadow.latent_shape else "unknown"

            points.append({
                "model": model_key,
                "format": info["format"],
                "offset": offset,
                "coords": coords,
                "label": label,
            })
    return points


# ===========================================================================
# Tests
# ===========================================================================


class TestDataAvailable:
    def test_sufficient_points(self, all_points):
        assert len(all_points) >= 30, f"Need >= 30 points, got {len(all_points)}"

    def test_multiple_labels(self, all_points):
        labels = set(p["label"] for p in all_points)
        assert len(labels) >= 2, f"Need >= 2 labels, got {labels}"

    def test_multiple_formats(self, all_points):
        formats = set(p["format"] for p in all_points)
        assert len(formats) >= 2, f"Need >= 2 formats, got {formats}"


class TestNaturalClustering:
    """Do tensor types form natural regions in ghost coordinate space?"""

    def test_class_separation_in_complexity(self, all_points):
        """Do different classes occupy different complexity ranges?"""
        by_label = defaultdict(list)
        for p in all_points:
            by_label[p["label"]].append(p["coords"]["complexity"])

        print(f"\n  Complexity by class:")
        for label in sorted(by_label.keys()):
            vals = by_label[label]
            print(f"    {label:<12} mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f} "
                  f"range=[{min(vals):.4f}, {max(vals):.4f}] "
                  f"n={len(vals)}")

        # Compute ANOVA-like statistic: between-class variance / within-class variance
        all_vals = [v for vs in by_label.values() for v in vs]
        grand_mean = np.mean(all_vals)

        between_var = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_label.values()) / len(all_vals)
        within_var = sum(np.var(vs) * len(vs)
                        for vs in by_label.values()) / len(all_vals)

        f_ratio = between_var / within_var if within_var > 1e-12 else 0.0
        print(f"\n  F-ratio (complexity): {f_ratio:.2f}")
        print(f"  (>1 means between-class > within-class variance)")

    def test_class_separation_in_predictability(self, all_points):
        """Do different classes occupy different predictability ranges?"""
        by_label = defaultdict(list)
        for p in all_points:
            by_label[p["label"]].append(p["coords"]["predictability"])

        print(f"\n  Predictability by class:")
        for label in sorted(by_label.keys()):
            vals = by_label[label]
            print(f"    {label:<12} mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f} "
                  f"range=[{min(vals):.4f}, {max(vals):.4f}] "
                  f"n={len(vals)}")

        all_vals = [v for vs in by_label.values() for v in vs]
        grand_mean = np.mean(all_vals)
        between_var = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_label.values()) / len(all_vals)
        within_var = sum(np.var(vs) * len(vs)
                        for vs in by_label.values()) / len(all_vals)
        f_ratio = between_var / within_var if within_var > 1e-12 else 0.0
        print(f"\n  F-ratio (predictability): {f_ratio:.2f}")

    def test_class_separation_in_locality(self, all_points):
        """Do different classes occupy different locality ranges?"""
        by_label = defaultdict(list)
        for p in all_points:
            by_label[p["label"]].append(p["coords"]["locality"])

        print(f"\n  Locality by class:")
        for label in sorted(by_label.keys()):
            vals = by_label[label]
            print(f"    {label:<12} mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f} "
                  f"range=[{min(vals):.4f}, {max(vals):.4f}] "
                  f"n={len(vals)}")

        all_vals = [v for vs in by_label.values() for v in vs]
        grand_mean = np.mean(all_vals)
        between_var = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_label.values()) / len(all_vals)
        within_var = sum(np.var(vs) * len(vs)
                        for vs in by_label.values()) / len(all_vals)
        f_ratio = between_var / within_var if within_var > 1e-12 else 0.0
        print(f"\n  F-ratio (locality): {f_ratio:.2f}")


class TestFormatRegions:
    """Do different formats occupy different regions?"""

    def test_format_separation_in_complexity(self, all_points):
        """CDNA vs GGUF vs safetensors in complexity space."""
        by_fmt = defaultdict(list)
        for p in all_points:
            by_fmt[p["format"]].append(p["coords"]["complexity"])

        print(f"\n  Complexity by format:")
        for fmt in sorted(by_fmt.keys()):
            vals = by_fmt[fmt]
            print(f"    {fmt:<14} mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f} "
                  f"range=[{min(vals):.4f}, {max(vals):.4f}] "
                  f"n={len(vals)}")

    def test_format_separation_in_predictability(self, all_points):
        """CDNA vs GGUF vs safetensors in predictability space."""
        by_fmt = defaultdict(list)
        for p in all_points:
            by_fmt[p["format"]].append(p["coords"]["predictability"])

        print(f"\n  Predictability by format:")
        for fmt in sorted(by_fmt.keys()):
            vals = by_fmt[fmt]
            print(f"    {fmt:<14} mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f} "
                  f"range=[{min(vals):.4f}, {max(vals):.4f}] "
                  f"n={len(vals)}")

    def test_formats_overlap_or_separate(self, all_points):
        """Key question: are formats separated or overlapping in 3D?"""
        by_fmt = defaultdict(list)
        for p in all_points:
            c = p["coords"]
            by_fmt[p["format"]].append([c["complexity"], c["predictability"], c["locality"]])

        print(f"\n  Format centroids in (complexity, predictability, locality):")
        centroids = {}
        for fmt in sorted(by_fmt.keys()):
            arr = np.array(by_fmt[fmt])
            centroid = arr.mean(axis=0)
            centroids[fmt] = centroid
            print(f"    {fmt:<14} ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")

        # Compute pairwise centroid distances
        fmts = sorted(centroids.keys())
        print(f"\n  Pairwise centroid distances:")
        for i in range(len(fmts)):
            for j in range(i + 1, len(fmts)):
                d = np.linalg.norm(centroids[fmts[i]] - centroids[fmts[j]])
                print(f"    {fmts[i]} ↔ {fmts[j]}: {d:.4f}")


class TestSilhouette:
    """Silhouette score: does the coordinate space naturally separate classes?"""

    def test_silhouette_by_label(self, all_points):
        """Compute silhouette score using class labels."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_points])
        labels = [p["label"] for p in all_points]
        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            pytest.skip("Need >= 2 labels for silhouette")

        # Simple silhouette: for each point, (b - a) / max(a, b)
        # a = mean distance to same-label points
        # b = mean distance to nearest different-label cluster
        label_indices = defaultdict(list)
        for i, l in enumerate(labels):
            label_indices[l].append(i)

        silhouettes = []
        for i in range(len(coords)):
            my_label = labels[i]

            # a: mean distance to same cluster
            same = [j for j in label_indices[my_label] if j != i]
            if same:
                a = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in same])
            else:
                a = 0.0

            # b: mean distance to nearest other cluster
            b = float("inf")
            for other_label in unique_labels:
                if other_label == my_label:
                    continue
                other = label_indices[other_label]
                if other:
                    mean_dist = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in other])
                    b = min(b, mean_dist)

            if b == float("inf"):
                b = 0.0

            s = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
            silhouettes.append(s)

        mean_sil = np.mean(silhouettes)
        print(f"\n  Silhouette score (by class label): {mean_sil:.4f}")
        print(f"  Interpretation:")
        print(f"    > 0.5: strong natural clustering")
        print(f"    > 0.25: weak but real clustering")
        print(f"    ~ 0.0: no structure")
        print(f"    < 0.0: wrong labels")

        # Per-label silhouette
        for label in unique_labels:
            indices = label_indices[label]
            label_sil = np.mean([silhouettes[i] for i in indices])
            print(f"    {label:<12} sil={label_sil:.4f} (n={len(indices)})")

    def test_silhouette_by_format(self, all_points):
        """Compute silhouette score using format as label."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_points])
        labels = [p["format"] for p in all_points]
        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            pytest.skip("Need >= 2 formats for silhouette")

        label_indices = defaultdict(list)
        for i, l in enumerate(labels):
            label_indices[l].append(i)

        silhouettes = []
        for i in range(len(coords)):
            my_label = labels[i]
            same = [j for j in label_indices[my_label] if j != i]
            if same:
                a = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in same])
            else:
                a = 0.0

            b = float("inf")
            for other_label in unique_labels:
                if other_label == my_label:
                    continue
                other = label_indices[other_label]
                if other:
                    mean_dist = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in other])
                    b = min(b, mean_dist)
            if b == float("inf"):
                b = 0.0

            s = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
            silhouettes.append(s)

        mean_sil = np.mean(silhouettes)
        print(f"\n  Silhouette score (by format): {mean_sil:.4f}")
        for label in unique_labels:
            indices = label_indices[label]
            label_sil = np.mean([silhouettes[i] for i in indices])
            print(f"    {label:<14} sil={label_sil:.4f} (n={len(indices)})")


class TestCoordinateReport:
    """Full Phase 0.12 report."""

    def test_full_report(self, all_points):
        n = len(all_points)
        models = set(p["model"] for p in all_points)
        formats = set(p["format"] for p in all_points)
        labels = set(p["label"] for p in all_points)

        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_points])

        # Compute silhouette for both groupings
        def silhouette(grouping):
            label_indices = defaultdict(list)
            for i, p in enumerate(all_points):
                label_indices[grouping(p)].append(i)
            unique = sorted(label_indices.keys())
            if len(unique) < 2:
                return 0.0
            sils = []
            for i in range(len(coords)):
                my_label = grouping(all_points[i])
                same = [j for j in label_indices[my_label] if j != i]
                a = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in same]) if same else 0.0
                b = float("inf")
                for ol in unique:
                    if ol == my_label:
                        continue
                    other = label_indices[ol]
                    if other:
                        b = min(b, np.mean([np.linalg.norm(coords[i] - coords[j]) for j in other]))
                if b == float("inf"):
                    b = 0.0
                sils.append((b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0)
            return float(np.mean(sils))

        sil_class = silhouette(lambda p: p["label"])
        sil_format = silhouette(lambda p: p["format"])

        # Per-dimension F-ratios for class labels
        f_ratios = []
        for dim, name in [(0, "complexity"), (1, "predictability"), (2, "locality")]:
            by_label = defaultdict(list)
            for p in all_points:
                by_label[p["label"]].append(coords[all_points.index(p), dim])
            # Recompute cleanly
            by_label = defaultdict(list)
            for i, p in enumerate(all_points):
                by_label[p["label"]].append(coords[i, dim])

            all_vals = [v for vs in by_label.values() for v in vs]
            grand_mean = np.mean(all_vals)
            between = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_label.values()) / len(all_vals)
            within = sum(np.var(vs) * len(vs)
                        for vs in by_label.values()) / len(all_vals)
            f = between / within if within > 1e-12 else 0.0
            f_ratios.append((name, f))

        print(f"""
================================================================================
PHASE 0.12: GHOST COORDINATE SPACE
================================================================================

Points:  {n}
Models:  {len(models)} ({', '.join(sorted(models))})
Formats: {len(formats)} ({', '.join(sorted(formats))})
Labels:  {len(labels)} ({', '.join(sorted(labels))})

labels_used_for_coordinates: NONE
body_opened: false
transition_features_only: true

--- Coordinate Ranges ---
  complexity:     [{coords[:,0].min():.4f}, {coords[:,0].max():.4f}] span={coords[:,0].max()-coords[:,0].min():.4f}
  predictability: [{coords[:,1].min():.4f}, {coords[:,1].max():.4f}] span={coords[:,1].max()-coords[:,1].min():.4f}
  locality:       [{coords[:,2].min():.4f}, {coords[:,2].max():.4f}] span={coords[:,2].max()-coords[:,2].min():.4f}

--- Class Separation (F-ratio: between/within variance, >1 = separable) ---""")
        for name, f in f_ratios:
            marker = "SEPARATES" if f > 1.0 else ("weak" if f > 0.3 else "NO")
            print(f"  {name:<16} F={f:.2f}  {marker}")

        print(f"""
--- Silhouette Scores ---
  By class label:  {sil_class:.4f}  {'CLUSTERS' if sil_class > 0.25 else 'WEAK' if sil_class > 0.0 else 'NONE'}
  By format:       {sil_format:.4f}  {'CLUSTERS' if sil_format > 0.25 else 'WEAK' if sil_format > 0.0 else 'NONE'}

================================================================================
VERDICT:
""")
        if sil_class > 0.25:
            print("  Classes form NATURAL CLUSTERS in ghost coordinate space.")
            print("  Labels are post-processing — the geometry exists before naming.")
            print("  Ghost = latent coordinate system, not classifier.")
        elif sil_class > 0.0:
            print("  Weak but real class structure in ghost coordinates.")
            print("  Some dimensions separate classes, others don't.")
            best_dim = max(f_ratios, key=lambda x: x[1])
            print(f"  Best separating dimension: {best_dim[0]} (F={best_dim[1]:.2f})")
        else:
            print("  No natural class clustering in ghost coordinate space.")
            print("  Labels do NOT emerge from transition structure alone.")

        if sil_format > 0.25:
            print(f"\n  Formats also cluster (sil={sil_format:.3f}).")
            print("  Ghost coordinates are partially format-dependent.")
            print("  This explains why fixed thresholds fail cross-format.")
        elif sil_format > 0.0:
            print(f"\n  Formats weakly separate (sil={sil_format:.3f}).")
        else:
            print(f"\n  Formats do NOT separate (sil={sil_format:.3f}).")
            print("  Ghost coordinates are format-independent.")

        print("================================================================================")
