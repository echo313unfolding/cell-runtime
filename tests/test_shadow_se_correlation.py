"""Phase 0.9: Can Shadow estimate Se components without opening the body?

The core question: does a shadow (raw encoded bytes, no decompression)
preserve enough structure to estimate H, U, D from the Se formula?

Ground truth: H, U, D computed from actual dequantized float tensors.
Shadow estimate: H, U, D estimated from raw U8 codebook indices only.

If corr(shadow_X, true_X) is high, the shadow carries that axis.
If corr is low, that axis requires touching the body.

Data: Mamba-130M HXQ safetensors (96 quantized tensors, U8 indices + F32 codebooks).

Se = H x U x D where:
  H = spectral spread (1 - energy_at_10pct from SVD)
  U = unstructuredness (1 - neighbor_coherence)
  D = effective rank depth (sqrt(rank_ratio))

Source: helix-cdc/tools/tensor_se_estimator.py (proven 2026-01-26)

WO-CRYSTAL-VAULT-01: Phase 0.9 — Shadow Se correlation proof
"""
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.vault_shard import GlyphDAR, Shadow

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MAMBA_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
    "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
)


# ---------------------------------------------------------------------------
# Safetensors reader
# ---------------------------------------------------------------------------

def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header, 8 + hlen  # header, data_start_offset


def load_tensor_numpy(path: Path, data_start: int, info: dict) -> np.ndarray:
    """Load a single tensor from safetensors as numpy array."""
    dtype_map = {"U8": np.uint8, "F32": np.float32, "I64": np.int64, "F16": np.float16}
    start, end = info["data_offsets"]
    dtype = dtype_map[info["dtype"]]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    arr = np.frombuffer(raw, dtype=dtype)
    if info["shape"]:
        arr = arr.reshape(info["shape"])
    return arr


def read_raw_bytes(path: Path, data_start: int, info: dict) -> bytes:
    """Read raw bytes of a tensor region. No interpretation. Shadow only."""
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        return f.read(end - start)


# ---------------------------------------------------------------------------
# Ground truth Se components (from tensor_se_estimator.py logic)
# ---------------------------------------------------------------------------

def true_se_components(tensor_2d: np.ndarray) -> dict:
    """Compute true H, U, D from an actual float tensor.

    Matches tensor_se_estimator.py logic exactly.
    """
    if tensor_2d.ndim == 1:
        tensor_2d = tensor_2d.reshape(1, -1)
    m, n = tensor_2d.shape
    k = min(m, n, 64)  # Cap SVD rank for speed

    # --- H: spectral spread ---
    try:
        U, S, Vt = np.linalg.svd(tensor_2d.astype(np.float64), full_matrices=False)
        S = S[:k]
        total_energy = float(np.sum(S ** 2))
        if total_energy > 0:
            cumulative = np.cumsum(S ** 2) / total_energy
            top_10pct_idx = max(1, int(0.1 * len(S)))
            energy_at_10pct = float(cumulative[top_10pct_idx - 1])
        else:
            energy_at_10pct = 1.0
    except np.linalg.LinAlgError:
        energy_at_10pct = 1.0

    H = 1.0 - energy_at_10pct

    # --- U: unstructuredness (neighbor coherence) ---
    # Row coherence: correlation between adjacent rows
    if m > 1:
        row_corrs = []
        for i in range(min(m - 1, 100)):  # Sample up to 100 pairs
            r1 = tensor_2d[i].astype(np.float64)
            r2 = tensor_2d[i + 1].astype(np.float64)
            s1, s2 = np.std(r1), np.std(r2)
            if s1 > 1e-12 and s2 > 1e-12:
                c = float(np.corrcoef(r1, r2)[0, 1])
                if not np.isnan(c):
                    row_corrs.append(abs(c))
        row_coherence = float(np.mean(row_corrs)) if row_corrs else 0.0
    else:
        row_coherence = 0.0

    # Column coherence: correlation between adjacent columns
    if n > 1:
        col_corrs = []
        for j in range(min(n - 1, 100)):
            c1 = tensor_2d[:, j].astype(np.float64)
            c2 = tensor_2d[:, j + 1].astype(np.float64)
            s1, s2 = np.std(c1), np.std(c2)
            if s1 > 1e-12 and s2 > 1e-12:
                c = float(np.corrcoef(c1, c2)[0, 1])
                if not np.isnan(c):
                    col_corrs.append(abs(c))
        col_coherence = float(np.mean(col_corrs)) if col_corrs else 0.0
    else:
        col_coherence = 0.0

    neighbor_coherence = max(row_coherence, col_coherence)
    U = 1.0 - neighbor_coherence

    # --- D: effective rank depth ---
    if total_energy > 0:
        normalized = (S ** 2) / total_energy
        normalized = normalized[normalized > 1e-12]
        spectral_entropy = -float(np.sum(normalized * np.log(normalized)))
        effective_rank = float(np.exp(spectral_entropy))
        rank_ratio = effective_rank / max(min(m, n), 1)
    else:
        rank_ratio = 0.0

    D = float(np.sqrt(min(rank_ratio, 1.0)))

    Se = H * U * D

    return {
        "H": round(H, 6),
        "U": round(U, 6),
        "D": round(D, 6),
        "Se": round(Se, 6),
        "energy_at_10pct": round(energy_at_10pct, 6),
        "row_coherence": round(row_coherence, 6),
        "col_coherence": round(col_coherence, 6),
        "neighbor_coherence": round(neighbor_coherence, 6),
        "rank_ratio": round(rank_ratio, 6),
    }


# ---------------------------------------------------------------------------
# Shadow Se estimators (raw bytes only, NO decompression)
# ---------------------------------------------------------------------------

def shadow_se_components(raw_bytes: bytes, shape: tuple) -> dict:
    """Estimate H, U, D from raw encoded bytes only.

    NO decompression. NO codebook lookup. NO weight materialization.
    body_opened_for_routing = False.

    Shadow-H: Shannon entropy of byte stream (proxy for spectral spread)
    Shadow-U: 1 - adjacent byte autocorrelation (proxy for neighbor coherence)
    Shadow-D: histogram diversity × meta-entropy (proxy for effective rank)
    """
    n = len(raw_bytes)
    if n < 64:
        return {"H": 0.0, "U": 0.0, "D": 0.0, "Se": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # --- Shadow-H: Shannon entropy of raw bytes ---
    # Maps to: spectral spread (complex spectrum = high byte entropy)
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / n
    shannon = -float(np.sum(probs * np.log2(probs)))
    # Normalize to [0, 1] (max = 8 bits for uint8)
    shadow_H = shannon / 8.0

    # --- Shadow-U: byte-level unstructuredness ---
    # Adjacent byte autocorrelation as proxy for neighbor coherence.
    # If codebook indices for adjacent elements are correlated,
    # the underlying tensor has spatial structure.
    #
    # We compute autocorrelation at two strides:
    #   stride=1 (adjacent bytes = within-row coherence for row-major layout)
    #   stride=cols (row-to-row coherence, if shape is known)
    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    mean_a, mean_b = np.mean(a), np.mean(b)
    std_a, std_b = np.std(a), np.std(b)
    if std_a > 1e-12 and std_b > 1e-12:
        col_autocorr = float(np.mean((a - mean_a) * (b - mean_b)) / (std_a * std_b))
    else:
        col_autocorr = 0.0

    row_autocorr = 0.0
    if len(shape) == 2 and shape[1] > 1:
        stride = shape[1]
        if n > stride:
            a = arr[:-stride].astype(np.float64)
            b = arr[stride:].astype(np.float64)
            mean_a, mean_b = np.mean(a), np.mean(b)
            std_a, std_b = np.std(a), np.std(b)
            if std_a > 1e-12 and std_b > 1e-12:
                row_autocorr = float(np.mean((a - mean_a) * (b - mean_b)) / (std_a * std_b))

    shadow_coherence = max(abs(col_autocorr), abs(row_autocorr))
    shadow_U = 1.0 - shadow_coherence

    # --- Shadow-D: diversity as rank proxy ---
    # High effective rank = complex structure = diverse byte patterns.
    # Two signals:
    #   1. Histogram flatness: how evenly distributed are byte values?
    #   2. Meta-entropy: entropy of per-block entropies (structural variation)
    #
    # Histogram flatness: ratio of nonzero bins to 256
    nonzero_bins = int(np.sum(counts > 0))
    bin_diversity = nonzero_bins / 256.0

    # Meta-entropy: entropy of per-block entropy profile
    block_size = min(4096, n // 4) if n > 256 else n
    block_entropies = []
    for i in range(0, n - block_size + 1, block_size):
        block = arr[i:i + block_size]
        bc = np.bincount(block, minlength=256).astype(np.float64)
        p = bc[bc > 0] / len(block)
        block_entropies.append(-float(np.sum(p * np.log2(p))))

    if len(block_entropies) >= 2:
        be = np.array(block_entropies)
        be_std = float(np.std(be))
        be_range = float(np.max(be) - np.min(be))
        # Higher variation in block entropies = more structural complexity = higher rank
        meta_entropy = min(be_std / 2.0, 1.0)  # Normalize roughly to [0, 1]
    else:
        meta_entropy = 0.5  # Default for tiny tensors

    # Combine: sqrt of product (geometric mean style, like true D uses sqrt)
    shadow_D = float(np.sqrt(bin_diversity * max(meta_entropy, 0.01)))

    shadow_Se = shadow_H * shadow_U * shadow_D

    return {
        "H": round(shadow_H, 6),
        "U": round(shadow_U, 6),
        "D": round(shadow_D, 6),
        "Se": round(shadow_Se, 6),
        "shannon_bits": round(shannon, 4),
        "col_autocorr": round(col_autocorr, 6),
        "row_autocorr": round(row_autocorr, 6),
        "shadow_coherence": round(shadow_coherence, 6),
        "bin_diversity": round(bin_diversity, 4),
        "meta_entropy": round(meta_entropy, 6),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hxq_tensors():
    """Load all HXQ index tensors with their codebooks."""
    if not MAMBA_HXQ.exists():
        pytest.skip(f"Mamba HXQ not found: {MAMBA_HXQ}")

    header, data_start = read_safetensors_header(MAMBA_HXQ)
    tensors_info = {k: v for k, v in header.items() if k != "__metadata__"}

    # Find index+codebook pairs
    pairs = []
    for name, info in tensors_info.items():
        if not name.endswith(".indices"):
            continue
        base = name.replace(".indices", "")
        cb_name = f"{base}.codebook"
        if cb_name not in tensors_info:
            continue

        # Skip very small tensors (< 1KB)
        byte_size = info["data_offsets"][1] - info["data_offsets"][0]
        if byte_size < 1024:
            continue

        pairs.append({
            "name": base,
            "indices_info": info,
            "codebook_info": tensors_info[cb_name],
        })

    return pairs, data_start


@pytest.fixture(scope="module")
def correlation_data(hxq_tensors):
    """Compute true and shadow Se components for all tensors."""
    pairs, data_start = hxq_tensors

    results = []
    for pair in pairs:
        name = pair["name"]
        idx_info = pair["indices_info"]
        cb_info = pair["codebook_info"]
        shape = tuple(idx_info["shape"])

        # Load indices and codebook (for ground truth only)
        indices = load_tensor_numpy(MAMBA_HXQ, data_start, idx_info)
        codebook = load_tensor_numpy(MAMBA_HXQ, data_start, cb_info)

        # Dequantize: float_tensor = codebook[indices]
        float_tensor = codebook[indices.astype(np.int32)]

        # Ground truth Se components
        true = true_se_components(float_tensor)

        # Shadow: raw bytes only
        raw_bytes = read_raw_bytes(MAMBA_HXQ, data_start, idx_info)
        shadow = shadow_se_components(raw_bytes, shape)

        results.append({
            "name": name,
            "shape": shape,
            "bytes": len(raw_bytes),
            "true": true,
            "shadow": shadow,
        })

    return results


def pearson_r(x, y):
    """Pearson correlation coefficient."""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.mean((x - mx) * (y - my)) / (sx * sy))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataAvailable:

    def test_tensors_found(self, hxq_tensors):
        pairs, _ = hxq_tensors
        assert len(pairs) >= 10, f"Expected >= 10 HXQ tensor pairs, got {len(pairs)}"

    def test_correlation_data_computed(self, correlation_data):
        assert len(correlation_data) >= 10


class TestShadowSeCorrelation:
    """The core experiment: correlate shadow estimates with ground truth."""

    def test_h_correlation(self, correlation_data):
        """Shadow-H (byte entropy) vs True-H (spectral spread).

        Shadow-H = Shannon entropy of raw bytes / 8.
        True-H = 1 - energy_at_10pct (SVD spectral concentration).
        """
        true_H = [r["true"]["H"] for r in correlation_data]
        shadow_H = [r["shadow"]["H"] for r in correlation_data]

        r = pearson_r(true_H, shadow_H)
        print(f"\n--- H Correlation (spectral spread) ---")
        print(f"  N tensors: {len(true_H)}")
        print(f"  True H:   mean={np.mean(true_H):.4f}, std={np.std(true_H):.4f}, "
              f"range=[{min(true_H):.4f}, {max(true_H):.4f}]")
        print(f"  Shadow H: mean={np.mean(shadow_H):.4f}, std={np.std(shadow_H):.4f}, "
              f"range=[{min(shadow_H):.4f}, {max(shadow_H):.4f}]")
        print(f"  Pearson r = {r:.4f}")

    def test_u_correlation(self, correlation_data):
        """Shadow-U (byte autocorrelation) vs True-U (neighbor coherence).

        Shadow-U = 1 - max(col_autocorr, row_autocorr) of byte stream.
        True-U = 1 - max(row_coherence, col_coherence) of float tensor.
        """
        true_U = [r["true"]["U"] for r in correlation_data]
        shadow_U = [r["shadow"]["U"] for r in correlation_data]

        r = pearson_r(true_U, shadow_U)
        print(f"\n--- U Correlation (unstructuredness) ---")
        print(f"  N tensors: {len(true_U)}")
        print(f"  True U:   mean={np.mean(true_U):.4f}, std={np.std(true_U):.4f}, "
              f"range=[{min(true_U):.4f}, {max(true_U):.4f}]")
        print(f"  Shadow U: mean={np.mean(shadow_U):.4f}, std={np.std(shadow_U):.4f}, "
              f"range=[{min(shadow_U):.4f}, {max(shadow_U):.4f}]")
        print(f"  Pearson r = {r:.4f}")

        # Also report raw coherence values
        true_coh = [r["true"]["neighbor_coherence"] for r in correlation_data]
        shadow_coh = [r["shadow"]["shadow_coherence"] for r in correlation_data]
        r_coh = pearson_r(true_coh, shadow_coh)
        print(f"\n  Raw coherence correlation:")
        print(f"  True coherence:   mean={np.mean(true_coh):.4f}, range=[{min(true_coh):.4f}, {max(true_coh):.4f}]")
        print(f"  Shadow coherence: mean={np.mean(shadow_coh):.4f}, range=[{min(shadow_coh):.4f}, {max(shadow_coh):.4f}]")
        print(f"  Pearson r (coherence) = {r_coh:.4f}")

    def test_d_correlation(self, correlation_data):
        """Shadow-D (byte diversity) vs True-D (effective rank).

        Shadow-D = sqrt(bin_diversity * meta_entropy).
        True-D = sqrt(rank_ratio).
        """
        true_D = [r["true"]["D"] for r in correlation_data]
        shadow_D = [r["shadow"]["D"] for r in correlation_data]

        r = pearson_r(true_D, shadow_D)
        print(f"\n--- D Correlation (rank depth) ---")
        print(f"  N tensors: {len(true_D)}")
        print(f"  True D:   mean={np.mean(true_D):.4f}, std={np.std(true_D):.4f}, "
              f"range=[{min(true_D):.4f}, {max(true_D):.4f}]")
        print(f"  Shadow D: mean={np.mean(shadow_D):.4f}, std={np.std(shadow_D):.4f}, "
              f"range=[{min(shadow_D):.4f}, {max(shadow_D):.4f}]")
        print(f"  Pearson r = {r:.4f}")

    def test_se_correlation(self, correlation_data):
        """Shadow-Se (H*U*D) vs True-Se (H*U*D).

        The combined metric. If this correlates, shadow-routing
        approximates full Se-routing.
        """
        true_Se = [r["true"]["Se"] for r in correlation_data]
        shadow_Se = [r["shadow"]["Se"] for r in correlation_data]

        r = pearson_r(true_Se, shadow_Se)
        print(f"\n--- Se Correlation (combined H*U*D) ---")
        print(f"  N tensors: {len(true_Se)}")
        print(f"  True Se:   mean={np.mean(true_Se):.4f}, std={np.std(true_Se):.4f}")
        print(f"  Shadow Se: mean={np.mean(shadow_Se):.4f}, std={np.std(shadow_Se):.4f}")
        print(f"  Pearson r = {r:.4f}")


class TestShadowSeReport:
    """Summary report with all correlations and per-tensor breakdown."""

    def test_full_report(self, correlation_data):
        """Print the complete Phase 0.9 correlation report."""
        true_H = [r["true"]["H"] for r in correlation_data]
        true_U = [r["true"]["U"] for r in correlation_data]
        true_D = [r["true"]["D"] for r in correlation_data]
        true_Se = [r["true"]["Se"] for r in correlation_data]

        shadow_H = [r["shadow"]["H"] for r in correlation_data]
        shadow_U = [r["shadow"]["U"] for r in correlation_data]
        shadow_D = [r["shadow"]["D"] for r in correlation_data]
        shadow_Se = [r["shadow"]["Se"] for r in correlation_data]

        r_H = pearson_r(true_H, shadow_H)
        r_U = pearson_r(true_U, shadow_U)
        r_D = pearson_r(true_D, shadow_D)
        r_Se = pearson_r(true_Se, shadow_Se)

        print("\n" + "=" * 70)
        print("PHASE 0.9: SHADOW Se CORRELATION REPORT")
        print("=" * 70)
        print(f"\nModel: Mamba-130M HXQ (96 quantized tensors)")
        print(f"Tensors analyzed: {len(correlation_data)}")
        print(f"body_opened_for_routing = False (shadow reads raw U8 indices only)")

        print(f"\n{'Axis':<8} {'corr(shadow, true)':<22} {'Shadow range':<20} {'True range':<20}")
        print("-" * 70)
        print(f"{'H':<8} {r_H:>+.4f}{'':16} [{min(shadow_H):.3f}, {max(shadow_H):.3f}]   "
              f"[{min(true_H):.3f}, {max(true_H):.3f}]")
        print(f"{'U':<8} {r_U:>+.4f}{'':16} [{min(shadow_U):.3f}, {max(shadow_U):.3f}]   "
              f"[{min(true_U):.3f}, {max(true_U):.3f}]")
        print(f"{'D':<8} {r_D:>+.4f}{'':16} [{min(shadow_D):.3f}, {max(shadow_D):.3f}]   "
              f"[{min(true_D):.3f}, {max(true_D):.3f}]")
        print(f"{'Se':<8} {r_Se:>+.4f}{'':16} [{min(shadow_Se):.3f}, {max(shadow_Se):.3f}]   "
              f"[{min(true_Se):.3f}, {max(true_Se):.3f}]")

        # Verdict
        print(f"\n{'=' * 70}")
        print("VERDICT:")
        for name, r in [("H", r_H), ("U", r_U), ("D", r_D), ("Se", r_Se)]:
            if abs(r) >= 0.7:
                verdict = "SHADOW CARRIES THIS AXIS"
            elif abs(r) >= 0.4:
                verdict = "PARTIAL — shadow approximates"
            elif abs(r) >= 0.2:
                verdict = "WEAK — shadow hints but insufficient"
            else:
                verdict = "DEAD — requires body access"
            print(f"  {name}: r={r:+.4f} → {verdict}")

        # The boundary question
        body_required = []
        shadow_sufficient = []
        for name, r in [("H", r_H), ("U", r_U), ("D", r_D)]:
            if abs(r) >= 0.5:
                shadow_sufficient.append(name)
            else:
                body_required.append(name)

        print(f"\n  Shadow-sufficient axes: {shadow_sufficient or ['NONE']}")
        print(f"  Body-required axes:    {body_required or ['NONE']}")
        print(f"{'=' * 70}")

        # Per-tensor top-5 and bottom-5 by Se error
        errors = []
        for r in correlation_data:
            se_err = abs(r["shadow"]["Se"] - r["true"]["Se"])
            errors.append((se_err, r["name"], r["true"]["Se"], r["shadow"]["Se"]))
        errors.sort()

        print(f"\nBest shadow Se estimates (lowest error):")
        for err, name, true, shadow in errors[:5]:
            print(f"  {name}: true={true:.4f}, shadow={shadow:.4f}, err={err:.4f}")

        print(f"\nWorst shadow Se estimates (highest error):")
        for err, name, true, shadow in errors[-5:]:
            print(f"  {name}: true={true:.4f}, shadow={shadow:.4f}, err={err:.4f}")
