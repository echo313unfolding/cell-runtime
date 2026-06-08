"""Phase 0.10: Projection Atlas — which shadow projections preserve H, U, D?

Phase 0.9 proved:
  H (byte entropy): r=-0.13 DEAD
  U (autocorrelation): r=+0.81 CARRIES
  D (histogram diversity): r=+0.46 PARTIAL

This phase tests whether TRANSITION-BASED projections improve the map.
The shadow is not a fingerprint. It's a projection operator.
Different projections preserve different properties.

Projections tested (all from raw U8 indices, NO decompression):
  1. byte_entropy       — Shannon entropy of byte stream (Phase 0.9 baseline)
  2. index_autocorr     — adjacent byte autocorrelation (Phase 0.9 baseline)
  3. transition_entropy — entropy of consecutive byte pairs (bigrams)
  4. transition_rank    — effective rank proxy of transition matrix
  5. run_length_mean    — mean run length of consecutive identical bytes
  6. run_length_max     — max run length
  7. local_entropy_var  — variance of per-block entropies
  8. histogram_kurtosis — excess kurtosis of byte histogram
  9. byte_skew          — skewness of byte histogram
  10. markov_order      — ratio of bigram entropy to unigram entropy

Pass condition:
  - Improve U beyond 0.81 OR
  - Improve D beyond 0.46 OR
  - Prove neither improves and log that honestly.

Receipt:
  body_opened_for_projection: false
  materialized_weight_bytes_for_projection: 0
  projection_source: raw_u8_indices

Data: Mamba-130M HXQ (96 tensors, same as Phase 0.9).

WO-CRYSTAL-VAULT-01: Phase 0.10 — Projection Atlas
"""
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Paths + helpers (shared with Phase 0.9)
# ---------------------------------------------------------------------------

MAMBA_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
    "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
)


def read_safetensors_header(path: Path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def load_tensor_numpy(path, data_start, info):
    dtype_map = {"U8": np.uint8, "F32": np.float32, "I64": np.int64, "F16": np.float16}
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    arr = np.frombuffer(raw, dtype=dtype_map[info["dtype"]])
    if info["shape"]:
        arr = arr.reshape(info["shape"])
    return arr


def read_raw_bytes(path, data_start, info):
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        return f.read(end - start)


# ---------------------------------------------------------------------------
# Ground truth (same as Phase 0.9)
# ---------------------------------------------------------------------------

def true_se_components(tensor_2d: np.ndarray) -> dict:
    if tensor_2d.ndim == 1:
        tensor_2d = tensor_2d.reshape(1, -1)
    m, n = tensor_2d.shape
    k = min(m, n, 64)

    try:
        _, S, _ = np.linalg.svd(tensor_2d.astype(np.float64), full_matrices=False)
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
        total_energy = 0.0
        S = np.array([0.0])

    H = 1.0 - energy_at_10pct

    if m > 1:
        row_corrs = []
        for i in range(min(m - 1, 100)):
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

    if total_energy > 0:
        normalized = (S ** 2) / total_energy
        normalized = normalized[normalized > 1e-12]
        spectral_entropy = -float(np.sum(normalized * np.log(normalized)))
        effective_rank = float(np.exp(spectral_entropy))
        rank_ratio = effective_rank / max(min(m, n), 1)
    else:
        rank_ratio = 0.0

    D = float(np.sqrt(min(rank_ratio, 1.0)))

    return {"H": H, "U": U, "D": D}


# ---------------------------------------------------------------------------
# Shadow projections (raw U8 bytes only, NO decompression)
# ---------------------------------------------------------------------------

def compute_all_projections(raw_bytes: bytes, shape: tuple) -> dict:
    """Compute all shadow projections from raw encoded bytes.

    body_opened_for_projection = false
    materialized_weight_bytes_for_projection = 0
    projection_source = raw_u8_indices
    """
    n = len(raw_bytes)
    if n < 64:
        return {k: 0.0 for k in [
            "byte_entropy", "index_autocorr", "transition_entropy",
            "transition_rank", "run_length_mean", "run_length_max",
            "local_entropy_var", "histogram_kurtosis", "byte_skew",
            "markov_order",
        ]}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    n = len(arr)

    # --- 1. Byte entropy (baseline from Phase 0.9) ---
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = counts[counts > 0] / n
    byte_entropy = -float(np.sum(probs * np.log2(probs))) / 8.0  # Normalized [0,1]

    # --- 2. Index autocorrelation (baseline from Phase 0.9) ---
    # Adjacent byte correlation (stride=1 = within-row for row-major)
    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.std(a), np.std(b)
    if sa > 1e-12 and sb > 1e-12:
        col_ac = float(np.mean((a - ma) * (b - mb)) / (sa * sb))
    else:
        col_ac = 0.0

    row_ac = 0.0
    if len(shape) == 2 and shape[1] > 1:
        stride = shape[1]
        if n > stride:
            a2 = arr[:-stride].astype(np.float64)
            b2 = arr[stride:].astype(np.float64)
            ma2, mb2 = np.mean(a2), np.mean(b2)
            sa2, sb2 = np.std(a2), np.std(b2)
            if sa2 > 1e-12 and sb2 > 1e-12:
                row_ac = float(np.mean((a2 - ma2) * (b2 - mb2)) / (sa2 * sb2))

    index_autocorr = max(abs(col_ac), abs(row_ac))

    # --- 3. Transition entropy (bigram entropy) ---
    # Entropy of consecutive byte pairs. Low = predictable sequences.
    # Use a sampled approach for large tensors.
    sample_size = min(n - 1, 500000)
    if sample_size < n - 1:
        idx = np.random.default_rng(42).choice(n - 1, sample_size, replace=False)
        idx.sort()
        pairs_a = arr[idx]
        pairs_b = arr[idx + 1]
    else:
        pairs_a = arr[:-1]
        pairs_b = arr[1:]

    # Encode pairs as uint16 for counting
    pair_codes = pairs_a.astype(np.uint16) * 256 + pairs_b.astype(np.uint16)
    pair_counts = np.bincount(pair_codes, minlength=65536).astype(np.float64)
    pair_probs = pair_counts[pair_counts > 0] / len(pair_codes)
    transition_entropy = -float(np.sum(pair_probs * np.log2(pair_probs))) / 16.0  # Normalized [0,1]

    # --- 4. Transition matrix effective rank proxy ---
    # How many distinct transition patterns exist?
    # Use: number of nonzero bigram types / 65536
    # Then: spectral proxy = entropy of row-wise transition distributions
    nonzero_bigrams = int(np.sum(pair_counts > 0))
    bigram_coverage = nonzero_bigrams / 65536.0

    # Row-wise transition entropy: for each source byte, entropy of next-byte distribution
    # This captures whether transitions are diverse (high rank) or concentrated (low rank)
    row_entropies = []
    transition_matrix = pair_counts.reshape(256, 256)
    for i in range(256):
        row = transition_matrix[i]
        row_sum = row.sum()
        if row_sum > 10:  # Need enough samples
            p = row[row > 0] / row_sum
            row_entropies.append(-float(np.sum(p * np.log2(p))) / 8.0)

    if row_entropies:
        # Effective rank proxy: mean row entropy × coverage
        mean_row_entropy = float(np.mean(row_entropies))
        transition_rank = mean_row_entropy * bigram_coverage
    else:
        transition_rank = 0.0

    # --- 5 & 6. Run-length statistics ---
    # Consecutive identical bytes. Long runs = structured/repetitive.
    run_lengths = []
    current_run = 1
    for i in range(1, min(n, 500000)):
        if arr[i] == arr[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    run_lengths_arr = np.array(run_lengths, dtype=np.float64)
    run_length_mean = float(np.mean(run_lengths_arr))
    run_length_max = float(np.max(run_lengths_arr))

    # --- 7. Local entropy variance ---
    # Variance of per-block entropies. High = structurally heterogeneous.
    block_size = min(4096, n // 8) if n > 1024 else n
    block_entropies = []
    for i in range(0, n - block_size + 1, block_size):
        block = arr[i:i + block_size]
        bc = np.bincount(block, minlength=256).astype(np.float64)
        p = bc[bc > 0] / len(block)
        block_entropies.append(-float(np.sum(p * np.log2(p))))

    if len(block_entropies) >= 2:
        local_entropy_var = float(np.var(block_entropies))
    else:
        local_entropy_var = 0.0

    # --- 8. Histogram kurtosis ---
    # Excess kurtosis of the byte value distribution.
    # High kurtosis = heavy tails = unusual index usage patterns.
    float_arr = arr.astype(np.float64)
    histogram_kurtosis = float(sp_stats.kurtosis(float_arr, fisher=True))

    # --- 9. Byte skew ---
    byte_skew = float(sp_stats.skew(float_arr))

    # --- 10. Markov order ---
    # Ratio of bigram entropy to unigram entropy.
    # If transitions add no information beyond frequency, ratio ≈ 1.
    # If transitions carry structure, ratio < 1.
    unigram_entropy = -float(np.sum(probs * np.log2(probs)))  # Unnormalized
    bigram_entropy_raw = -float(np.sum(pair_probs * np.log2(pair_probs)))
    if unigram_entropy > 0.01:
        markov_order = bigram_entropy_raw / (2 * unigram_entropy)  # Normalized
    else:
        markov_order = 1.0

    return {
        "byte_entropy": round(byte_entropy, 6),
        "index_autocorr": round(index_autocorr, 6),
        "transition_entropy": round(transition_entropy, 6),
        "transition_rank": round(transition_rank, 6),
        "run_length_mean": round(run_length_mean, 6),
        "run_length_max": round(run_length_max, 6),
        "local_entropy_var": round(local_entropy_var, 6),
        "histogram_kurtosis": round(histogram_kurtosis, 6),
        "byte_skew": round(byte_skew, 6),
        "markov_order": round(markov_order, 6),
    }


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def pearson_r(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.mean((x - mx) * (y - my)) / (sx * sy))


def spearman_r(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    r, _ = sp_stats.spearmanr(x, y)
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def atlas_data():
    """Compute ground truth and all projections for every HXQ tensor."""
    if not MAMBA_HXQ.exists():
        pytest.skip(f"Mamba HXQ not found: {MAMBA_HXQ}")

    header, data_start = read_safetensors_header(MAMBA_HXQ)
    tensors_info = {k: v for k, v in header.items() if k != "__metadata__"}

    results = []
    for name, info in tensors_info.items():
        if not name.endswith(".indices"):
            continue
        base = name.replace(".indices", "")
        cb_name = f"{base}.codebook"
        if cb_name not in tensors_info:
            continue

        byte_size = info["data_offsets"][1] - info["data_offsets"][0]
        if byte_size < 1024:
            continue

        shape = tuple(info["shape"])

        # Ground truth: dequantize and compute true H, U, D
        indices = load_tensor_numpy(MAMBA_HXQ, data_start, info)
        codebook = load_tensor_numpy(MAMBA_HXQ, data_start, tensors_info[cb_name])
        float_tensor = codebook[indices.astype(np.int32)]
        true = true_se_components(float_tensor)

        # Shadow projections: raw bytes only
        raw_bytes = read_raw_bytes(MAMBA_HXQ, data_start, info)
        projections = compute_all_projections(raw_bytes, shape)

        results.append({
            "name": base,
            "shape": shape,
            "bytes": len(raw_bytes),
            "true": true,
            "projections": projections,
        })

    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataAvailable:

    def test_tensors_found(self, atlas_data):
        assert len(atlas_data) >= 10

    def test_projections_computed(self, atlas_data):
        for r in atlas_data:
            assert len(r["projections"]) == 10, f"{r['name']} missing projections"


class TestProjectionAtlas:
    """The core experiment: correlate each projection against H, U, D."""

    def test_full_correlation_table(self, atlas_data):
        """Build and print the complete projection → {H, U, D} correlation table.

        This is the measurement atlas. No interpretation — just numbers.
        """
        projection_names = list(atlas_data[0]["projections"].keys())
        targets = ["H", "U", "D"]

        true_values = {t: [r["true"][t] for r in atlas_data] for t in targets}
        proj_values = {p: [r["projections"][p] for r in atlas_data] for p in projection_names}

        # Phase 0.9 baselines
        baselines = {"H": -0.13, "U": 0.81, "D": 0.46}

        print(f"\n{'=' * 80}")
        print(f"PHASE 0.10: PROJECTION ATLAS — {len(atlas_data)} tensors")
        print(f"body_opened_for_projection = false")
        print(f"materialized_weight_bytes_for_projection = 0")
        print(f"projection_source = raw_u8_indices")
        print(f"{'=' * 80}")

        print(f"\n{'Projection':<24} {'corr_H':>8} {'corr_U':>8} {'corr_D':>8}   "
              f"{'spear_H':>8} {'spear_U':>8} {'spear_D':>8}")
        print("-" * 80)

        results = {}
        for proj_name in projection_names:
            pvals = proj_values[proj_name]
            row = {}
            for target in targets:
                tvals = true_values[target]
                row[f"pearson_{target}"] = pearson_r(pvals, tvals)
                row[f"spearman_{target}"] = spearman_r(pvals, tvals)
            results[proj_name] = row

            print(f"{proj_name:<24} "
                  f"{row['pearson_H']:>+8.4f} {row['pearson_U']:>+8.4f} {row['pearson_D']:>+8.4f}   "
                  f"{row['spearman_H']:>+8.4f} {row['spearman_U']:>+8.4f} {row['spearman_D']:>+8.4f}")

        # Baselines
        print(f"\n{'Phase 0.9 baselines':<24} "
              f"{baselines['H']:>+8.4f} {baselines['U']:>+8.4f} {baselines['D']:>+8.4f}")

        # Find best projection for each target
        print(f"\n--- Best projections per target (Pearson) ---")
        for target in targets:
            best_name = max(projection_names,
                           key=lambda p: abs(results[p][f"pearson_{target}"]))
            best_r = results[best_name][f"pearson_{target}"]
            baseline = baselines[target]
            improved = abs(best_r) > abs(baseline)
            marker = "IMPROVED" if improved else "no improvement"
            print(f"  {target}: {best_name} (r={best_r:+.4f}) vs baseline (r={baseline:+.4f}) → {marker}")

        # Verdict
        print(f"\n{'=' * 80}")
        print("VERDICT:")

        best_U_name = max(projection_names, key=lambda p: abs(results[p]["pearson_U"]))
        best_U_r = results[best_U_name]["pearson_U"]
        best_D_name = max(projection_names, key=lambda p: abs(results[p]["pearson_D"]))
        best_D_r = results[best_D_name]["pearson_D"]

        u_improved = abs(best_U_r) > 0.81
        d_improved = abs(best_D_r) > 0.46

        if u_improved:
            print(f"  U: IMPROVED — {best_U_name} r={best_U_r:+.4f} > baseline 0.81")
        else:
            print(f"  U: NOT IMPROVED — best={best_U_name} r={best_U_r:+.4f}, baseline=0.81")

        if d_improved:
            print(f"  D: IMPROVED — {best_D_name} r={best_D_r:+.4f} > baseline 0.46")
        else:
            print(f"  D: NOT IMPROVED — best={best_D_name} r={best_D_r:+.4f}, baseline=0.46")

        if not u_improved and not d_improved:
            print(f"\n  Neither U nor D improved. Transition projections do not")
            print(f"  add signal beyond Phase 0.9 baselines on this data.")

        print(f"\n  best_U_projection: {best_U_name} (r={best_U_r:+.4f})")
        print(f"  best_D_projection: {best_D_name} (r={best_D_r:+.4f})")
        print(f"{'=' * 80}")

    def test_d_improvement_detail(self, atlas_data):
        """Detailed analysis of D correlations — the main target."""
        projection_names = list(atlas_data[0]["projections"].keys())
        true_D = [r["true"]["D"] for r in atlas_data]

        print(f"\n--- D (rank depth) correlation detail ---")
        print(f"  Phase 0.9 baseline: r=+0.46 (histogram diversity × meta-entropy)")
        print(f"  True D range: [{min(true_D):.4f}, {max(true_D):.4f}], mean={np.mean(true_D):.4f}")
        print()

        d_results = []
        for proj_name in projection_names:
            pvals = [r["projections"][proj_name] for r in atlas_data]
            pr = pearson_r(pvals, true_D)
            sr = spearman_r(pvals, true_D)
            d_results.append((proj_name, pr, sr))

        d_results.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, pr, sr in d_results:
            marker = " *** BEST" if abs(pr) == max(abs(x[1]) for x in d_results) else ""
            print(f"  {name:<24} pearson={pr:+.4f}  spearman={sr:+.4f}{marker}")

    def test_u_improvement_detail(self, atlas_data):
        """Detailed analysis of U correlations."""
        projection_names = list(atlas_data[0]["projections"].keys())
        true_U = [r["true"]["U"] for r in atlas_data]

        print(f"\n--- U (unstructuredness) correlation detail ---")
        print(f"  Phase 0.9 baseline: r=+0.81 (adjacent byte autocorrelation)")
        print()

        u_results = []
        for proj_name in projection_names:
            pvals = [r["projections"][proj_name] for r in atlas_data]
            pr = pearson_r(pvals, true_U)
            sr = spearman_r(pvals, true_U)
            u_results.append((proj_name, pr, sr))

        u_results.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, pr, sr in u_results:
            marker = " *** BEST" if abs(pr) == max(abs(x[1]) for x in u_results) else ""
            print(f"  {name:<24} pearson={pr:+.4f}  spearman={sr:+.4f}{marker}")

    def test_projection_ranges(self, atlas_data):
        """Report the range and variance of each projection.

        Flat projections (near-zero variance) cannot carry signal.
        """
        projection_names = list(atlas_data[0]["projections"].keys())

        print(f"\n--- Projection ranges (flat = dead) ---")
        for proj_name in projection_names:
            vals = [r["projections"][proj_name] for r in atlas_data]
            v_min, v_max = min(vals), max(vals)
            v_std = float(np.std(vals))
            v_range = v_max - v_min
            flat = "FLAT" if v_range < 0.01 else ""
            print(f"  {proj_name:<24} range=[{v_min:.6f}, {v_max:.6f}] "
                  f"std={v_std:.6f} span={v_range:.6f} {flat}")
