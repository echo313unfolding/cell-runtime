"""Phase 0.16: Ghost Execution Prediction — can compressed-domain features
predict execution-side behavior?

Classification is solved (Phase 0.15: 73.3% role, 92.8% family/arch).
This phase asks the harder question: can Ghost predict what a tensor DOES?

v2: Extended to 292 tensors (Mamba-130M + Qwen-1.5B) to break shape confound.
    Mamba roles have fixed shapes per role; Qwen has varied shapes across roles.
    Mixed-architecture dataset forces shape and ghost to compete on unequal footing.

Inputs (compressed-domain, body unopened):
  Ghost coordinates: transition_entropy, transition_rank, markov_order,
                     index_autocorr, complexity, predictability, locality

Targets (from dequantized body):
  spectral_norm     — largest singular value (amplification factor)
  stable_rank       — ||W||_F^2 / ||W||_2^2 (effective dimensionality)
  condition_number  — sigma_max / sigma_min (numerical stability)
  output_norm       — ||X @ W|| for fixed random X (actual amplification)
  output_variance   — var(X @ W) (output spread)
  output_sparsity   — fraction of |Y_i| < threshold (activation sparsity)

Critical controls:
  1. Shape baseline — big tensors can fake correlation. rows, cols, numel
     are baseline predictors. Ghost must beat or improve on shape-only.
  2. Three predictor sets compared:
     - shape-only (rows, cols, numel)
     - ghost-only (7 ghost features)
     - shape+ghost (all features)
  3. Per-role stratification — signal must not be pure role-label leakage.

Hydra Router connection:
  Current Hydra routes on kurtosis + cosine_under_each_head + tensor_type.
  ALL of those require the decompressed body.
  If Ghost predicts execution behavior from compressed domain, Hydra could
  use Ghost features as a pre-routing signal BEFORE trial quantization.

Pass condition:
  Ghost-only beats shape-only on at least one execution target, OR
  shape+ghost improves materially (>0.05 R²) over shape-only.

Receipt fields:
  ghost_source = raw_u8_indices
  body_opened_for_ghost = false
  body_opened_for_targets = true
  prediction_target = execution_behavior

Data: 96 Mamba-130M HXQ + 196 Qwen-1.5B HXQ = 292 tensors, 11 roles,
      varied shapes (shape confound broken).

WO-CRYSTAL-VAULT-01: Phase 0.16 — Ghost Execution Prediction
"""
import json
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MAMBA_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
    "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
)

QWEN_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--qwen2.5-coder-1.5b-helix"
    "/snapshots/0a5c17fba5cc81018423eba394295ca8568caff2/model.safetensors"
)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

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
# Role classifiers
# ---------------------------------------------------------------------------

def classify_mamba_role(name: str) -> str:
    parent = name.replace(".indices", "")
    if "in_proj" in parent:
        return "ssm_in_proj"
    if "out_proj" in parent:
        return "ssm_out_proj"
    if "dt_proj" in parent:
        return "ssm_dt"
    if "x_proj" in parent:
        return "ssm_x"
    return "other"


def classify_qwen_role(name: str) -> str:
    n = name.lower()
    if "q_proj" in n:
        return "attn_q"
    if "k_proj" in n:
        return "attn_k"
    if "v_proj" in n:
        return "attn_v"
    if "o_proj" in n:
        return "attn_o"
    if "gate_proj" in n:
        return "ffn_gate"
    if "up_proj" in n:
        return "ffn_up"
    if "down_proj" in n:
        return "ffn_down"
    return "other"


def role_family(role: str) -> str:
    if role.startswith("attn_"):
        return "attention"
    if role.startswith("ffn_"):
        return "ffn"
    if role.startswith("ssm_"):
        return "ssm"
    return role


# ---------------------------------------------------------------------------
# Ghost features (compressed-domain, body unopened)
# ---------------------------------------------------------------------------

def ghost_features_from_bytes(raw_bytes: bytes, shape: tuple) -> dict:
    """All 7 ghost features from raw U8 index bytes.

    body_opened_for_ghost = false
    ghost_source = raw_u8_indices
    """
    n = len(raw_bytes)
    if n < 64:
        return {k: 0.0 for k in [
            "te", "tr", "mo", "ac", "complexity", "predictability", "locality"
        ]}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Bigram matrix (sampled for large tensors)
    sample_size = min(n - 1, 500000)
    rng = np.random.default_rng(42)
    if sample_size < n - 1:
        idx = rng.choice(n - 1, sample_size, replace=False)
        idx.sort()
        pairs_a = arr[idx]
        pairs_b = arr[idx + 1]
    else:
        pairs_a = arr[:-1]
        pairs_b = arr[1:]

    pair_codes = pairs_a.astype(np.uint16) * 256 + pairs_b.astype(np.uint16)
    pair_counts = np.bincount(pair_codes, minlength=65536).astype(np.float64)
    total_pairs = len(pair_codes)

    # TE
    pair_probs = pair_counts[pair_counts > 0] / total_pairs
    bigram_h = -float(np.sum(pair_probs * np.log2(pair_probs)))
    max_bigram_h = 2.0 * np.log2(256)
    te = bigram_h / max_bigram_h if max_bigram_h > 0 else 0.0

    # TR
    transition_matrix = pair_counts.reshape(256, 256)
    row_sums = transition_matrix.sum(axis=1)
    row_entropies = []
    for i in range(256):
        if row_sums[i] > 10:
            rp = transition_matrix[i]
            rp = rp[rp > 0] / row_sums[i]
            row_entropies.append(-float(np.sum(rp * np.log2(rp))) / 8.0)
    nonzero_bigrams = int(np.sum(pair_counts > 0))
    bigram_coverage = nonzero_bigrams / 65536.0
    tr = float(np.mean(row_entropies)) * bigram_coverage if row_entropies else 0.0

    # MO
    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 1.0

    # AC
    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    col_ac = float(np.mean((a - ma) * (b - mb)) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0

    row_ac = 0.0
    if len(shape) == 2 and shape[1] > 1:
        stride = shape[1]
        if n > stride:
            a2 = arr[:-stride].astype(np.float64)
            b2 = arr[stride:].astype(np.float64)
            ma2, mb2 = a2.mean(), b2.mean()
            sa2, sb2 = a2.std(), b2.std()
            if sa2 > 1e-12 and sb2 > 1e-12:
                row_ac = float(np.mean((a2 - ma2) * (b2 - mb2)) / (sa2 * sb2))

    ac = max(abs(col_ac), abs(row_ac))

    complexity = (te + tr) / 2.0
    predictability = mo
    locality = ac

    return {
        "te": round(te, 6),
        "tr": round(tr, 6),
        "mo": round(mo, 6),
        "ac": round(ac, 6),
        "complexity": round(complexity, 6),
        "predictability": round(predictability, 6),
        "locality": round(locality, 6),
    }


# ---------------------------------------------------------------------------
# Execution targets (from dequantized body)
# ---------------------------------------------------------------------------

def execution_targets(float_tensor: np.ndarray) -> dict:
    """Compute execution-side measurements from dequantized tensor.

    body_opened_for_targets = true
    """
    W = float_tensor.astype(np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)

    rows, cols = W.shape

    # Spectral norm — largest singular value
    k = min(rows, cols, 64)
    try:
        _, S, _ = np.linalg.svd(W, full_matrices=False)
        S = S[:k]
        spectral_norm = float(S[0])
        sigma_min = float(S[min(len(S) - 1, k - 1)])
        condition_number = spectral_norm / sigma_min if sigma_min > 1e-12 else 1e12
        frobenius_sq = float(np.sum(S ** 2))
        stable_rank = frobenius_sq / (spectral_norm ** 2) if spectral_norm > 1e-12 else 0.0
    except np.linalg.LinAlgError:
        spectral_norm = 0.0
        condition_number = 1e12
        stable_rank = 0.0

    # Output on fixed random X
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1, rows))
    Y = X @ W
    output_norm = float(np.linalg.norm(Y))
    output_variance = float(np.var(Y))
    output_sparsity = float(np.mean(np.abs(Y) < 1e-3))

    return {
        "spectral_norm": spectral_norm,
        "stable_rank": stable_rank,
        "condition_number": min(condition_number, 1e12),
        "output_norm": output_norm,
        "output_variance": output_variance,
        "output_sparsity": output_sparsity,
    }


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def r_squared(X: np.ndarray, y: np.ndarray) -> float:
    """OLS R² for X (n, p) predicting y (n,)."""
    n, p = X.shape
    if n <= p + 1:
        return 0.0
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_hat = X_aug @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return 1.0 - ss_res / ss_tot
    except np.linalg.LinAlgError:
        return 0.0


def adjusted_r_squared(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return 0.0
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def spearman_r(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if len(x) < 3:
        return 0.0, 1.0
    r, p = sp_stats.spearmanr(x, y)
    return (float(r), float(p)) if not np.isnan(r) else (0.0, 1.0)


# ---------------------------------------------------------------------------
# Fixture: scan both models
# ---------------------------------------------------------------------------

def _scan_model(path, role_fn, arch_name):
    """Scan one model's HXQ safetensors for ghost features + execution targets."""
    if not path.exists():
        return []

    header, data_start = read_safetensors_header(path)
    tensors_info = {k: v for k, v in header.items() if k != "__metadata__"}

    results = []
    for name, info in tensors_info.items():
        if not name.endswith(".indices"):
            continue
        if info["dtype"] != "U8":
            continue

        base = name.replace(".indices", "")
        cb_name = f"{base}.codebook"
        if cb_name not in tensors_info:
            continue

        role = role_fn(name)
        if role == "other" or role == "skip":
            continue

        byte_size = info["data_offsets"][1] - info["data_offsets"][0]
        if byte_size < 1024:
            continue

        shape = tuple(info["shape"])

        # Ghost features — compressed domain
        raw_bytes = read_raw_bytes(path, data_start, info)
        ghost = ghost_features_from_bytes(raw_bytes, shape)

        # Execution targets — dequantize
        indices = load_tensor_numpy(path, data_start, info)
        codebook = load_tensor_numpy(path, data_start, tensors_info[cb_name])
        float_tensor = codebook[indices.astype(np.int32)]
        targets = execution_targets(float_tensor)

        # Shape features
        rows = shape[0] if len(shape) >= 2 else 1
        cols = shape[1] if len(shape) >= 2 else shape[0]

        # Layer number
        layer = -1
        for part in name.split("."):
            if part.isdigit():
                layer = int(part)
                break

        results.append({
            "name": base,
            "role": role,
            "family": role_family(role),
            "arch": arch_name,
            "layer": layer,
            "shape": shape,
            "rows": rows,
            "cols": cols,
            "numel": rows * cols,
            "ghost": ghost,
            "targets": targets,
        })

    return results


@pytest.fixture(scope="module")
def prediction_data():
    """Ghost features + execution targets + shape for 292 tensors."""
    results = []
    results.extend(_scan_model(MAMBA_HXQ, classify_mamba_role, "mamba"))
    results.extend(_scan_model(QWEN_HXQ, classify_qwen_role, "transformer"))

    if len(results) < 50:
        pytest.skip(f"Need >= 50 tensors, got {len(results)}")

    return results


# ===========================================================================
# Tests
# ===========================================================================


class TestDataAvailable:
    def test_sufficient_tensors(self, prediction_data):
        n = len(prediction_data)
        assert n >= 100, f"Need >= 100 tensors, got {n}"
        print(f"\n  Tensors: {n}")

    def test_shape_diversity(self, prediction_data):
        """Verify shapes are diverse — the key to breaking the confound."""
        shapes = set(r["shape"] for r in prediction_data)
        print(f"\n  Unique shapes: {len(shapes)}")
        numel_vals = [r["numel"] for r in prediction_data]
        print(f"  numel range: [{min(numel_vals)}, {max(numel_vals)}]")
        print(f"  numel unique: {len(set(numel_vals))}")
        # Qwen has varied shapes across roles — must have >4 unique shapes
        assert len(shapes) > 4, f"Need shape diversity, got only {len(shapes)}"

    def test_role_balance(self, prediction_data):
        role_counts = Counter(r["role"] for r in prediction_data)
        print(f"\n  Role distribution:")
        for role, count in sorted(role_counts.items()):
            print(f"    {role:<16} {count}")

    def test_target_ranges(self, prediction_data):
        target_names = list(prediction_data[0]["targets"].keys())
        print(f"\n  Execution target ranges:")
        for tn in target_names:
            vals = [r["targets"][tn] for r in prediction_data]
            v_min, v_max = min(vals), max(vals)
            v_std = float(np.std(vals))
            if tn == "condition_number":
                log_vals = [np.log10(max(v, 1.0)) for v in vals]
                print(f"    {tn:<20} [{v_min:.4g}, {v_max:.4g}] "
                      f"log10=[{min(log_vals):.2f}, {max(log_vals):.2f}]")
            else:
                print(f"    {tn:<20} [{v_min:.6g}, {v_max:.6g}] std={v_std:.6g}")


class TestShapeLeakageControl:
    """The critical control: does Ghost beat shape-only prediction?"""

    def test_three_predictor_comparison(self, prediction_data):
        """Compare shape-only, ghost-only, and shape+ghost for every target."""
        n = len(prediction_data)
        target_names = list(prediction_data[0]["targets"].keys())

        shape_features = np.array([
            [np.log(max(r["rows"], 1)), np.log(max(r["cols"], 1)),
             np.log(max(r["numel"], 1))]
            for r in prediction_data
        ])

        ghost_features = np.array([
            [r["ghost"]["te"], r["ghost"]["tr"], r["ghost"]["mo"],
             r["ghost"]["ac"], r["ghost"]["complexity"],
             r["ghost"]["predictability"], r["ghost"]["locality"]]
            for r in prediction_data
        ])

        combined_features = np.column_stack([shape_features, ghost_features])

        print(f"\n{'=' * 80}")
        print("PHASE 0.16: GHOST EXECUTION PREDICTION (292 tensors, mixed architecture)")
        print(f"{'=' * 80}")
        print(f"\n  Tensors: {n} (Mamba + Qwen)")
        print(f"  Unique shapes: {len(set(r['shape'] for r in prediction_data))}")
        print(f"  Shape features: 3 (log_rows, log_cols, log_numel)")
        print(f"  Ghost features: 7 (te, tr, mo, ac, complexity, predictability, locality)")
        print(f"\n  ghost_source = raw_u8_indices")
        print(f"  body_opened_for_ghost = false")
        print(f"  body_opened_for_targets = true")

        print(f"\n  {'Target':<20} {'Shape R²':>10} {'Ghost R²':>10} "
              f"{'Both R²':>10} {'Δ(G-S)':>10} {'Δ(B-S)':>10} {'Winner':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        ghost_wins = 0
        combo_wins = 0
        best_target = None
        best_delta = -999

        for tn in target_names:
            y = np.array([r["targets"][tn] for r in prediction_data])
            if tn == "condition_number":
                y = np.log10(np.clip(y, 1.0, 1e12))

            adj_shape = adjusted_r_squared(r_squared(shape_features, y), n, 3)
            adj_ghost = adjusted_r_squared(r_squared(ghost_features, y), n, 7)
            adj_both = adjusted_r_squared(r_squared(combined_features, y), n, 10)

            delta = adj_ghost - adj_shape
            combo_delta = adj_both - adj_shape

            if delta > 0:
                winner = "GHOST"
                ghost_wins += 1
            else:
                winner = "shape"
            if combo_delta > 0.05:
                combo_wins += 1

            if delta > best_delta:
                best_delta = delta
                best_target = tn

            print(f"  {tn:<20} {adj_shape:>10.4f} {adj_ghost:>10.4f} "
                  f"{adj_both:>10.4f} {delta:>+10.4f} {combo_delta:>+10.4f} {winner:>8}")

        print(f"\n  Ghost beats shape-only on {ghost_wins}/{len(target_names)} targets")
        print(f"  Shape+Ghost material gain (>0.05) on {combo_wins}/{len(target_names)} targets")
        print(f"  Best ghost-vs-shape delta: {best_target} ({best_delta:+.4f})")

    def test_shape_leakage_check(self, prediction_data):
        """Ghost↔shape correlation — is Ghost just rediscovering size?"""
        ghost_feat_names = ["te", "tr", "mo", "ac", "complexity", "predictability", "locality"]
        log_numel = np.log(np.array([max(r["numel"], 1) for r in prediction_data], dtype=np.float64))

        print(f"\n  Ghost ↔ log(numel) correlation (shape leakage check):")
        max_leak = 0.0
        for gf in ghost_feat_names:
            vals = np.array([r["ghost"][gf] for r in prediction_data])
            rho, p = spearman_r(vals, log_numel)
            leak = "LEAKS" if abs(rho) > 0.5 else ("weak" if abs(rho) > 0.3 else "clean")
            print(f"    {gf:<16} rho={rho:+.4f}  p={p:.4g}  {leak}")
            max_leak = max(max_leak, abs(rho))
        print(f"  Max leakage: {max_leak:.4f}")


class TestPerRoleExecution:
    """Per-role stratification — does signal survive within same-shape tensors?"""

    def test_within_role_correlation(self, prediction_data):
        """Within-role Ghost↔execution correlation. Same shape → no shape leakage."""
        target_names = list(prediction_data[0]["targets"].keys())
        ghost_feat_names = ["te", "tr", "mo", "ac", "complexity", "predictability", "locality"]

        by_role = defaultdict(list)
        for r in prediction_data:
            by_role[r["role"]].append(r)

        print(f"\n  Within-role Ghost↔execution correlation (Spearman):")
        print(f"  (Same role = same shape → shape leakage impossible)")

        sig_count = 0
        total_tests = 0

        for role in sorted(by_role.keys()):
            pts = by_role[role]
            if len(pts) < 5:
                continue
            print(f"\n    {role} (n={len(pts)}, shape={pts[0]['shape']}):")

            for tn in target_names:
                y = np.array([p["targets"][tn] for p in pts])
                if tn == "condition_number":
                    y = np.log10(np.clip(y, 1.0, 1e12))
                if np.std(y) < 1e-12:
                    continue

                best_rho = 0.0
                best_gf = ""
                best_p = 1.0
                for gf in ghost_feat_names:
                    gv = np.array([p["ghost"][gf] for p in pts])
                    if np.std(gv) < 1e-12:
                        continue
                    rho, p = spearman_r(gv, y)
                    if abs(rho) > abs(best_rho):
                        best_rho = rho
                        best_gf = gf
                        best_p = p

                total_tests += 1
                sig = "***" if best_p < 0.001 else ("**" if best_p < 0.01 else ("*" if best_p < 0.05 else ""))
                if best_p < 0.05:
                    sig_count += 1
                print(f"      {tn:<20} best={best_gf:<14} rho={best_rho:+.4f}  p={best_p:.4g} {sig}")

        print(f"\n  Significant within-role correlations: {sig_count}/{total_tests} (p<0.05)")


class TestHydraRouterRelevance:
    """Can Ghost predict features that Hydra Router currently needs the body for?"""

    def test_ghost_predicts_kurtosis_proxy(self, prediction_data):
        """Hydra routes on kurtosis. Can Ghost predict kurtosis-like behavior?

        We don't have raw kurtosis (would need FP16 originals), but
        condition_number and spectral_norm are correlated proxies.
        If Ghost predicts these, it could pre-route before trial quantization.
        """
        ghost_feat_names = ["te", "tr", "mo", "ac", "complexity", "predictability", "locality"]

        # Hydra-relevant targets: the things Hydra currently needs body access for
        hydra_targets = {
            "condition_number": "numerical stability → fragile tensor detection",
            "spectral_norm": "amplification factor → quantization sensitivity proxy",
            "stable_rank": "effective rank → complexity of weight structure",
        }

        print(f"\n  Hydra Router relevance — can Ghost pre-route?")
        print(f"  (Hydra currently requires body access for kurtosis, cosine, tensor_type)")
        print(f"  (Ghost sees NONE of these — only compressed-domain transition features)")

        for tn, desc in hydra_targets.items():
            y = np.array([r["targets"][tn] for r in prediction_data])
            if tn == "condition_number":
                y = np.log10(np.clip(y, 1.0, 1e12))

            print(f"\n    {tn} ({desc}):")
            for gf in ghost_feat_names:
                gv = np.array([r["ghost"][gf] for r in prediction_data])
                rho, p = spearman_r(gv, y)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                if abs(rho) > 0.3:
                    print(f"      {gf:<16} rho={rho:+.4f} {sig}")

    def test_ghost_fragile_tensor_detection(self, prediction_data):
        """Can Ghost flag fragile tensors (high condition number) without body access?

        Hydra's edge_balanced policy sends fragile tensors to affine6.
        If Ghost can identify fragile tensors, it can pre-filter before profiling.
        """
        # Define "fragile" as top 25% condition number
        cond_nums = [r["targets"]["condition_number"] for r in prediction_data]
        threshold = np.percentile(cond_nums, 75)

        fragile = [1 if r["targets"]["condition_number"] >= threshold else 0
                   for r in prediction_data]
        fragile_arr = np.array(fragile)

        ghost_feat_names = ["te", "tr", "mo", "ac", "complexity", "predictability", "locality"]

        print(f"\n  Fragile tensor detection (condition_number >= {threshold:.2f}):")
        print(f"  Fragile: {sum(fragile)}/{len(fragile)}")

        # Point-biserial correlation (= Pearson for binary vs continuous)
        for gf in ghost_feat_names:
            gv = np.array([r["ghost"][gf] for r in prediction_data])
            rho, p = spearman_r(gv, fragile_arr)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if abs(rho) > 0.2:
                print(f"    {gf:<16} rho={rho:+.4f} {sig}")

        # Simple threshold classifier: can Ghost's best feature identify fragile?
        best_gf = None
        best_auc = 0.5
        for gf in ghost_feat_names:
            gv = np.array([r["ghost"][gf] for r in prediction_data])
            # AUC via rank correlation
            rho, _ = spearman_r(gv, fragile_arr)
            auc = 0.5 + rho / 2  # Approximate AUC from Spearman
            if abs(auc - 0.5) > abs(best_auc - 0.5):
                best_auc = auc
                best_gf = gf

        print(f"\n  Best fragile detector: {best_gf} (approx AUC={best_auc:.3f})")
        if best_auc > 0.65:
            print(f"  → Ghost could pre-screen fragile tensors for Hydra")
        else:
            print(f"  → Ghost cannot reliably detect fragile tensors")


class TestReport:
    def test_full_report(self, prediction_data):
        n = len(prediction_data)
        target_names = list(prediction_data[0]["targets"].keys())
        ghost_feat_names = ["te", "tr", "mo", "ac", "complexity", "predictability", "locality"]

        shape_features = np.array([
            [np.log(max(r["rows"], 1)), np.log(max(r["cols"], 1)),
             np.log(max(r["numel"], 1))]
            for r in prediction_data
        ])
        ghost_features = np.array([
            [r["ghost"]["te"], r["ghost"]["tr"], r["ghost"]["mo"],
             r["ghost"]["ac"], r["ghost"]["complexity"],
             r["ghost"]["predictability"], r["ghost"]["locality"]]
            for r in prediction_data
        ])
        combined_features = np.column_stack([shape_features, ghost_features])

        ghost_wins = 0
        combo_material_wins = 0
        best_target = None
        best_delta = -999
        best_ghost_metric = None

        results = {}
        for tn in target_names:
            y = np.array([r["targets"][tn] for r in prediction_data])
            if tn == "condition_number":
                y = np.log10(np.clip(y, 1.0, 1e12))

            adj_shape = adjusted_r_squared(r_squared(shape_features, y), n, 3)
            adj_ghost = adjusted_r_squared(r_squared(ghost_features, y), n, 7)
            adj_both = adjusted_r_squared(r_squared(combined_features, y), n, 10)

            delta = adj_ghost - adj_shape
            combo_delta = adj_both - adj_shape

            # Best single ghost feature
            best_gf_rho = 0.0
            best_gf_name = ""
            for gf in ghost_feat_names:
                gv = np.array([r["ghost"][gf] for r in prediction_data])
                rho, _ = spearman_r(gv, y)
                if abs(rho) > abs(best_gf_rho):
                    best_gf_rho = rho
                    best_gf_name = gf

            results[tn] = {
                "adj_shape": adj_shape, "adj_ghost": adj_ghost, "adj_both": adj_both,
                "delta": delta, "combo_delta": combo_delta,
                "best_gf": best_gf_name, "best_gf_rho": best_gf_rho,
            }

            if delta > 0:
                ghost_wins += 1
            if combo_delta > 0.05:
                combo_material_wins += 1
            if delta > best_delta:
                best_delta = delta
                best_target = tn
                best_ghost_metric = best_gf_name

        # Shape leakage
        log_numel = np.log(np.array([max(r["numel"], 1) for r in prediction_data], dtype=np.float64))
        max_leak = 0.0
        for gf in ghost_feat_names:
            gv = np.array([r["ghost"][gf] for r in prediction_data])
            rho, _ = spearman_r(gv, log_numel)
            max_leak = max(max_leak, abs(rho))

        # Within-role sig count
        by_role = defaultdict(list)
        for r in prediction_data:
            by_role[r["role"]].append(r)
        within_sig = 0
        within_total = 0
        for role, pts in by_role.items():
            if len(pts) < 5:
                continue
            for tn in target_names:
                y = np.array([p["targets"][tn] for p in pts])
                if tn == "condition_number":
                    y = np.log10(np.clip(y, 1.0, 1e12))
                if np.std(y) < 1e-12:
                    continue
                within_total += 1
                for gf in ghost_feat_names:
                    gv = np.array([p["ghost"][gf] for p in pts])
                    if np.std(gv) < 1e-12:
                        continue
                    _, p_val = spearman_r(gv, y)
                    if p_val < 0.05:
                        within_sig += 1
                        break  # count each target once per role

        passed = ghost_wins > 0 or combo_material_wins > 0

        print(f"""
{'=' * 80}
PHASE 0.16: GHOST EXECUTION PREDICTION — FINAL REPORT (v2, 292 tensors)
{'=' * 80}

Tensors: {n} (Mamba-130M: 96, Qwen-1.5B: 196)
Roles: {len(set(r['role'] for r in prediction_data))}
Unique shapes: {len(set(r['shape'] for r in prediction_data))}
Ghost features: 7 (compressed-domain transition graph)
Shape features: 3 (log_rows, log_cols, log_numel)
Execution targets: {len(target_names)}

Receipt:
  ghost_source = raw_u8_indices
  body_opened_for_ghost = false
  body_opened_for_targets = true
  prediction_target = execution_behavior
  best_target = {best_target}
  best_ghost_metric = {best_ghost_metric}
  shape_baseline_delta = {best_delta:+.4f}
  max_shape_leakage = {max_leak:.4f}
  within_role_significant = {within_sig}/{within_total}

--- Predictor Comparison (Adjusted R²) ---

  {'Target':<20} {'Shape':>8} {'Ghost':>8} {'Both':>8} {'Δ(G-S)':>10} {'Δ(B-S)':>10}
  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}""")

        for tn in target_names:
            r = results[tn]
            w = "GHOST" if r["delta"] > 0 else "shape"
            print(f"  {tn:<20} {r['adj_shape']:>8.4f} {r['adj_ghost']:>8.4f} "
                  f"{r['adj_both']:>8.4f} {r['delta']:>+10.4f} {r['combo_delta']:>+10.4f}  {w}")

        print(f"""
--- Summary ---
  Ghost beats shape: {ghost_wins}/{len(target_names)} targets
  Shape+Ghost gain >0.05: {combo_material_wins}/{len(target_names)} targets
  Best target: {best_target} (Δ={best_delta:+.4f})
  Shape leakage: {max_leak:.4f} {'(CLEAN <0.5)' if max_leak < 0.5 else '(WARNING >0.5)'}
  Within-role signal: {within_sig}/{within_total} (shape-independent)

--- Hydra Router Implication ---""")

        # Quick hydra check
        cond_rhos = []
        for gf in ghost_feat_names:
            gv = np.array([r["ghost"][gf] for r in prediction_data])
            y = np.log10(np.clip(np.array([r["targets"]["condition_number"] for r in prediction_data]), 1.0, 1e12))
            rho, _ = spearman_r(gv, y)
            cond_rhos.append((gf, rho))
        best_cond = max(cond_rhos, key=lambda x: abs(x[1]))
        print(f"  Best Ghost→condition_number: {best_cond[0]} (rho={best_cond[1]:+.4f})")
        if abs(best_cond[1]) > 0.4:
            print(f"  → Ghost can pre-screen fragile tensors for Hydra Router")
            print(f"  → Hydra could use Ghost features as pre-routing signal")
        else:
            print(f"  → Ghost cannot reliably predict Hydra-relevant features")

        print(f"""
{'=' * 80}
VERDICT:
""")
        if passed and max_leak < 0.5:
            print(f"  PASS — Ghost predicts execution behavior beyond shape.")
            print(f"  Transition structure contains behavioral signal that shape misses.")
        elif passed and max_leak >= 0.5:
            if within_sig > 0:
                print(f"  CONDITIONAL PASS — shape leakage present ({max_leak:.3f}),")
                print(f"  but {within_sig}/{within_total} within-role correlations are significant.")
                print(f"  Within-role = same shape → shape leakage impossible there.")
                print(f"  Ghost carries execution signal BEYOND shape.")
            else:
                print(f"  CONDITIONAL PASS — Ghost predicts, but shape leakage ({max_leak:.3f})")
                print(f"  and no within-role signal. May be shape in disguise.")
        else:
            print(f"  FAIL — Ghost does not predict execution behavior beyond shape.")
            print(f"  Classification works (Phase 0.15), prediction does not.")
        print(f"{'=' * 80}")
