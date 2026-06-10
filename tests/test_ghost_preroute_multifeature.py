"""Phase 0.17b: Architecture-Aware Multi-Feature Pre-Route

Phase 0.17 showed: single global AC threshold clears only 8.6% of tensors.
Root cause: Mamba and Transformer occupy different Ghost regimes.
All Mamba tensors cluster at AC ≈ 0 — one threshold can't separate them.

This phase tests:
1. Global single-feature threshold (baseline, Phase 0.17 replay)
2. Global multi-feature threshold (logistic regression on all 4 ghost features)
3. Architecture-aware multi-feature threshold (separate model per architecture)

Safety constraints (same as Phase 0.17):
  precision_safe >= 0.95
  recall_fragile >= 0.90 (tightened from 0.80)

Efficiency target:
  cleared_fraction >= 0.30 first (relaxed from 0.50)

Data: 292 tensors from Phase 0.16/0.17.

WO-CRYSTAL-VAULT-01: Phase 0.17b — Architecture-Aware Multi-Feature Pre-Route
"""
import json
import struct
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
# File helpers (same as Phase 0.17)
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
    if "q_proj" in n: return "attn_q"
    if "k_proj" in n: return "attn_k"
    if "v_proj" in n: return "attn_v"
    if "o_proj" in n: return "attn_o"
    if "gate_proj" in n: return "ffn_gate"
    if "up_proj" in n: return "ffn_up"
    if "down_proj" in n: return "ffn_down"
    return "other"


# ---------------------------------------------------------------------------
# Ghost features (compressed-domain only)
# ---------------------------------------------------------------------------

def ghost_features_from_bytes(raw_bytes: bytes, shape: tuple) -> dict:
    n = len(raw_bytes)
    if n < 64:
        return {k: 0.0 for k in ["te", "tr", "mo", "ac"]}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

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

    pair_probs = pair_counts[pair_counts > 0] / total_pairs
    bigram_h = -float(np.sum(pair_probs * np.log2(pair_probs)))
    max_bigram_h = 2.0 * np.log2(256)
    te = bigram_h / max_bigram_h if max_bigram_h > 0 else 0.0

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

    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 1.0

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

    return {"te": round(te, 6), "tr": round(tr, 6), "mo": round(mo, 6), "ac": round(ac, 6)}


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def condition_number_from_tensor(float_tensor: np.ndarray) -> float:
    W = float_tensor.astype(np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    k = min(W.shape[0], W.shape[1], 64)
    try:
        _, S, _ = np.linalg.svd(W, full_matrices=False)
        S = S[:k]
        sigma_min = float(S[min(len(S) - 1, k - 1)])
        return float(S[0]) / sigma_min if sigma_min > 1e-12 else 1e12
    except np.linalg.LinAlgError:
        return 1e12


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def logistic_score(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simple logistic regression via OLS on log-odds proxy.

    Returns predicted probabilities. Uses leave-one-out to avoid overfitting.
    """
    n = len(y)
    probs = np.zeros(n)
    X_aug = np.column_stack([np.ones(n), X])

    for i in range(n):
        # Leave one out
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X_aug[mask]
        y_train = y[mask].astype(np.float64)

        # Regularized least squares on labels (poor man's logistic)
        try:
            beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
            raw = float(X_aug[i] @ beta)
            probs[i] = np.clip(raw, 0.0, 1.0)
        except np.linalg.LinAlgError:
            probs[i] = 0.5

    return probs


def evaluate_decision(probs: np.ndarray, labels: np.ndarray,
                      min_precision: float = 0.95,
                      min_recall: float = 0.90) -> dict:
    """Find best threshold on predicted probabilities.

    Optimize cleared_fraction while meeting safety constraints.
    """
    # Try 100 threshold candidates
    thresholds = np.linspace(0.0, 1.0, 200)
    best = None

    for t in thresholds:
        ghost_fragile = probs >= t
        ghost_safe = ~ghost_fragile
        truly_fragile = labels == 1
        truly_safe = labels == 0

        n_ghost_safe = ghost_safe.sum()
        if n_ghost_safe == 0:
            continue

        true_safe_in_ghost_safe = (ghost_safe & truly_safe).sum()
        precision_safe = true_safe_in_ghost_safe / n_ghost_safe

        n_truly_fragile = truly_fragile.sum()
        caught = (ghost_fragile & truly_fragile).sum()
        recall_fragile = caught / n_truly_fragile if n_truly_fragile > 0 else 1.0

        fn = (ghost_safe & truly_fragile).sum()
        fn_rate = fn / n_truly_fragile if n_truly_fragile > 0 else 0.0

        cleared = n_ghost_safe / len(labels)

        if precision_safe >= min_precision and recall_fragile >= min_recall:
            if best is None or cleared > best["cleared_fraction"]:
                best = {
                    "threshold": float(t),
                    "precision_safe": precision_safe,
                    "recall_fragile": recall_fragile,
                    "false_negatives": int(fn),
                    "false_negative_rate": fn_rate,
                    "cleared_fraction": cleared,
                    "n_cleared": int(n_ghost_safe),
                    "n_profiled": int(ghost_fragile.sum()),
                }

    return best


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _scan_model(path, role_fn, arch_name):
    if not path.exists():
        return []
    header, data_start = read_safetensors_header(path)
    tensors_info = {k: v for k, v in header.items() if k != "__metadata__"}
    results = []
    for name, info in tensors_info.items():
        if not name.endswith(".indices") or info["dtype"] != "U8":
            continue
        base = name.replace(".indices", "")
        cb_name = f"{base}.codebook"
        if cb_name not in tensors_info:
            continue
        role = role_fn(name)
        if role in ("other", "skip"):
            continue
        byte_size = info["data_offsets"][1] - info["data_offsets"][0]
        if byte_size < 1024:
            continue
        shape = tuple(info["shape"])

        raw_bytes = read_raw_bytes(path, data_start, info)
        ghost = ghost_features_from_bytes(raw_bytes, shape)

        indices = load_tensor_numpy(path, data_start, info)
        codebook = load_tensor_numpy(path, data_start, tensors_info[cb_name])
        float_tensor = codebook[indices.astype(np.int32)]
        cond = condition_number_from_tensor(float_tensor)

        results.append({
            "name": base, "role": role, "arch": arch_name,
            "shape": shape, "ghost": ghost, "condition_number": cond,
        })
    return results


@pytest.fixture(scope="module")
def decision_data():
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
    def test_sufficient_tensors(self, decision_data):
        n = len(decision_data)
        assert n >= 100
        print(f"\n  Tensors: {n}")

    def test_arch_split(self, decision_data):
        by_arch = Counter(r["arch"] for r in decision_data)
        print(f"\n  Architecture split: {dict(by_arch)}")
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        for arch in sorted(by_arch.keys()):
            arch_conds = [r["condition_number"] for r in decision_data if r["arch"] == arch]
            n_frag = sum(1 for c in arch_conds if c >= p75)
            print(f"    {arch}: {n_frag}/{len(arch_conds)} fragile (P75={p75:.3f})")


class TestMethod1GlobalSingle:
    """Baseline: best single global feature threshold (Phase 0.17 replay)."""

    def test_global_single(self, decision_data):
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)

        features = ["te", "tr", "mo", "ac"]
        best = None
        for gf in features:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            # Try both directions
            for direction in ["above", "below"]:
                sorted_vals = np.sort(np.unique(vals))
                for i in range(len(sorted_vals) - 1):
                    t = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
                    if direction == "above":
                        fragile_pred = vals >= t
                    else:
                        fragile_pred = vals <= t
                    safe_pred = ~fragile_pred

                    n_safe = safe_pred.sum()
                    if n_safe == 0:
                        continue

                    prec = (safe_pred & (labels == 0)).sum() / n_safe
                    n_frag = labels.sum()
                    rec = (fragile_pred & (labels == 1)).sum() / n_frag if n_frag > 0 else 1.0
                    fn = (safe_pred & (labels == 1)).sum()
                    cleared = n_safe / len(labels)

                    if prec >= 0.95 and rec >= 0.90:
                        if best is None or cleared > best["cleared_fraction"]:
                            best = {
                                "feature": gf, "direction": direction, "threshold": t,
                                "precision_safe": prec, "recall_fragile": rec,
                                "false_negatives": int(fn), "cleared_fraction": cleared,
                                "n_cleared": int(n_safe),
                            }

        print(f"\n  Method 1: Global single-feature threshold")
        if best:
            print(f"    Feature: {best['feature']} ({best['direction']}_is_fragile)")
            print(f"    precision_safe={best['precision_safe']:.4f}  "
                  f"recall_fragile={best['recall_fragile']:.4f}  "
                  f"cleared={best['cleared_fraction']:.4f} ({best['n_cleared']}/{len(labels)})  "
                  f"FN={best['false_negatives']}")
        else:
            print(f"    NO THRESHOLD meets constraints (prec>=0.95, rec>=0.90)")


class TestMethod2GlobalMulti:
    """Global multi-feature: leave-one-out linear model on all 4 ghost features."""

    def test_global_multi(self, decision_data):
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)

        X = np.array([[r["ghost"]["te"], r["ghost"]["tr"],
                        r["ghost"]["mo"], r["ghost"]["ac"]]
                       for r in decision_data])

        probs = logistic_score(X, labels)
        result = evaluate_decision(probs, labels, min_precision=0.95, min_recall=0.90)

        print(f"\n  Method 2: Global multi-feature (LOO linear model)")
        if result:
            print(f"    precision_safe={result['precision_safe']:.4f}  "
                  f"recall_fragile={result['recall_fragile']:.4f}  "
                  f"cleared={result['cleared_fraction']:.4f} ({result['n_cleared']}/{len(labels)})  "
                  f"FN={result['false_negatives']}")
        else:
            print(f"    NO THRESHOLD meets constraints")


class TestMethod3ArchAwareMulti:
    """Architecture-aware: separate model per architecture, combined decision."""

    def test_arch_aware_multi(self, decision_data):
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)

        # Split by architecture
        archs = np.array([r["arch"] for r in decision_data])
        unique_archs = sorted(set(archs))

        # Train separate LOO models per architecture
        probs = np.zeros(len(decision_data))

        for arch in unique_archs:
            arch_mask = archs == arch
            arch_idx = np.where(arch_mask)[0]

            X_arch = np.array([[decision_data[i]["ghost"]["te"],
                                decision_data[i]["ghost"]["tr"],
                                decision_data[i]["ghost"]["mo"],
                                decision_data[i]["ghost"]["ac"]]
                               for i in arch_idx])
            y_arch = labels[arch_idx]

            arch_probs = logistic_score(X_arch, y_arch)
            probs[arch_idx] = arch_probs

        result = evaluate_decision(probs, labels, min_precision=0.95, min_recall=0.90)

        print(f"\n  Method 3: Architecture-aware multi-feature (LOO per arch)")
        if result:
            print(f"    precision_safe={result['precision_safe']:.4f}  "
                  f"recall_fragile={result['recall_fragile']:.4f}  "
                  f"cleared={result['cleared_fraction']:.4f} ({result['n_cleared']}/{len(labels)})  "
                  f"FN={result['false_negatives']}")

            # Per-architecture breakdown
            for arch in unique_archs:
                arch_mask = archs == arch
                arch_probs = probs[arch_mask]
                arch_labels = labels[arch_mask]
                arch_result = evaluate_decision(arch_probs, arch_labels,
                                                min_precision=0.90, min_recall=0.80)
                if arch_result:
                    print(f"      {arch}: cleared={arch_result['cleared_fraction']:.4f} "
                          f"({arch_result['n_cleared']}/{arch_mask.sum()})  "
                          f"prec={arch_result['precision_safe']:.3f}  "
                          f"rec={arch_result['recall_fragile']:.3f}  "
                          f"FN={arch_result['false_negatives']}")
                else:
                    print(f"      {arch}: no viable threshold (relaxed constraints)")
        else:
            print(f"    NO THRESHOLD meets constraints")


class TestReport:
    def test_comparison_report(self, decision_data):
        n = len(decision_data)
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)
        archs = np.array([r["arch"] for r in decision_data])

        # Method 1: global single
        m1_best = None
        for gf in ["te", "tr", "mo", "ac"]:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            for direction in ["above", "below"]:
                sorted_vals = np.sort(np.unique(vals))
                for i in range(len(sorted_vals) - 1):
                    t = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
                    fragile_pred = vals >= t if direction == "above" else vals <= t
                    safe_pred = ~fragile_pred
                    n_safe = safe_pred.sum()
                    if n_safe == 0:
                        continue
                    prec = (safe_pred & (labels == 0)).sum() / n_safe
                    n_frag = labels.sum()
                    rec = (fragile_pred & (labels == 1)).sum() / n_frag if n_frag > 0 else 1.0
                    fn = int((safe_pred & (labels == 1)).sum())
                    cleared = n_safe / n
                    if prec >= 0.95 and rec >= 0.90:
                        if m1_best is None or cleared > m1_best["cleared"]:
                            m1_best = {"name": f"{gf}({direction})", "prec": prec,
                                       "rec": rec, "cleared": cleared, "fn": fn}

        # Method 2: global multi
        X_all = np.array([[r["ghost"]["te"], r["ghost"]["tr"],
                           r["ghost"]["mo"], r["ghost"]["ac"]]
                          for r in decision_data])
        probs_m2 = logistic_score(X_all, labels)
        m2_best = evaluate_decision(probs_m2, labels, 0.95, 0.90)

        # Method 3: arch-aware multi
        probs_m3 = np.zeros(n)
        for arch in sorted(set(archs)):
            mask = archs == arch
            idx = np.where(mask)[0]
            X_a = np.array([[decision_data[i]["ghost"]["te"],
                             decision_data[i]["ghost"]["tr"],
                             decision_data[i]["ghost"]["mo"],
                             decision_data[i]["ghost"]["ac"]] for i in idx])
            y_a = labels[idx]
            probs_m3[idx] = logistic_score(X_a, y_a)
        m3_best = evaluate_decision(probs_m3, labels, 0.95, 0.90)

        print(f"""
{'=' * 80}
PHASE 0.17b: ARCHITECTURE-AWARE MULTI-FEATURE PRE-ROUTE — COMPARISON
{'=' * 80}

Tensors: {n}
Fragile (P75): {labels.sum()}/{n}
Safety: precision_safe >= 0.95, recall_fragile >= 0.90

--- Method Comparison ---

  {'Method':<40} {'Prec':>6} {'Rec':>6} {'Cleared':>8} {'FN':>4}
  {'-'*40} {'-'*6} {'-'*6} {'-'*8} {'-'*4}""")

        if m1_best:
            print(f"  {'1. Global single (' + m1_best['name'] + ')':<40} "
                  f"{m1_best['prec']:>6.3f} {m1_best['rec']:>6.3f} "
                  f"{m1_best['cleared']:>8.3f} {m1_best['fn']:>4}")
        else:
            print(f"  {'1. Global single':<40} {'—':>6} {'—':>6} {'FAIL':>8} {'—':>4}")

        if m2_best:
            print(f"  {'2. Global multi-feature (LOO)':<40} "
                  f"{m2_best['precision_safe']:>6.3f} {m2_best['recall_fragile']:>6.3f} "
                  f"{m2_best['cleared_fraction']:>8.3f} {m2_best['false_negatives']:>4}")
        else:
            print(f"  {'2. Global multi-feature (LOO)':<40} {'—':>6} {'—':>6} {'FAIL':>8} {'—':>4}")

        if m3_best:
            print(f"  {'3. Arch-aware multi-feature (LOO)':<40} "
                  f"{m3_best['precision_safe']:>6.3f} {m3_best['recall_fragile']:>6.3f} "
                  f"{m3_best['cleared_fraction']:>8.3f} {m3_best['false_negatives']:>4}")
        else:
            print(f"  {'3. Arch-aware multi-feature (LOO)':<40} {'—':>6} {'—':>6} {'FAIL':>8} {'—':>4}")

        # Improvement over baseline
        baseline_cleared = m1_best["cleared"] if m1_best else 0.0
        m2_cleared = m2_best["cleared_fraction"] if m2_best else 0.0
        m3_cleared = m3_best["cleared_fraction"] if m3_best else 0.0

        best_cleared = max(baseline_cleared, m2_cleared, m3_cleared)
        if best_cleared == m3_cleared and m3_best:
            winner = "arch-aware multi-feature"
        elif best_cleared == m2_cleared and m2_best:
            winner = "global multi-feature"
        else:
            winner = "global single-feature"

        print(f"""
--- Improvement ---
  Baseline (Phase 0.17 single): {baseline_cleared:.1%}
  Global multi-feature:         {m2_cleared:.1%} ({m2_cleared - baseline_cleared:+.1%})
  Arch-aware multi-feature:     {m3_cleared:.1%} ({m3_cleared - baseline_cleared:+.1%})
  Winner: {winner}

{'=' * 80}
VERDICT:
""")
        gate_30 = best_cleared >= 0.30
        gate_50 = best_cleared >= 0.50

        if gate_50:
            print(f"  PASS (50%) — {winner} clears {best_cleared:.1%} of tensors.")
            print(f"  Ghost IS a viable pre-routing primitive.")
        elif gate_30:
            print(f"  PASS (30%) — {winner} clears {best_cleared:.1%} of tensors.")
            print(f"  Ghost is a useful but not dominant pre-routing signal.")
            print(f"  Practical value: {int(best_cleared * n)} tensors skip profiling.")
        elif best_cleared > baseline_cleared:
            print(f"  IMPROVED but below 30% gate — {winner} clears {best_cleared:.1%}.")
            print(f"  Multi-feature helps but not enough for runtime primitive.")
        else:
            print(f"  NO IMPROVEMENT over single-feature baseline ({baseline_cleared:.1%}).")
            print(f"  Architecture-aware multi-feature does not help.")
        print(f"{'=' * 80}")
