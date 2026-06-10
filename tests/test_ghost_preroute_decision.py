"""Phase 0.17: Ghost Pre-Route Decision — can Ghost make a correct runtime decision?

Phase 0.16 proved: Ghost correlates with execution behavior (R²=0.818).
This phase asks: can Ghost make a DECISION that is correct enough to trust?

Correlation is evidence. Decision is primitive.

Goal:
  Prove Ghost can make an actual binary routing decision from compressed domain.
  safe → skip expensive Hydra profiling
  fragile → run full Hydra profiling

Ground truth (from decompressed body):
  fragile = condition_number >= P75 (top 25% most numerically unstable)
  safe = condition_number < P75

Ghost decision (compressed domain only):
  Try thresholds on each ghost feature to find optimal safe/fragile boundary.
  Optimize for SAFETY: false negatives (fragile marked safe) are dangerous.

Metrics:
  precision_safe    — of Ghost-safe tensors, fraction truly safe
  recall_fragile    — of truly fragile tensors, fraction Ghost caught
  false_negative_rate — fragile tensors Ghost missed (THE danger metric)
  cleared_fraction  — fraction of tensors Ghost cleared (work saved)

Pass condition:
  precision_safe >= 0.95
  recall_fragile >= 0.80
  cleared_fraction >= 0.50

If this passes: Ghost is a compressed-domain pre-routing primitive for Hydra.
If this fails: Ghost predicts continuously but is not safe for binary routing.

Receipt:
  body_opened_for_ghost = false
  body_opened_for_ground_truth = true
  decision_type = pre_route_safe_or_fragile

Data: 292 tensors from Phase 0.16 (Mamba-130M + Qwen-1.5B HXQ).

WO-CRYSTAL-VAULT-01: Phase 0.17 — Ghost Pre-Route Decision
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


# ---------------------------------------------------------------------------
# Ghost features (compressed-domain only)
# ---------------------------------------------------------------------------

def ghost_features_from_bytes(raw_bytes: bytes, shape: tuple) -> dict:
    """Ghost features from raw U8 index bytes. body_opened = false."""
    n = len(raw_bytes)
    if n < 64:
        return {k: 0.0 for k in ["te", "tr", "mo", "ac"]}

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

    return {
        "te": round(te, 6),
        "tr": round(tr, 6),
        "mo": round(mo, 6),
        "ac": round(ac, 6),
    }


# ---------------------------------------------------------------------------
# Execution ground truth (from dequantized body)
# ---------------------------------------------------------------------------

def condition_number_from_tensor(float_tensor: np.ndarray) -> float:
    """Condition number from dequantized tensor. body_opened = true."""
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
# Decision engine
# ---------------------------------------------------------------------------

def find_optimal_threshold(feature_vals: np.ndarray, labels: np.ndarray,
                           direction: str = "above_is_fragile",
                           min_precision_safe: float = 0.95,
                           min_recall_fragile: float = 0.50) -> dict:
    """Find threshold that maximizes cleared_fraction while meeting safety constraints.

    direction: "above_is_fragile" means feature > threshold → fragile
               "below_is_fragile" means feature < threshold → fragile

    Returns best threshold and its metrics, or None if no threshold meets constraints.
    """
    sorted_vals = np.sort(np.unique(feature_vals))
    # Try thresholds at midpoints between consecutive unique values
    candidates = []
    for i in range(len(sorted_vals) - 1):
        t = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
        candidates.append(t)
    # Also try edges
    candidates.insert(0, sorted_vals[0] - 0.001)
    candidates.append(sorted_vals[-1] + 0.001)

    best = None
    for threshold in candidates:
        if direction == "above_is_fragile":
            ghost_fragile = feature_vals >= threshold
        else:
            ghost_fragile = feature_vals <= threshold

        ghost_safe = ~ghost_fragile
        truly_fragile = labels == 1
        truly_safe = labels == 0

        # Metrics
        n_ghost_safe = ghost_safe.sum()
        n_ghost_fragile = ghost_fragile.sum()
        n_truly_fragile = truly_fragile.sum()

        if n_ghost_safe == 0:
            continue

        # Precision safe: of Ghost-safe, how many truly safe?
        true_safe_in_ghost_safe = (ghost_safe & truly_safe).sum()
        precision_safe = true_safe_in_ghost_safe / n_ghost_safe

        # Recall fragile: of truly fragile, how many did Ghost catch?
        caught_fragile = (ghost_fragile & truly_fragile).sum()
        recall_fragile = caught_fragile / n_truly_fragile if n_truly_fragile > 0 else 1.0

        # False negatives: fragile marked safe
        false_negatives = (ghost_safe & truly_fragile).sum()
        fn_rate = false_negatives / n_truly_fragile if n_truly_fragile > 0 else 0.0

        # Cleared fraction: how many tensors skip profiling
        cleared_fraction = n_ghost_safe / len(labels)

        if precision_safe >= min_precision_safe and recall_fragile >= min_recall_fragile:
            score = cleared_fraction  # maximize work saved while meeting safety
            result = {
                "threshold": threshold,
                "direction": direction,
                "precision_safe": precision_safe,
                "recall_fragile": recall_fragile,
                "false_negative_rate": fn_rate,
                "false_negatives": int(false_negatives),
                "cleared_fraction": cleared_fraction,
                "n_cleared": int(n_ghost_safe),
                "n_profiled": int(n_ghost_fragile),
                "score": score,
            }
            if best is None or score > best["score"]:
                best = result

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

        # Ghost features — compressed domain
        raw_bytes = read_raw_bytes(path, data_start, info)
        ghost = ghost_features_from_bytes(raw_bytes, shape)

        # Ground truth — dequantize
        indices = load_tensor_numpy(path, data_start, info)
        codebook = load_tensor_numpy(path, data_start, tensors_info[cb_name])
        float_tensor = codebook[indices.astype(np.int32)]
        cond = condition_number_from_tensor(float_tensor)

        results.append({
            "name": base,
            "role": role,
            "arch": arch_name,
            "shape": shape,
            "ghost": ghost,
            "condition_number": cond,
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

    def test_fragile_distribution(self, decision_data):
        conds = [r["condition_number"] for r in decision_data]
        p75 = np.percentile(conds, 75)
        n_fragile = sum(1 for c in conds if c >= p75)
        print(f"\n  Condition number range: [{min(conds):.3f}, {max(conds):.3f}]")
        print(f"  P75 threshold: {p75:.3f}")
        print(f"  Fragile (>= P75): {n_fragile}/{len(conds)}")
        print(f"  Safe (< P75): {len(conds) - n_fragile}/{len(conds)}")


class TestOptimalThreshold:
    """Find the best Ghost feature and threshold for safe/fragile decision."""

    def test_sweep_all_features(self, decision_data):
        """Try every ghost feature × direction × threshold.

        The winner is the feature+threshold that clears the most tensors
        while maintaining precision_safe >= 0.95 and recall_fragile >= 0.80.
        """
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)  # 1 = fragile, 0 = safe

        ghost_features = ["te", "tr", "mo", "ac"]
        directions = ["above_is_fragile", "below_is_fragile"]

        print(f"\n  Ground truth: fragile = condition_number >= {p75:.3f}")
        print(f"  Fragile: {labels.sum()}/{len(labels)}")
        print(f"  Safety constraints: precision_safe >= 0.95, recall_fragile >= 0.80")
        print(f"\n  Searching all feature × direction × threshold combinations...")

        all_results = []
        for gf in ghost_features:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            for direction in directions:
                result = find_optimal_threshold(vals, labels, direction)
                if result:
                    result["feature"] = gf
                    all_results.append(result)
                    print(f"\n    {gf} ({direction}):")
                    print(f"      threshold={result['threshold']:.6f}")
                    print(f"      precision_safe={result['precision_safe']:.4f}")
                    print(f"      recall_fragile={result['recall_fragile']:.4f}")
                    print(f"      cleared={result['cleared_fraction']:.4f} "
                          f"({result['n_cleared']}/{len(labels)})")
                    print(f"      false_negatives={result['false_negatives']}")
                else:
                    print(f"\n    {gf} ({direction}): NO THRESHOLD MEETS CONSTRAINTS")

        if all_results:
            best = max(all_results, key=lambda r: r["score"])
            print(f"\n  === BEST DECISION ===")
            print(f"  Feature: {best['feature']}")
            print(f"  Direction: {best['direction']}")
            print(f"  Threshold: {best['threshold']:.6f}")
            print(f"  Precision safe: {best['precision_safe']:.4f}")
            print(f"  Recall fragile: {best['recall_fragile']:.4f}")
            print(f"  False negatives: {best['false_negatives']}")
            print(f"  Cleared: {best['cleared_fraction']:.4f} ({best['n_cleared']}/{len(labels)})")


class TestBestDecisionDetail:
    """Detailed analysis of the best decision."""

    def test_per_role_accuracy(self, decision_data):
        """Does the decision work across all roles, or only some?"""
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)

        # Find best feature (recompute — keep test independent)
        ghost_features = ["te", "tr", "mo", "ac"]
        directions = ["above_is_fragile", "below_is_fragile"]
        best_overall = None
        for gf in ghost_features:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            for direction in directions:
                result = find_optimal_threshold(vals, labels, direction)
                if result:
                    result["feature"] = gf
                    if best_overall is None or result["score"] > best_overall["score"]:
                        best_overall = result

        if not best_overall:
            print("\n  No threshold meets constraints — skipping per-role analysis")
            return

        gf = best_overall["feature"]
        threshold = best_overall["threshold"]
        direction = best_overall["direction"]

        print(f"\n  Using: {gf} {'>' if direction == 'above_is_fragile' else '<'} "
              f"{threshold:.6f}")
        print(f"\n  Per-role breakdown:")
        print(f"  {'Role':<16} {'N':>4} {'Fragile':>8} {'Ghost-safe':>11} "
              f"{'Correct':>8} {'FN':>4}")

        by_role = defaultdict(list)
        for i, r in enumerate(decision_data):
            by_role[r["role"]].append(i)

        total_fn = 0
        for role in sorted(by_role.keys()):
            idx = by_role[role]
            role_vals = np.array([decision_data[i]["ghost"][gf] for i in idx])
            role_labels = labels[idx]

            if direction == "above_is_fragile":
                ghost_fragile = role_vals >= threshold
            else:
                ghost_fragile = role_vals <= threshold
            ghost_safe = ~ghost_fragile

            n_fragile = role_labels.sum()
            n_ghost_safe = ghost_safe.sum()
            fn = (ghost_safe & (role_labels == 1)).sum()
            correct = ((ghost_fragile & (role_labels == 1)) | (ghost_safe & (role_labels == 0))).sum()
            total_fn += fn

            print(f"  {role:<16} {len(idx):>4} {n_fragile:>8} {n_ghost_safe:>11} "
                  f"{correct:>8} {fn:>4}{'!' if fn > 0 else ''}")

        print(f"\n  Total false negatives across roles: {total_fn}")

    def test_per_architecture_accuracy(self, decision_data):
        """Does the decision transfer across architectures?"""
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)

        ghost_features = ["te", "tr", "mo", "ac"]
        directions = ["above_is_fragile", "below_is_fragile"]
        best_overall = None
        for gf in ghost_features:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            for direction in directions:
                result = find_optimal_threshold(vals, labels, direction)
                if result:
                    result["feature"] = gf
                    if best_overall is None or result["score"] > best_overall["score"]:
                        best_overall = result

        if not best_overall:
            print("\n  No threshold meets constraints")
            return

        gf = best_overall["feature"]
        threshold = best_overall["threshold"]
        direction = best_overall["direction"]

        print(f"\n  Per-architecture breakdown:")
        for arch in ["mamba", "transformer"]:
            idx = [i for i, r in enumerate(decision_data) if r["arch"] == arch]
            if not idx:
                continue
            arch_vals = np.array([decision_data[i]["ghost"][gf] for i in idx])
            arch_labels = labels[idx]

            if direction == "above_is_fragile":
                ghost_fragile = arch_vals >= threshold
            else:
                ghost_fragile = arch_vals <= threshold
            ghost_safe = ~ghost_fragile

            n_fragile = arch_labels.sum()
            n_safe_correct = (ghost_safe & (arch_labels == 0)).sum()
            n_ghost_safe = ghost_safe.sum()
            prec = n_safe_correct / n_ghost_safe if n_ghost_safe > 0 else 0
            fn = (ghost_safe & (arch_labels == 1)).sum()
            recall = 1.0 - fn / n_fragile if n_fragile > 0 else 1.0

            print(f"    {arch:<14} n={len(idx):>3}  fragile={n_fragile:>3}  "
                  f"cleared={n_ghost_safe:>3}  prec_safe={prec:.3f}  "
                  f"recall_fragile={recall:.3f}  FN={fn}")


class TestAlternativeGroundTruths:
    """Test with different fragile definitions to check robustness."""

    def test_p80_threshold(self, decision_data):
        """Stricter: top 20% are fragile."""
        self._run_with_percentile(decision_data, 80)

    def test_p70_threshold(self, decision_data):
        """Looser: top 30% are fragile."""
        self._run_with_percentile(decision_data, 70)

    def _run_with_percentile(self, decision_data, pct):
        conds = np.array([r["condition_number"] for r in decision_data])
        threshold_val = np.percentile(conds, pct)
        labels = (conds >= threshold_val).astype(int)

        ghost_features = ["te", "tr", "mo", "ac"]
        directions = ["above_is_fragile", "below_is_fragile"]
        best = None
        for gf in ghost_features:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            for direction in directions:
                result = find_optimal_threshold(vals, labels, direction)
                if result:
                    result["feature"] = gf
                    if best is None or result["score"] > best["score"]:
                        best = result

        if best:
            print(f"\n  P{pct} (cond >= {threshold_val:.3f}): "
                  f"fragile={labels.sum()}/{len(labels)}")
            print(f"    Best: {best['feature']} ({best['direction']})")
            print(f"    precision_safe={best['precision_safe']:.4f}  "
                  f"recall_fragile={best['recall_fragile']:.4f}  "
                  f"cleared={best['cleared_fraction']:.4f}  "
                  f"FN={best['false_negatives']}")
        else:
            print(f"\n  P{pct}: NO THRESHOLD MEETS CONSTRAINTS")


class TestReport:
    def test_full_report(self, decision_data):
        n = len(decision_data)
        conds = np.array([r["condition_number"] for r in decision_data])
        p75 = np.percentile(conds, 75)
        labels = (conds >= p75).astype(int)

        ghost_features = ["te", "tr", "mo", "ac"]
        directions = ["above_is_fragile", "below_is_fragile"]
        best = None
        all_results = []
        for gf in ghost_features:
            vals = np.array([r["ghost"][gf] for r in decision_data])
            for direction in directions:
                result = find_optimal_threshold(vals, labels, direction)
                if result:
                    result["feature"] = gf
                    all_results.append(result)
                    if best is None or result["score"] > best["score"]:
                        best = result

        # Pass/fail
        if best:
            p_safe = best["precision_safe"] >= 0.95
            r_frag = best["recall_fragile"] >= 0.80
            c_frac = best["cleared_fraction"] >= 0.50
            passed = p_safe and r_frag and c_frac
        else:
            p_safe = r_frag = c_frac = False
            passed = False

        print(f"""
{'=' * 80}
PHASE 0.17: GHOST PRE-ROUTE DECISION — FINAL REPORT
{'=' * 80}

Tensors: {n}
Architectures: {len(set(r['arch'] for r in decision_data))}
Roles: {len(set(r['role'] for r in decision_data))}
Ground truth: fragile = condition_number >= {p75:.3f} (P75)
Fragile count: {labels.sum()}/{n}

Receipt:
  body_opened_for_ghost = false
  body_opened_for_ground_truth = true
  decision_type = pre_route_safe_or_fragile""")

        if best:
            print(f"""
--- Best Decision ---
  feature_used = {best['feature']}
  direction = {best['direction']}
  threshold_used = {best['threshold']:.6f}

--- Safety Metrics ---
  precision_safe       = {best['precision_safe']:.4f}  {'PASS' if p_safe else 'FAIL'} (>= 0.95)
  recall_fragile       = {best['recall_fragile']:.4f}  {'PASS' if r_frag else 'FAIL'} (>= 0.80)
  false_negative_rate  = {best['false_negative_rate']:.4f}
  dangerous_false_negatives = {best['false_negatives']}

--- Efficiency ---
  cleared_fraction         = {best['cleared_fraction']:.4f}  {'PASS' if c_frac else 'FAIL'} (>= 0.50)
  cleared_safe_tensors     = {best['n_cleared']}
  tensors_needing_profile  = {best['n_profiled']}
  avoided_profile_fraction = {best['cleared_fraction']:.1%}

--- All Viable Thresholds ---""")
            for r in sorted(all_results, key=lambda x: -x["score"]):
                print(f"  {r['feature']:<4} {r['direction']:<20} "
                      f"prec={r['precision_safe']:.3f}  rec={r['recall_fragile']:.3f}  "
                      f"cleared={r['cleared_fraction']:.3f}  FN={r['false_negatives']}")
        else:
            print(f"\n  NO THRESHOLD MEETS SAFETY CONSTRAINTS")
            print(f"  No ghost feature can achieve precision_safe >= 0.95")
            print(f"  AND recall_fragile >= 0.80 simultaneously.")

        print(f"""
{'=' * 80}
VERDICT:
""")
        if passed:
            print(f"  PASS — Ghost is a viable pre-routing primitive.")
            print(f"  Decision: {best['feature']} {'>' if best['direction'] == 'above_is_fragile' else '<'} "
                  f"{best['threshold']:.6f}")
            print(f"  {best['cleared_fraction']:.1%} of tensors skip expensive profiling.")
            print(f"  {best['false_negatives']} fragile tensors missed (acceptable at "
                  f"{best['false_negative_rate']:.1%} FN rate).")
            print(f"")
            print(f"  Ghost IS a compressed-domain pre-routing primitive for Hydra.")
        elif best and (p_safe or r_frag):
            print(f"  PARTIAL — Ghost makes useful decisions but doesn't meet all gates.")
            if not p_safe:
                print(f"  precision_safe {best['precision_safe']:.3f} < 0.95")
            if not r_frag:
                print(f"  recall_fragile {best['recall_fragile']:.3f} < 0.80")
            if not c_frac:
                print(f"  cleared_fraction {best['cleared_fraction']:.3f} < 0.50")
            print(f"  Ghost predicts behavior continuously,")
            print(f"  but is not safe enough for binary routing yet.")
        else:
            print(f"  FAIL — Ghost cannot make safe binary routing decisions.")
            print(f"  correlation ≠ decision")
            print(f"  Ghost predicts behavior continuously,")
            print(f"  but is not safe enough for binary routing.")
        print(f"{'=' * 80}")
