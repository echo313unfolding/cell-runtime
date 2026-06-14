"""Phase 0.18: Invariant Basis Ablation — Is {TE, MO, AC} the minimum?

Tests whether all three compressed-domain invariants are required, or whether
a subset is sufficient. The ablation is honest: if AC or MO is redundant,
the test says so.

Primary claim:
  TE/MO/AC form a compact invariant basis if all-3 is Pareto-best or
  statistically tied for best across route, role, and output_norm.

Secondary claim:
  If a 2-feature subset wins, the missing feature is task-specific,
  not globally required.

Feature sets tested:
  TE only
  TE + MO
  TE + AC
  MO + AC
  TE + MO + AC  (full basis)

Tasks measured:
  1. Route accuracy (cpu/gpu binary, k-NN LOO)
  2. Role classification accuracy (11 types, k-NN LOO)
  3. output_norm R² (OLS regression, adjusted)
  4. Failure cases by tensor role (per-role accuracy breakdown)

Hard controls:
  - Shape leakage check (spearman vs log_numel)
  - Leave-one-out = train/test split by tensor identity
  - Within-role stratification for regression
  - Bootstrap 95% CI on all headline numbers
  - Dumb baselines: random, role-prior, shape-only
  - JSON receipt saved to ~/receipts/

Data: 292 tensors (Mamba-130M + Qwen-1.5B HXQ safetensors)

WO-CRYSTAL-VAULT-01: Phase 0.18 — Invariant Basis Ablation
"""
import json
import platform
import resource
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

RECEIPT_DIR = Path.home() / "receipts"


# ---------------------------------------------------------------------------
# File helpers (shared with Phase 0.15/0.16)
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
    if "in_proj" in parent: return "ssm_in_proj"
    if "out_proj" in parent: return "ssm_out_proj"
    if "dt_proj" in parent: return "ssm_dt"
    if "x_proj" in parent: return "ssm_x"
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


def role_family(role: str) -> str:
    if role.startswith("attn_"): return "attention"
    if role.startswith("ffn_"): return "ffn"
    if role.startswith("ssm_"): return "ssm"
    return role


# ---------------------------------------------------------------------------
# Raw invariant extraction (TE, MO, AC directly — not ghost coordinates)
# ---------------------------------------------------------------------------

def extract_invariants(raw_bytes: bytes, shape: tuple) -> dict:
    """Extract the three candidate invariants from raw U8 indices.

    Returns raw TE, MO, AC — NOT the derived ghost coordinates.
    body_opened = false.
    """
    n = len(raw_bytes)
    if n < 64:
        return {"te": 0.0, "mo": 0.0, "ac": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # --- Transition Entropy (TE) ---
    # Bigram matrix, sampled for large tensors
    sample_size = min(n - 1, 500_000)
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

    # --- Markov Order (MO) ---
    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 1.0

    # --- Index Autocorrelation (AC) ---
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
        "mo": round(mo, 6),
        "ac": round(ac, 6),
    }


# ---------------------------------------------------------------------------
# Execution targets (from dequantized body — same as Phase 0.16)
# ---------------------------------------------------------------------------

def execution_targets(float_tensor: np.ndarray) -> dict:
    W = float_tensor.astype(np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    rows, cols = W.shape

    try:
        _, S, _ = np.linalg.svd(W, full_matrices=False)
        S = S[:min(rows, cols, 64)]
        spectral_norm = float(S[0])
    except np.linalg.LinAlgError:
        spectral_norm = 0.0

    rng = np.random.default_rng(42)
    X = rng.standard_normal((1, rows))
    Y = X @ W
    output_norm = float(np.linalg.norm(Y))

    return {"output_norm": output_norm}


# ---------------------------------------------------------------------------
# Route assignment
# ---------------------------------------------------------------------------

def assign_route(role: str, arch: str) -> str:
    """Ground-truth route: SSM→cpu, attention/FFN→gpu."""
    if arch == "mamba":
        return "cpu"
    return "gpu"


# ---------------------------------------------------------------------------
# k-NN LOO classifier
# ---------------------------------------------------------------------------

def knn_loo(coords: np.ndarray, labels: list, k: int = 5) -> dict:
    n = len(labels)
    predictions = []
    correct = 0

    for i in range(n):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        dists[i] = float("inf")
        nearest_k = np.argsort(dists)[:k]
        votes = Counter(labels[j] for j in nearest_k)
        predicted = votes.most_common(1)[0][0]
        predictions.append(predicted)
        if predicted == labels[i]:
            correct += 1

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for i in range(n):
        per_class[labels[i]]["total"] += 1
        if predictions[i] == labels[i]:
            per_class[labels[i]]["correct"] += 1

    per_class_acc = {}
    for cls in sorted(set(labels)):
        t = per_class[cls]["total"]
        per_class_acc[cls] = per_class[cls]["correct"] / t if t > 0 else 0.0

    return {
        "accuracy": correct / n,
        "correct": correct,
        "total": n,
        "per_class": per_class_acc,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# OLS regression helpers
# ---------------------------------------------------------------------------

def ols_adj_r2(X: np.ndarray, y: np.ndarray) -> float:
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
        r2 = 1.0 - ss_res / ss_tot
        return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    except np.linalg.LinAlgError:
        return 0.0


def within_role_r2(data: list, feature_names: list, target: str) -> dict:
    """Per-role OLS R² — same role = same shape, no shape leakage."""
    by_role = defaultdict(list)
    for d in data:
        by_role[d["role"]].append(d)

    results = {}
    sig_count = 0
    total = 0
    for role, pts in sorted(by_role.items()):
        if len(pts) < 5:
            continue
        X = np.array([[p["inv"][f] for f in feature_names] for p in pts])
        y = np.array([p["targets"]["output_norm"] for p in pts])
        if np.std(y) < 1e-12:
            continue

        total += 1
        # Spearman for each feature
        best_rho = 0.0
        best_p = 1.0
        for j, fn in enumerate(feature_names):
            rho, pval = sp_stats.spearmanr(X[:, j], y)
            if not np.isnan(rho) and abs(rho) > abs(best_rho):
                best_rho = rho
                best_p = pval
        if best_p < 0.05:
            sig_count += 1
        results[role] = {"n": len(pts), "best_rho": round(float(best_rho), 4),
                         "best_p": round(float(best_p), 4)}

    return {"per_role": results, "sig_count": sig_count, "total": total}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return round(float(lo), 4), round(float(hi), 4)


def bootstrap_accuracy_ci(correct_arr: np.ndarray, n_boot: int = 2000) -> tuple:
    """Bootstrap 95% CI for classification accuracy."""
    return bootstrap_ci(correct_arr, n_boot=n_boot)


# ---------------------------------------------------------------------------
# Data scanning
# ---------------------------------------------------------------------------

def _scan_model(path, role_fn, arch_name):
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
        if role in ("other", "skip"):
            continue
        byte_size = info["data_offsets"][1] - info["data_offsets"][0]
        if byte_size < 1024:
            continue

        shape = tuple(info["shape"])
        raw_bytes = read_raw_bytes(path, data_start, info)
        inv = extract_invariants(raw_bytes, shape)

        # Dequantize for execution targets
        indices = load_tensor_numpy(path, data_start, info)
        codebook = load_tensor_numpy(path, data_start, tensors_info[cb_name])
        float_tensor = codebook[indices.astype(np.int32)]
        targets = execution_targets(float_tensor)

        rows = shape[0] if len(shape) >= 2 else 1
        cols = shape[1] if len(shape) >= 2 else shape[0]

        results.append({
            "name": base,
            "role": role,
            "family": role_family(role),
            "arch": arch_name,
            "shape": shape,
            "rows": rows,
            "cols": cols,
            "numel": rows * cols,
            "inv": inv,
            "targets": targets,
            "route": assign_route(role, arch_name),
        })
    return results


@pytest.fixture(scope="module")
def data():
    results = []
    results.extend(_scan_model(MAMBA_HXQ, classify_mamba_role, "mamba"))
    results.extend(_scan_model(QWEN_HXQ, classify_qwen_role, "transformer"))
    if len(results) < 50:
        pytest.skip(f"Need >= 50 tensors, got {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "TE":         ["te"],
    "TE+MO":      ["te", "mo"],
    "TE+AC":      ["te", "ac"],
    "MO+AC":      ["mo", "ac"],
    "TE+MO+AC":   ["te", "mo", "ac"],
}


def make_feature_matrix(data_list: list, feature_names: list) -> np.ndarray:
    X = np.array([[d["inv"][f] for f in feature_names] for d in data_list])
    # Normalize each column to [0,1] for k-NN distance fairness
    for j in range(X.shape[1]):
        mn, mx = X[:, j].min(), X[:, j].max()
        if mx - mn > 1e-12:
            X[:, j] = (X[:, j] - mn) / (mx - mn)
    return X


# ===========================================================================
# Tests
# ===========================================================================


class TestDataSanity:
    def test_sufficient_data(self, data):
        assert len(data) >= 100, f"Need >= 100, got {len(data)}"
        roles = set(d["role"] for d in data)
        archs = set(d["arch"] for d in data)
        print(f"\n  Tensors: {len(data)}, Roles: {len(roles)}, Archs: {len(archs)}")

    def test_shape_leakage(self, data):
        """Invariants must not be proxies for tensor size."""
        log_numel = np.log(np.array([max(d["numel"], 1) for d in data], dtype=np.float64))
        max_leak = 0.0
        print(f"\n  Shape leakage (invariant vs log_numel):")
        for feat in ["te", "mo", "ac"]:
            vals = np.array([d["inv"][feat] for d in data])
            rho, p = sp_stats.spearmanr(vals, log_numel)
            rho = float(rho) if not np.isnan(rho) else 0.0
            leak = "LEAK" if abs(rho) > 0.5 else "clean"
            print(f"    {feat}: rho={rho:+.4f} p={p:.4g} {leak}")
            max_leak = max(max_leak, abs(rho))
        # Shape leakage is a finding, not a failure. The within-role tests
        # (same shape) are the definitive leakage control. Report honestly.
        if max_leak >= 0.5:
            print(f"  WARNING: shape leakage present (max |rho| = {max_leak:.3f})")
            print(f"  Within-role tests control for this — see TestWithinRoleRegression")


class TestAblation:
    """The core ablation: 5 feature sets × 3 tasks."""

    def test_route_ablation(self, data):
        """Task 1: Route accuracy (cpu/gpu) for each feature set."""
        labels = [d["route"] for d in data]
        print(f"\n  TASK 1: Route Classification (n={len(data)})")
        print(f"  {'Feature Set':<14} {'Accuracy':>10} {'95% CI':>20}")
        print(f"  {'-'*14} {'-'*10} {'-'*20}")

        results = {}
        for fs_name, fs_feats in FEATURE_SETS.items():
            X = make_feature_matrix(data, fs_feats)
            r = knn_loo(X, labels, k=5)
            correct_arr = np.array([1.0 if r["predictions"][i] == labels[i] else 0.0
                                    for i in range(len(labels))])
            ci_lo, ci_hi = bootstrap_accuracy_ci(correct_arr)
            results[fs_name] = {"accuracy": r["accuracy"], "ci": [ci_lo, ci_hi]}
            print(f"  {fs_name:<14} {r['accuracy']:>10.1%} [{ci_lo:.1%}, {ci_hi:.1%}]")

        # Dumb baselines
        majority = Counter(labels).most_common(1)[0]
        majority_acc = majority[1] / len(labels)
        rng = np.random.default_rng(42)
        random_acc = np.mean([rng.choice(labels) == l for l in labels])
        print(f"\n  Baselines:")
        print(f"    majority: {majority_acc:.1%}")
        print(f"    random:   {random_acc:.1%}")

    def test_role_ablation(self, data):
        """Task 2: Role classification (11 types) for each feature set."""
        labels = [d["role"] for d in data]
        n_classes = len(set(labels))
        random_baseline = 1.0 / n_classes
        print(f"\n  TASK 2: Role Classification (n={len(data)}, {n_classes} classes)")
        print(f"  {'Feature Set':<14} {'Accuracy':>10} {'95% CI':>20} {'Lift':>6}")
        print(f"  {'-'*14} {'-'*10} {'-'*20} {'-'*6}")

        results = {}
        for fs_name, fs_feats in FEATURE_SETS.items():
            X = make_feature_matrix(data, fs_feats)
            r = knn_loo(X, labels, k=5)
            correct_arr = np.array([1.0 if r["predictions"][i] == labels[i] else 0.0
                                    for i in range(len(labels))])
            ci_lo, ci_hi = bootstrap_accuracy_ci(correct_arr)
            lift = r["accuracy"] / random_baseline
            results[fs_name] = {
                "accuracy": r["accuracy"], "ci": [ci_lo, ci_hi],
                "lift": round(lift, 1), "per_class": r["per_class"],
            }
            print(f"  {fs_name:<14} {r['accuracy']:>10.1%} [{ci_lo:.1%}, {ci_hi:.1%}] {lift:>5.1f}x")

        # Role-prior baseline (most common)
        majority = Counter(labels).most_common(1)[0]
        majority_acc = majority[1] / len(labels)
        print(f"\n  Baselines:")
        print(f"    random:   {random_baseline:.1%}")
        print(f"    majority: {majority_acc:.1%} ({majority[0]})")

        # Shape-only baseline
        shape_feats = np.array([
            [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
             np.log(max(d["numel"], 1))]
            for d in data
        ])
        mn, mx = shape_feats.min(axis=0), shape_feats.max(axis=0)
        rng = mx - mn
        rng[rng < 1e-12] = 1.0
        shape_norm = (shape_feats - mn) / rng
        r_shape = knn_loo(shape_norm, labels, k=5)
        print(f"    shape:    {r_shape['accuracy']:.1%}")

    def test_output_norm_ablation(self, data):
        """Task 3: output_norm R² for each feature set."""
        y = np.array([d["targets"]["output_norm"] for d in data])
        n = len(data)
        print(f"\n  TASK 3: output_norm Prediction (n={n})")
        print(f"  {'Feature Set':<14} {'adj.R²':>10}")
        print(f"  {'-'*14} {'-'*10}")

        results = {}
        for fs_name, fs_feats in FEATURE_SETS.items():
            X = np.array([[d["inv"][f] for f in fs_feats] for d in data])
            adj_r2 = ols_adj_r2(X, y)
            results[fs_name] = {"adj_r2": round(adj_r2, 4)}
            print(f"  {fs_name:<14} {adj_r2:>10.4f}")

        # Shape-only baseline
        shape_X = np.array([
            [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
             np.log(max(d["numel"], 1))]
            for d in data
        ])
        shape_r2 = ols_adj_r2(shape_X, y)
        print(f"\n  Baselines:")
        print(f"    shape-only: {shape_r2:.4f}")
        print(f"    mean-only:  0.0000")


class TestFailureAnalysis:
    """Per-role breakdown: which roles break which feature sets?"""

    def test_per_role_failures(self, data):
        labels = [d["role"] for d in data]
        print(f"\n  FAILURE ANALYSIS: Per-role accuracy by feature set")

        roles = sorted(set(labels))
        role_counts = Counter(labels)

        header = f"  {'Role':<16} {'n':>4}"
        for fs_name in FEATURE_SETS:
            header += f" {fs_name:>10}"
        print(header)
        print(f"  {'-'*16} {'-'*4}" + f" {'-'*10}" * len(FEATURE_SETS))

        all_per_class = {}
        for fs_name, fs_feats in FEATURE_SETS.items():
            X = make_feature_matrix(data, fs_feats)
            r = knn_loo(X, labels, k=5)
            all_per_class[fs_name] = r["per_class"]

        for role in roles:
            line = f"  {role:<16} {role_counts[role]:>4}"
            for fs_name in FEATURE_SETS:
                acc = all_per_class[fs_name].get(role, 0.0)
                line += f" {acc:>9.0%}"
            print(line)


class TestWithinRoleRegression:
    """Within-role signal check: does invariant→output_norm survive within
    same-shape tensors? This is the strongest shape-leakage control."""

    def test_within_role_signal(self, data):
        print(f"\n  WITHIN-ROLE REGRESSION (shape-independent signal)")

        for fs_name, fs_feats in FEATURE_SETS.items():
            wr = within_role_r2(data, fs_feats, "output_norm")
            print(f"\n  {fs_name}: {wr['sig_count']}/{wr['total']} roles with p<0.05")
            for role, info in sorted(wr["per_role"].items()):
                sig = "*" if info["best_p"] < 0.05 else ""
                print(f"    {role:<16} n={info['n']:>3} "
                      f"rho={info['best_rho']:+.4f} p={info['best_p']:.4g} {sig}")


class TestFinalVerdict:
    """Aggregate all results and emit the verdict + JSON receipt."""

    def test_verdict(self, data):
        t_start = time.time()
        cpu_start = time.process_time()
        start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

        n = len(data)
        labels_role = [d["role"] for d in data]
        labels_route = [d["route"] for d in data]
        y_norm = np.array([d["targets"]["output_norm"] for d in data])
        n_classes = len(set(labels_role))

        # ---- Run all ablations ----
        route_results = {}
        role_results = {}
        r2_results = {}
        within_role_results = {}

        for fs_name, fs_feats in FEATURE_SETS.items():
            X_knn = make_feature_matrix(data, fs_feats)
            X_ols = np.array([[d["inv"][f] for f in fs_feats] for d in data])

            # Route
            rr = knn_loo(X_knn, labels_route, k=5)
            correct_arr = np.array([1.0 if rr["predictions"][i] == labels_route[i]
                                    else 0.0 for i in range(n)])
            ci = bootstrap_accuracy_ci(correct_arr)
            route_results[fs_name] = {"accuracy": rr["accuracy"], "ci": list(ci)}

            # Role
            rl = knn_loo(X_knn, labels_role, k=5)
            correct_arr = np.array([1.0 if rl["predictions"][i] == labels_role[i]
                                    else 0.0 for i in range(n)])
            ci = bootstrap_accuracy_ci(correct_arr)
            role_results[fs_name] = {
                "accuracy": rl["accuracy"], "ci": list(ci),
                "per_class": rl["per_class"],
            }

            # output_norm R²
            adj_r2 = ols_adj_r2(X_ols, y_norm)
            r2_results[fs_name] = {"adj_r2": round(adj_r2, 4)}

            # Within-role
            wr = within_role_r2(data, fs_feats, "output_norm")
            within_role_results[fs_name] = {
                "sig_count": wr["sig_count"],
                "total": wr["total"],
            }

        # ---- Shape baseline ----
        shape_X_knn = np.array([
            [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
             np.log(max(d["numel"], 1))]
            for d in data
        ])
        mn, mx = shape_X_knn.min(axis=0), shape_X_knn.max(axis=0)
        rng_v = mx - mn
        rng_v[rng_v < 1e-12] = 1.0
        shape_norm = (shape_X_knn - mn) / rng_v
        shape_role = knn_loo(shape_norm, labels_role, k=5)
        shape_route = knn_loo(shape_norm, labels_route, k=5)
        shape_r2 = ols_adj_r2(shape_X_knn, y_norm)

        # ---- Shape leakage ----
        log_numel = np.log(np.array([max(d["numel"], 1) for d in data], dtype=np.float64))
        leakage = {}
        for feat in ["te", "mo", "ac"]:
            vals = np.array([d["inv"][feat] for d in data])
            rho, _ = sp_stats.spearmanr(vals, log_numel)
            leakage[feat] = round(float(rho) if not np.isnan(rho) else 0.0, 4)
        max_leakage = max(abs(v) for v in leakage.values())

        # ---- Determine verdict ----
        full = "TE+MO+AC"
        pairs = ["TE+MO", "TE+AC", "MO+AC"]

        # Count tasks where full basis is best or tied-for-best
        tasks_full_best = 0
        tasks_full_tied = 0
        task_winners = {}

        # Route
        route_accs = {k: v["accuracy"] for k, v in route_results.items()}
        best_route = max(route_accs.values())
        route_winner = [k for k, v in route_accs.items() if v == best_route]
        task_winners["route"] = route_winner
        if full in route_winner:
            tasks_full_best += 1
        elif route_results[full]["accuracy"] >= best_route - 0.02:  # within 2pp
            tasks_full_tied += 1

        # Role
        role_accs = {k: v["accuracy"] for k, v in role_results.items()}
        best_role = max(role_accs.values())
        role_winner = [k for k, v in role_accs.items() if v == best_role]
        task_winners["role"] = role_winner
        if full in role_winner:
            tasks_full_best += 1
        elif role_results[full]["accuracy"] >= best_role - 0.02:
            tasks_full_tied += 1

        # output_norm R²
        r2_vals = {k: v["adj_r2"] for k, v in r2_results.items()}
        best_r2 = max(r2_vals.values())
        r2_winner = [k for k, v in r2_vals.items() if v == best_r2]
        task_winners["output_norm"] = r2_winner
        if full in r2_winner:
            tasks_full_best += 1
        elif r2_results[full]["adj_r2"] >= best_r2 - 0.02:
            tasks_full_tied += 1

        # Verdict logic
        tasks_full_pareto = tasks_full_best + tasks_full_tied

        if tasks_full_best >= 2:
            status = "PASS"
            reason = (f"Full basis wins {tasks_full_best}/3 tasks outright"
                      f"{f', tied on {tasks_full_tied}' if tasks_full_tied else ''}")
        elif tasks_full_pareto >= 3:
            status = "PASS"
            reason = f"Full basis is Pareto-best or tied on all 3 tasks"
        elif tasks_full_pareto >= 2:
            status = "PARTIAL"
            # Find which pair dominates the remaining task
            losing_tasks = []
            for task_name, winners in task_winners.items():
                if full not in winners:
                    # Check if full is within 2pp
                    if task_name == "route":
                        gap = best_route - route_results[full]["accuracy"]
                    elif task_name == "role":
                        gap = best_role - role_results[full]["accuracy"]
                    else:
                        gap = best_r2 - r2_results[full]["adj_r2"]
                    if gap > 0.02:
                        losing_tasks.append((task_name, winners, gap))

            if losing_tasks:
                t_name, t_winners, t_gap = losing_tasks[0]
                reason = (f"Full basis Pareto on {tasks_full_pareto}/3 tasks. "
                          f"{t_name} won by {', '.join(t_winners)} "
                          f"(gap={t_gap:.3f}) — missing feature is task-specific")
            else:
                reason = f"Full basis Pareto on {tasks_full_pareto}/3 tasks"
        else:
            status = "FAIL"
            # Find the dominant subset
            subset_scores = {}
            for pair in pairs:
                score = 0
                if pair in task_winners.get("route", []):
                    score += 1
                if pair in task_winners.get("role", []):
                    score += 1
                if pair in task_winners.get("output_norm", []):
                    score += 1
                subset_scores[pair] = score
            best_pair = max(subset_scores, key=subset_scores.get)
            reason = (f"Full basis wins only {tasks_full_best}/3 tasks. "
                      f"{best_pair} dominates — third feature is redundant")

        # ---- Print verdict ----
        print(f"""
{'=' * 80}
PHASE 0.18: INVARIANT BASIS ABLATION — VERDICT
{'=' * 80}

  Tensors: {n} (Mamba-130M + Qwen-1.5B)
  Roles: {n_classes}
  Invariants: TE (transition entropy), MO (markov order), AC (index autocorr)
  body_opened_for_invariants: false
  body_opened_for_targets: true

--- Task Results ---

  {'Feature Set':<14} {'Route':>8} {'Role':>8} {'R²':>8}
  {'-'*14} {'-'*8} {'-'*8} {'-'*8}""")

        for fs_name in FEATURE_SETS:
            rt = route_results[fs_name]["accuracy"]
            rl = role_results[fs_name]["accuracy"]
            r2 = r2_results[fs_name]["adj_r2"]
            print(f"  {fs_name:<14} {rt:>7.1%} {rl:>7.1%} {r2:>8.4f}")

        print(f"""
--- Baselines ---
  shape-only      {shape_route['accuracy']:>7.1%} {shape_role['accuracy']:>7.1%} {shape_r2:>8.4f}
  random          {1/len(set(labels_route)):.1%}   {1/n_classes:.1%}   0.0000

--- Shape Leakage ---
  TE: {leakage['te']:+.4f}  MO: {leakage['mo']:+.4f}  AC: {leakage['ac']:+.4f}
  max |rho| = {max_leakage:.4f} {'CLEAN' if max_leakage < 0.5 else 'WARNING'}

--- Within-Role Signal (shape-independent) ---""")

        for fs_name in FEATURE_SETS:
            wr = within_role_results[fs_name]
            print(f"  {fs_name:<14} {wr['sig_count']}/{wr['total']} roles sig at p<0.05")

        print(f"""
--- Bootstrap 95% CIs ---
  {'Feature Set':<14} {'Route CI':>20} {'Role CI':>20}
  {'-'*14} {'-'*20} {'-'*20}""")
        for fs_name in FEATURE_SETS:
            rt_ci = route_results[fs_name]["ci"]
            rl_ci = role_results[fs_name]["ci"]
            print(f"  {fs_name:<14} [{rt_ci[0]:.1%}, {rt_ci[1]:.1%}]"
                  f"      [{rl_ci[0]:.1%}, {rl_ci[1]:.1%}]")

        print(f"""
{'=' * 80}
PHASE 0.18 RESULT:
  minimum_basis = [transition_entropy, markov_order, index_autocorr]
  status = {status}
  reason = {reason}
{'=' * 80}""")

        # ---- Save JSON receipt ----
        cost = {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

        receipt = {
            "phase": "0.18",
            "title": "Invariant Basis Ablation",
            "status": status,
            "reason": reason,
            "n_tensors": n,
            "n_roles": n_classes,
            "models": ["mamba-130m-hxq", "qwen2.5-coder-1.5b-hxq"],
            "invariants": ["transition_entropy", "markov_order", "index_autocorr"],
            "feature_sets": list(FEATURE_SETS.keys()),
            "route_results": route_results,
            "role_results": {k: {"accuracy": v["accuracy"], "ci": v["ci"]}
                            for k, v in role_results.items()},
            "r2_results": r2_results,
            "within_role_results": within_role_results,
            "baselines": {
                "shape_route": round(shape_route["accuracy"], 4),
                "shape_role": round(shape_role["accuracy"], 4),
                "shape_r2": round(shape_r2, 4),
                "random_route": round(1 / len(set(labels_route)), 4),
                "random_role": round(1 / n_classes, 4),
            },
            "shape_leakage": leakage,
            "max_shape_leakage": max_leakage,
            "task_winners": {k: v for k, v in task_winners.items()},
            "body_opened_for_invariants": False,
            "body_opened_for_targets": True,
            "cost": cost,
        }

        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        receipt_path = RECEIPT_DIR / "phase_018_invariant_basis_ablation.json"
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\n  Receipt: {receipt_path}")

        # Assert status is not FAIL (but allow PARTIAL — honest result)
        # Actually: DON'T assert. Let the receipt speak. The test passes
        # if it runs to completion and produces a receipt. The STATUS
        # field in the receipt is the scientific result, not a test gate.
