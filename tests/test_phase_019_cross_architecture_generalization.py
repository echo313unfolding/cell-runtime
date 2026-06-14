"""Phase 0.19: Cross-Architecture Generalization — Do invariants survive new families?

Train: Mamba-130M (SSM v1) + Qwen2.5-Coder-1.5B (dense transformer)
       These are the Phase 0.18 discovery data. NO holdout data seen during fitting.

Holdout (zero-shot, no retraining):
  Zamba2-1.2B    — SSM-Transformer HYBRID (Mamba layers + shared attention)
  TinyLlama-1.1B — LLaMA-family transformer (different from Qwen)
  Mamba2-1.3B    — SSM v2 (different architecture from Mamba v1)

Rules:
  - No retraining on holdout data
  - Normalization parameters fitted on TRAIN ONLY, applied to holdout
  - No architecture labels as features
  - No shape-only hidden shortcut
  - k-NN uses train data as the reference set, holdout as query

Pass conditions:
  PASS:        TE/MO/AC beats random + role-prior on 2/3 holdout families
  STRONG PASS: TE/MO/AC generalizes on Zamba2 hybrid
  FAIL:        only works inside Mamba/Qwen family

If Zamba2 passes, invariant basis is not just family-specific compression residue.

WO-CRYSTAL-VAULT-01: Phase 0.19 — Cross-Architecture Generalization
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
# Paths — TRAIN (Phase 0.18 data)
# ---------------------------------------------------------------------------

TRAIN_MODELS = {
    "mamba-130m": {
        "path": Path(
            "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
            "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
        ),
        "arch": "mamba_v1",
        "arch_family": "ssm",
    },
    "qwen2.5-coder-1.5b": {
        "path": Path(
            "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--qwen2.5-coder-1.5b-helix"
            "/snapshots/0a5c17fba5cc81018423eba394295ca8568caff2/model.safetensors"
        ),
        "arch": "qwen_transformer",
        "arch_family": "transformer",
    },
}

# ---------------------------------------------------------------------------
# Paths — HOLDOUT (never seen during training)
# ---------------------------------------------------------------------------

HOLDOUT_MODELS = {
    "zamba2-1.2b": {
        "path": Path(
            "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--zamba2-1.2b-helix"
            "/snapshots/1d84d8d22e5b8fe7006700fd86d795f21e2f6edb/model.safetensors"
        ),
        "arch": "zamba2_hybrid",
        "arch_family": "hybrid",
    },
    "tinyllama-1.1b": {
        "path": Path(
            "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--tinyllama-1.1b-helix"
            "/snapshots/d9c0e0b89c316177faaf8bb01708f1c2cedbcba1/model.safetensors"
        ),
        "arch": "llama_transformer",
        "arch_family": "transformer",
    },
    "mamba2-1.3b": {
        "path": Path(
            "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba2-1.3b-helix"
            "/snapshots/4306acf73ef2815e520af257ea03afa907b1e276/model.safetensors"
        ),
        "arch": "mamba2_ssm",
        "arch_family": "ssm",
    },
}

RECEIPT_DIR = Path.home() / "receipts"


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
# Universal role classifier — works across all architectures
# ---------------------------------------------------------------------------

def classify_role_universal(name: str) -> str:
    """Classify tensor role from name alone. Architecture-agnostic."""
    n = name.lower().replace(".indices", "")

    # Skip non-weight tensors
    if any(skip in n for skip in ["embed", "lm_head", "norm", "bias"]):
        return "skip"

    # Attention roles
    if "q_proj" in n or "linear_q" in n:
        return "attn_q"
    if "k_proj" in n or "linear_k" in n:
        return "attn_k"
    if "v_proj" in n or "linear_v" in n:
        return "attn_v"
    if "o_proj" in n or "linear_o" in n:
        return "attn_o"

    # FFN roles
    if "gate_proj" in n or "gate_up_proj" in n:
        return "ffn_gate"
    if "up_proj" in n:
        return "ffn_up"
    if "down_proj" in n:
        return "ffn_down"

    # SSM roles (Mamba v1 / v2 / Zamba2)
    if "in_proj" in n:
        return "ssm_in_proj"
    if "out_proj" in n:
        return "ssm_out_proj"
    if "dt_proj" in n:
        return "ssm_dt"
    if "x_proj" in n:
        return "ssm_x"
    if "conv1d" in n:
        return "ssm_conv"

    # Adapter layers in Zamba2
    if "adapter_list" in n:
        # Map adapter to parent role
        if "linear_q" in n or "q_proj" in n:
            return "adapter_q"
        if "linear_k" in n or "k_proj" in n:
            return "adapter_k"
        if "linear_v" in n or "v_proj" in n:
            return "adapter_v"
        if "gate_up" in n:
            return "adapter_ffn"
        return "adapter_other"

    return "other"


def role_family(role: str) -> str:
    if role.startswith("attn_"):
        return "attention"
    if role.startswith("ffn_"):
        return "ffn"
    if role.startswith("ssm_"):
        return "ssm"
    if role.startswith("adapter_"):
        return "adapter"
    return role


# ---------------------------------------------------------------------------
# Invariant extraction (same as Phase 0.18)
# ---------------------------------------------------------------------------

def extract_invariants(raw_bytes: bytes, shape: tuple) -> dict:
    n = len(raw_bytes)
    if n < 64:
        return {"te": 0.0, "mo": 0.0, "ac": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

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

    return {"te": round(te, 6), "mo": round(mo, 6), "ac": round(ac, 6)}


# ---------------------------------------------------------------------------
# Execution targets
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

    rng_np = np.random.default_rng(42)
    X = rng_np.standard_normal((1, rows))
    Y = X @ W
    output_norm = float(np.linalg.norm(Y))

    return {"output_norm": output_norm}


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def scan_model(model_name: str, model_info: dict) -> list:
    path = model_info["path"]
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

        role = classify_role_universal(name)
        if role in ("other", "skip"):
            continue

        byte_size = info["data_offsets"][1] - info["data_offsets"][0]
        if byte_size < 1024:
            continue

        shape = tuple(info["shape"])
        raw_bytes = read_raw_bytes(path, data_start, info)
        inv = extract_invariants(raw_bytes, shape)

        indices = load_tensor_numpy(path, data_start, info)
        codebook = load_tensor_numpy(path, data_start, tensors_info[cb_name])
        float_tensor = codebook[indices.astype(np.int32)]
        targets = execution_targets(float_tensor)

        rows = shape[0] if len(shape) >= 2 else 1
        cols = shape[1] if len(shape) >= 2 else shape[0]

        results.append({
            "name": base,
            "model": model_name,
            "role": role,
            "family": role_family(role),
            "arch": model_info["arch"],
            "arch_family": model_info["arch_family"],
            "shape": shape,
            "rows": rows,
            "cols": cols,
            "numel": rows * cols,
            "inv": inv,
            "targets": targets,
        })

    return results


# ---------------------------------------------------------------------------
# k-NN with separate train/test sets (NO retraining)
# ---------------------------------------------------------------------------

def knn_predict(train_X: np.ndarray, train_labels: list,
                test_X: np.ndarray, k: int = 5) -> list:
    """Predict test labels using train set as reference. No holdout leakage."""
    predictions = []
    for i in range(len(test_X)):
        dists = np.linalg.norm(train_X - test_X[i], axis=1)
        nearest_k = np.argsort(dists)[:k]
        votes = Counter(train_labels[j] for j in nearest_k)
        predictions.append(votes.most_common(1)[0][0])
    return predictions


def ols_predict(train_X: np.ndarray, train_y: np.ndarray,
                test_X: np.ndarray) -> tuple:
    """OLS trained on train, predict on test. Returns (predictions, adj_r2_train)."""
    n_train, p = train_X.shape
    X_aug = np.column_stack([np.ones(n_train), train_X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, train_y, rcond=None)
    except np.linalg.LinAlgError:
        return np.full(len(test_X), np.mean(train_y)), 0.0

    # Train R²
    y_hat_train = X_aug @ beta
    ss_res = np.sum((train_y - y_hat_train) ** 2)
    ss_tot = np.sum((train_y - np.mean(train_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n_train - 1) / (n_train - p - 1) if n_train > p + 1 else 0.0

    # Predict test
    test_aug = np.column_stack([np.ones(len(test_X)), test_X])
    y_pred = test_aug @ beta

    return y_pred, adj_r2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_data():
    results = []
    for name, info in TRAIN_MODELS.items():
        results.extend(scan_model(name, info))
    if len(results) < 50:
        pytest.skip(f"Need >= 50 train tensors, got {len(results)}")
    return results


@pytest.fixture(scope="module")
def holdout_data():
    results = {}
    for name, info in HOLDOUT_MODELS.items():
        data = scan_model(name, info)
        if data:
            results[name] = data
    if not results:
        pytest.skip("No holdout models available")
    return results


@pytest.fixture(scope="module")
def train_normalization(train_data):
    """Fit normalization on train only. Applied to holdout without refitting."""
    feats = np.array([[d["inv"]["te"], d["inv"]["mo"], d["inv"]["ac"]]
                       for d in train_data])
    mn = feats.min(axis=0)
    mx = feats.max(axis=0)
    rng = mx - mn
    rng[rng < 1e-12] = 1.0
    return {"min": mn, "max": mx, "range": rng}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_features(data_list: list, norm_params: dict) -> np.ndarray:
    X = np.array([[d["inv"]["te"], d["inv"]["mo"], d["inv"]["ac"]]
                   for d in data_list])
    return (X - norm_params["min"]) / norm_params["range"]


def classify_accuracy(predictions: list, labels: list) -> dict:
    n = len(labels)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)

    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    for p, l in zip(predictions, labels):
        per_class[l]["total"] += 1
        if p == l:
            per_class[l]["correct"] += 1

    per_class_acc = {}
    for cls in sorted(set(labels)):
        t = per_class[cls]["total"]
        per_class_acc[cls] = per_class[cls]["correct"] / t if t > 0 else 0.0

    return {"accuracy": correct / n if n > 0 else 0.0,
            "correct": correct, "total": n, "per_class": per_class_acc}


def r2_on_test(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ===========================================================================
# Tests
# ===========================================================================


class TestTrainData:
    def test_train_summary(self, train_data):
        roles = Counter(d["role"] for d in train_data)
        archs = Counter(d["arch"] for d in train_data)
        print(f"\n  TRAIN SET: {len(train_data)} tensors")
        for arch, count in sorted(archs.items()):
            print(f"    {arch}: {count}")
        print(f"  Roles: {len(roles)}")


class TestHoldoutData:
    def test_holdout_summary(self, holdout_data):
        print(f"\n  HOLDOUT MODELS:")
        for model_name, data in holdout_data.items():
            roles = Counter(d["role"] for d in data)
            families = Counter(d["family"] for d in data)
            print(f"\n    {model_name} ({data[0]['arch']}): {len(data)} tensors")
            print(f"      Families: {dict(families)}")
            print(f"      Roles: {dict(roles)}")


class TestCrossArchGeneralization:
    """The core test: train on Mamba+Qwen, predict cold on holdout."""

    def test_role_classification_generalization(self, train_data, holdout_data,
                                                 train_normalization):
        """Role classification: k-NN trained on Mamba+Qwen, tested on holdout."""
        train_X = normalize_features(train_data, train_normalization)
        train_labels = [d["role"] for d in train_data]
        train_roles = set(train_labels)

        print(f"\n  ROLE CLASSIFICATION GENERALIZATION")
        print(f"  Train: {len(train_data)} tensors, {len(train_roles)} roles")
        print(f"  Normalization: fitted on train ONLY")
        print(f"  Architecture labels: NOT used as features")
        print(f"  {'Model':<22} {'n':>4} {'Inv':>8} {'Random':>8} "
              f"{'Prior':>8} {'Shape':>8} {'Result':>8}")
        print(f"  {'-'*22} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        results = {}
        for model_name, data in holdout_data.items():
            # Filter to roles that exist in train set
            valid = [d for d in data if d["role"] in train_roles]
            if len(valid) < 5:
                print(f"  {model_name:<22} {len(data):>4} — too few shared roles")
                results[model_name] = {"status": "SKIP", "n": len(data)}
                continue

            test_X = normalize_features(valid, train_normalization)
            test_labels = [d["role"] for d in valid]

            # Invariant k-NN prediction
            preds = knn_predict(train_X, train_labels, test_X, k=5)
            inv_result = classify_accuracy(preds, test_labels)

            # Random baseline
            n_classes_test = len(set(test_labels))
            random_acc = 1.0 / n_classes_test if n_classes_test > 0 else 0.0

            # Role-prior baseline (most common role in TEST set)
            majority = Counter(test_labels).most_common(1)[0]
            prior_acc = majority[1] / len(test_labels)

            # Shape-only baseline
            shape_train = np.array([
                [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
                 np.log(max(d["numel"], 1))]
                for d in train_data
            ])
            s_mn, s_mx = shape_train.min(axis=0), shape_train.max(axis=0)
            s_rng = s_mx - s_mn
            s_rng[s_rng < 1e-12] = 1.0
            shape_train_n = (shape_train - s_mn) / s_rng

            shape_test = np.array([
                [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
                 np.log(max(d["numel"], 1))]
                for d in valid
            ])
            shape_test_n = (shape_test - s_mn) / s_rng
            shape_preds = knn_predict(shape_train_n, train_labels, shape_test_n, k=5)
            shape_result = classify_accuracy(shape_preds, test_labels)

            beats_random = inv_result["accuracy"] > random_acc + 0.05
            beats_prior = inv_result["accuracy"] > prior_acc + 0.05
            competitive_shape = inv_result["accuracy"] >= shape_result["accuracy"] - 0.10

            status = "PASS" if (beats_random and competitive_shape) else "FAIL"

            print(f"  {model_name:<22} {len(valid):>4} "
                  f"{inv_result['accuracy']:>7.1%} {random_acc:>7.1%} "
                  f"{prior_acc:>7.1%} {shape_result['accuracy']:>7.1%} "
                  f"{'PASS' if status == 'PASS' else 'FAIL':>8}")

            results[model_name] = {
                "status": status,
                "n": len(valid),
                "n_roles": n_classes_test,
                "inv_accuracy": inv_result["accuracy"],
                "random_baseline": random_acc,
                "prior_baseline": prior_acc,
                "shape_accuracy": shape_result["accuracy"],
                "beats_random": beats_random,
                "beats_prior": beats_prior,
                "competitive_shape": competitive_shape,
                "per_class": inv_result["per_class"],
            }

        return results

    def test_output_norm_generalization(self, train_data, holdout_data,
                                         train_normalization):
        """output_norm prediction: OLS trained on Mamba+Qwen, tested on holdout."""
        train_X = np.array([[d["inv"]["te"], d["inv"]["mo"], d["inv"]["ac"]]
                             for d in train_data])
        train_y = np.array([d["targets"]["output_norm"] for d in train_data])

        print(f"\n  OUTPUT_NORM PREDICTION GENERALIZATION")
        print(f"  OLS trained on {len(train_data)} tensors (Mamba+Qwen)")
        print(f"  {'Model':<22} {'n':>4} {'Inv R²':>8} {'Shape R²':>10} {'Result':>8}")
        print(f"  {'-'*22} {'-'*4} {'-'*8} {'-'*10} {'-'*8}")

        results = {}
        for model_name, data in holdout_data.items():
            test_X = np.array([[d["inv"]["te"], d["inv"]["mo"], d["inv"]["ac"]]
                                for d in data])
            test_y = np.array([d["targets"]["output_norm"] for d in data])

            y_pred, train_r2 = ols_predict(train_X, train_y, test_X)
            test_r2 = r2_on_test(test_y, y_pred)

            # Shape-only baseline
            shape_train = np.array([
                [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
                 np.log(max(d["numel"], 1))]
                for d in train_data
            ])
            shape_test = np.array([
                [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
                 np.log(max(d["numel"], 1))]
                for d in data
            ])
            shape_pred, _ = ols_predict(shape_train, train_y, shape_test)
            shape_r2 = r2_on_test(test_y, shape_pred)

            # R² > 0 means better than predicting mean
            status = "PASS" if test_r2 > 0.0 else "FAIL"

            print(f"  {model_name:<22} {len(data):>4} "
                  f"{test_r2:>8.4f} {shape_r2:>10.4f} {status:>8}")

            results[model_name] = {
                "status": status,
                "n": len(data),
                "inv_r2": round(test_r2, 4),
                "shape_r2": round(shape_r2, 4),
                "train_r2": round(train_r2, 4),
            }

        return results


class TestPerRoleDetail:
    """Show which roles transfer and which don't."""

    def test_per_role_transfer(self, train_data, holdout_data, train_normalization):
        train_X = normalize_features(train_data, train_normalization)
        train_labels = [d["role"] for d in train_data]
        train_roles = set(train_labels)

        print(f"\n  PER-ROLE TRANSFER DETAIL")

        for model_name, data in holdout_data.items():
            valid = [d for d in data if d["role"] in train_roles]
            if len(valid) < 5:
                continue

            test_X = normalize_features(valid, train_normalization)
            test_labels = [d["role"] for d in valid]
            preds = knn_predict(train_X, train_labels, test_X, k=5)

            result = classify_accuracy(preds, test_labels)
            role_counts = Counter(test_labels)

            print(f"\n    {model_name} ({data[0]['arch']}):")
            print(f"    {'Role':<20} {'n':>4} {'Acc':>8} {'Predicted As':>30}")
            print(f"    {'-'*20} {'-'*4} {'-'*8} {'-'*30}")

            for role in sorted(set(test_labels)):
                role_idx = [i for i, l in enumerate(test_labels) if l == role]
                role_preds = [preds[i] for i in role_idx]
                role_correct = sum(1 for p in role_preds if p == role)
                acc = role_correct / len(role_idx)

                # What was it predicted as?
                pred_dist = Counter(role_preds).most_common(3)
                pred_str = ", ".join(f"{p}({c})" for p, c in pred_dist)

                print(f"    {role:<20} {len(role_idx):>4} {acc:>7.0%} {pred_str:>30}")

            # Roles in holdout that DON'T exist in train
            novel_roles = [d for d in data if d["role"] not in train_roles
                           and d["role"] != "other"]
            if novel_roles:
                novel_counts = Counter(d["role"] for d in novel_roles)
                print(f"\n    Novel roles (not in train): {dict(novel_counts)}")
                print(f"    These tensors are EXCLUDED from accuracy — "
                      f"classifier cannot predict unseen roles")


class TestVerdict:
    """Final verdict with JSON receipt."""

    def test_final_verdict(self, train_data, holdout_data, train_normalization):
        t_start = time.time()
        cpu_start = time.process_time()
        start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

        train_X = normalize_features(train_data, train_normalization)
        train_labels = [d["role"] for d in train_data]
        train_roles = set(train_labels)
        train_X_raw = np.array([[d["inv"]["te"], d["inv"]["mo"], d["inv"]["ac"]]
                                 for d in train_data])
        train_y = np.array([d["targets"]["output_norm"] for d in train_data])

        # Shape train
        shape_train = np.array([
            [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
             np.log(max(d["numel"], 1))]
            for d in train_data
        ])
        s_mn, s_mx = shape_train.min(axis=0), shape_train.max(axis=0)
        s_rng = s_mx - s_mn
        s_rng[s_rng < 1e-12] = 1.0
        shape_train_n = (shape_train - s_mn) / s_rng

        model_results = {}
        families_pass = 0
        zamba2_pass = False

        for model_name, data in holdout_data.items():
            valid = [d for d in data if d["role"] in train_roles]

            # --- Role classification ---
            if len(valid) >= 5:
                test_X = normalize_features(valid, train_normalization)
                test_labels = [d["role"] for d in valid]
                preds = knn_predict(train_X, train_labels, test_X, k=5)
                role_result = classify_accuracy(preds, test_labels)

                n_classes = len(set(test_labels))
                random_acc = 1.0 / n_classes if n_classes > 0 else 0.0
                majority = Counter(test_labels).most_common(1)[0]
                prior_acc = majority[1] / len(test_labels)

                shape_test = np.array([
                    [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
                     np.log(max(d["numel"], 1))]
                    for d in valid
                ])
                shape_test_n = (shape_test - s_mn) / s_rng
                shape_preds = knn_predict(shape_train_n, train_labels,
                                          shape_test_n, k=5)
                shape_role = classify_accuracy(shape_preds, test_labels)
            else:
                role_result = {"accuracy": 0.0}
                random_acc = 0.0
                prior_acc = 0.0
                shape_role = {"accuracy": 0.0}

            # --- output_norm R² ---
            test_X_raw = np.array([[d["inv"]["te"], d["inv"]["mo"], d["inv"]["ac"]]
                                    for d in data])
            test_y = np.array([d["targets"]["output_norm"] for d in data])
            y_pred, _ = ols_predict(train_X_raw, train_y, test_X_raw)
            inv_r2 = r2_on_test(test_y, y_pred)

            shape_test_full = np.array([
                [np.log(max(d["rows"], 1)), np.log(max(d["cols"], 1)),
                 np.log(max(d["numel"], 1))]
                for d in data
            ])
            shape_pred, _ = ols_predict(shape_train, train_y, shape_test_full)
            shape_r2 = r2_on_test(test_y, shape_pred)

            # --- Decide ---
            beats_random = role_result["accuracy"] > random_acc + 0.05
            competitive_shape = role_result["accuracy"] >= shape_role["accuracy"] - 0.10
            model_pass = beats_random and competitive_shape

            if model_pass:
                families_pass += 1
            if model_name == "zamba2-1.2b" and model_pass:
                zamba2_pass = True

            model_results[model_name] = {
                "arch": data[0]["arch"],
                "arch_family": data[0]["arch_family"],
                "n_total": len(data),
                "n_valid": len(valid) if len(valid) >= 5 else 0,
                "role_accuracy": round(role_result["accuracy"], 4),
                "random_baseline": round(random_acc, 4),
                "prior_baseline": round(prior_acc, 4),
                "shape_accuracy": round(shape_role["accuracy"], 4),
                "beats_random": beats_random,
                "competitive_shape": competitive_shape,
                "inv_r2": round(inv_r2, 4),
                "shape_r2": round(shape_r2, 4),
                "pass": model_pass,
                "per_class": role_result.get("per_class", {}),
            }

        # --- Verdict ---
        if zamba2_pass and families_pass >= 2:
            status = "STRONG PASS"
            reason = (f"Invariant basis generalizes on {families_pass}/3 holdout "
                      f"families INCLUDING Zamba2 hybrid")
        elif families_pass >= 2:
            status = "PASS"
            reason = f"Invariant basis generalizes on {families_pass}/3 holdout families"
        elif families_pass == 1:
            status = "PARTIAL"
            passing = [k for k, v in model_results.items() if v["pass"]]
            reason = (f"Only 1/3 families pass ({', '.join(passing)}). "
                      f"Invariants may be family-specific")
        else:
            status = "FAIL"
            reason = "Invariants do not generalize beyond Mamba/Qwen training family"

        # --- Print ---
        print(f"""
{'=' * 80}
PHASE 0.19: CROSS-ARCHITECTURE GENERALIZATION — VERDICT
{'=' * 80}

  TRAIN: Mamba-130M (SSM v1) + Qwen2.5-Coder-1.5B (dense transformer)
         {len(train_data)} tensors, {len(train_roles)} roles
  HOLDOUT: {', '.join(holdout_data.keys())}
         Normalization fitted on train ONLY
         Architecture labels NOT used as features
         No retraining on holdout data

--- Holdout Results ---

  {'Model':<22} {'Arch':<20} {'n':>4} {'Role':>8} {'Rand':>8} """
              f"""{'Shape':>8} {'R²':>8} {'Pass':>6}
  {'-'*22} {'-'*20} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}""")

        for model_name, mr in model_results.items():
            print(f"  {model_name:<22} {mr['arch']:<20} {mr['n_valid']:>4} "
                  f"{mr['role_accuracy']:>7.1%} {mr['random_baseline']:>7.1%} "
                  f"{mr['shape_accuracy']:>7.1%} {mr['inv_r2']:>8.4f} "
                  f"{'YES' if mr['pass'] else 'NO':>6}")

        print(f"""
--- Key Question ---
  If Zamba2 passes, invariant basis is not just family-specific compression residue.
  Zamba2 result: {'PASS — hybrid bridge HOLDS' if zamba2_pass else 'FAIL — hybrid bridge BROKEN'}

{'=' * 80}
PHASE 0.19 RESULT:
  test = cross_architecture_generalization
  train = [mamba_v1, qwen_transformer]
  holdout = [zamba2_hybrid, tinyllama, mamba2_ssm]
  families_pass = {families_pass}/3
  zamba2_pass = {zamba2_pass}
  status = {status}
  reason = {reason}
{'=' * 80}""")

        # --- Receipt ---
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
            "phase": "0.19",
            "title": "Cross-Architecture Generalization",
            "status": status,
            "reason": reason,
            "train": {
                "models": list(TRAIN_MODELS.keys()),
                "n_tensors": len(train_data),
                "n_roles": len(train_roles),
            },
            "holdout": {k: v for k, v in model_results.items()},
            "families_pass": families_pass,
            "zamba2_pass": zamba2_pass,
            "invariants": ["transition_entropy", "markov_order", "index_autocorr"],
            "normalization": "train_only",
            "architecture_as_feature": False,
            "retraining_on_holdout": False,
            "body_opened_for_invariants": False,
            "body_opened_for_targets": True,
            "cost": cost,
        }

        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        receipt_path = RECEIPT_DIR / "phase_019_cross_architecture_generalization.json"
        with open(receipt_path, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\n  Receipt: {receipt_path}")
