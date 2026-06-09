"""Phase 0.15: Can Ghost Coordinates Predict Tensor Function?

Phase 0.13 showed F-ratios of 546 (complexity) and 1094 (predictability)
for role separation — the signal is massive. Phase 0.11 showed fixed
thresholds fail. This phase tests whether a LEARNED classifier succeeds.

Classifier: k-NN (simplest possible — no parameters beyond k).
Validation: leave-one-out cross-validation (each tensor predicted from all others).
Features: ghost coordinates (complexity, predictability, locality).

Test at three granularities:
  Fine:   11 roles (attn_q, attn_k, ssm_dt, ffn_gate, etc.)
  Coarse: 3 families (attention, ffn, ssm)
  Arch:   2 architectures (mamba, transformer)

Also test which dimensions carry the signal:
  complexity only
  predictability only
  complexity + predictability
  all three (+ locality)

body_opened: false
labels_used_for_features: NONE (labels only used as ground truth for scoring)

WO-CRYSTAL-VAULT-01: Phase 0.15 — Ghost Coordinate Classifier
"""
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Data (shared with Phase 0.13/0.14)
# ---------------------------------------------------------------------------

MAMBA_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
    "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
)

QWEN_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--qwen2.5-coder-1.5b-helix"
    "/snapshots/0a5c17fba5cc81018423eba394295ca8568caff2/model.safetensors"
)


def read_safetensors_index(path: Path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return {k: v for k, v in header.items() if k != "__metadata__"}, 8 + hlen


def classify_role(name: str, arch: str) -> str:
    n = name.lower()
    if ".indices" not in n:
        return "skip"
    if arch == "mamba":
        parent = name.replace(".indices", "")
        if "in_proj" in parent: return "ssm_in_proj"
        if "out_proj" in parent: return "ssm_out_proj"
        if "dt_proj" in parent: return "ssm_dt"
        if "x_proj" in parent: return "ssm_x"
    else:
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


def ghost_coordinates_from_bytes(raw_bytes: bytes) -> dict:
    n = len(raw_bytes)
    if n < 64:
        return {"complexity": 0.0, "predictability": 0.0, "locality": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    bigram_counts = np.zeros((256, 256), dtype=np.int64)
    for i in range(n - 1):
        bigram_counts[arr[i], arr[i + 1]] += 1
    total_bigrams = n - 1

    bigram_probs = bigram_counts[bigram_counts > 0] / total_bigrams
    bigram_h = -float(np.sum(bigram_probs * np.log2(bigram_probs)))
    max_bigram_h = 2.0 * np.log2(256)
    te = bigram_h / max_bigram_h if max_bigram_h > 0 else 0.0

    row_sums = bigram_counts.sum(axis=1)
    row_entropies = []
    for r in range(256):
        if row_sums[r] > 0:
            rp = bigram_counts[r][bigram_counts[r] > 0] / row_sums[r]
            row_entropies.append(-float(np.sum(rp * np.log2(rp))))
    tr = np.mean(row_entropies) / np.log2(256) if row_entropies else 0.0

    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 0.0

    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    ac = float(np.mean((a - ma) * (b - mb)) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0

    return {
        "complexity": round((te + tr) / 2.0, 6),
        "predictability": round(mo, 6),
        "locality": round(ac, 6),
    }


# ---------------------------------------------------------------------------
# k-NN classifier (no external dependencies)
# ---------------------------------------------------------------------------

def knn_leave_one_out(coords: np.ndarray, labels: list, k: int = 5) -> dict:
    """Leave-one-out k-NN classification.

    For each point, find k nearest neighbors (excluding self),
    vote on label, compare to ground truth.

    Returns: {accuracy, per_class_accuracy, confusion, predictions}
    """
    n = len(labels)
    predictions = []
    correct = 0

    for i in range(n):
        # Distances from point i to all others
        dists = np.linalg.norm(coords - coords[i], axis=1)
        dists[i] = float("inf")  # Exclude self
        nearest_k = np.argsort(dists)[:k]

        # Vote
        votes = Counter(labels[j] for j in nearest_k)
        predicted = votes.most_common(1)[0][0]
        predictions.append(predicted)
        if predicted == labels[i]:
            correct += 1

    accuracy = correct / n

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for i in range(n):
        class_total[labels[i]] += 1
        if predictions[i] == labels[i]:
            class_correct[labels[i]] += 1

    per_class = {}
    for cls in sorted(set(labels)):
        per_class[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0

    # Confusion matrix as dict
    confusion = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        confusion[labels[i]][predictions[i]] += 1

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "per_class": per_class,
        "confusion": dict(confusion),
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_tensor_data():
    points = []
    for model_path, arch in [(MAMBA_HXQ, "mamba"), (QWEN_HXQ, "transformer")]:
        if not model_path.exists():
            continue
        tensors, data_start = read_safetensors_index(model_path)
        for name, info in tensors.items():
            if info["dtype"] != "U8":
                continue
            role = classify_role(name, arch)
            if role in ("skip", "other"):
                continue

            start, end = info["data_offsets"]
            with open(model_path, "rb") as f:
                f.seek(data_start + start)
                raw = f.read(end - start)

            coords = ghost_coordinates_from_bytes(raw)
            points.append({
                "name": name, "role": role,
                "family": role_family(role), "arch": arch,
                "coords": coords,
            })
    return points


# ===========================================================================
# Tests
# ===========================================================================


class TestDataAvailable:
    def test_sufficient_tensors(self, all_tensor_data):
        assert len(all_tensor_data) >= 200

    def test_role_balance(self, all_tensor_data):
        roles = Counter(p["role"] for p in all_tensor_data)
        print(f"\n  Role distribution:")
        for role, count in sorted(roles.items()):
            print(f"    {role:<16} {count}")
        # Each role should have enough for k-NN
        assert all(c >= 10 for c in roles.values()), "Some roles too small for k-NN"


class TestRoleClassification:
    """Can k-NN predict the 11 fine-grained tensor roles?"""

    def test_knn_role_all_dims(self, all_tensor_data):
        """k-NN on (complexity, predictability, locality) → role."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["role"] for p in all_tensor_data]

        # Normalize each dimension to [0,1] for fair distance
        for d in range(coords.shape[1]):
            mn, mx = coords[:, d].min(), coords[:, d].max()
            if mx - mn > 1e-12:
                coords[:, d] = (coords[:, d] - mn) / (mx - mn)

        result = knn_leave_one_out(coords, labels, k=5)
        n_classes = len(set(labels))
        random_baseline = 1.0 / n_classes

        print(f"\n  k-NN role classification (3D, k=5):")
        print(f"  Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
        print(f"  Random baseline: {random_baseline:.1%} ({n_classes} classes)")
        print(f"  Lift over random: {result['accuracy'] / random_baseline:.1f}x")
        print(f"\n  Per-role accuracy:")
        for role, acc in sorted(result["per_class"].items()):
            n = sum(1 for l in labels if l == role)
            print(f"    {role:<16} {acc:.1%} ({int(acc * n)}/{n})")

        assert result["accuracy"] > random_baseline * 2, \
            f"Accuracy {result['accuracy']:.1%} not much better than random {random_baseline:.1%}"

    def test_knn_role_complexity_only(self, all_tensor_data):
        """k-NN on complexity alone → role."""
        coords = np.array([[p["coords"]["complexity"]] for p in all_tensor_data])
        labels = [p["role"] for p in all_tensor_data]
        mn, mx = coords.min(), coords.max()
        if mx - mn > 1e-12:
            coords = (coords - mn) / (mx - mn)

        result = knn_leave_one_out(coords, labels, k=5)
        print(f"\n  k-NN role (complexity only): {result['accuracy']:.1%}")

    def test_knn_role_predictability_only(self, all_tensor_data):
        """k-NN on predictability alone → role."""
        coords = np.array([[p["coords"]["predictability"]] for p in all_tensor_data])
        labels = [p["role"] for p in all_tensor_data]
        mn, mx = coords.min(), coords.max()
        if mx - mn > 1e-12:
            coords = (coords - mn) / (mx - mn)

        result = knn_leave_one_out(coords, labels, k=5)
        print(f"\n  k-NN role (predictability only): {result['accuracy']:.1%}")

    def test_knn_role_2d(self, all_tensor_data):
        """k-NN on (complexity, predictability) → role."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"]] for p in all_tensor_data])
        labels = [p["role"] for p in all_tensor_data]
        for d in range(2):
            mn, mx = coords[:, d].min(), coords[:, d].max()
            if mx - mn > 1e-12:
                coords[:, d] = (coords[:, d] - mn) / (mx - mn)

        result = knn_leave_one_out(coords, labels, k=5)
        print(f"\n  k-NN role (2D: complexity + predictability): {result['accuracy']:.1%}")


class TestFamilyClassification:
    """Can k-NN predict the 3 coarse families?"""

    def test_knn_family(self, all_tensor_data):
        """k-NN on all 3 dims → family."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["family"] for p in all_tensor_data]
        for d in range(3):
            mn, mx = coords[:, d].min(), coords[:, d].max()
            if mx - mn > 1e-12:
                coords[:, d] = (coords[:, d] - mn) / (mx - mn)

        result = knn_leave_one_out(coords, labels, k=5)
        print(f"\n  k-NN family classification (3D, k=5):")
        print(f"  Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
        print(f"  Random baseline: {100/len(set(labels)):.1f}%")
        for fam, acc in sorted(result["per_class"].items()):
            n = sum(1 for l in labels if l == fam)
            print(f"    {fam:<12} {acc:.1%} ({int(acc * n)}/{n})")


class TestArchitectureClassification:
    """Can k-NN predict mamba vs transformer?"""

    def test_knn_arch(self, all_tensor_data):
        """k-NN on all 3 dims → architecture."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["arch"] for p in all_tensor_data]
        for d in range(3):
            mn, mx = coords[:, d].min(), coords[:, d].max()
            if mx - mn > 1e-12:
                coords[:, d] = (coords[:, d] - mn) / (mx - mn)

        result = knn_leave_one_out(coords, labels, k=5)
        print(f"\n  k-NN architecture classification (3D, k=5):")
        print(f"  Accuracy: {result['correct']}/{result['total']} = {result['accuracy']:.1%}")
        for arch, acc in sorted(result["per_class"].items()):
            n = sum(1 for l in labels if l == arch)
            print(f"    {arch:<14} {acc:.1%} ({int(acc * n)}/{n})")


class TestKSensitivity:
    """How sensitive is role classification to k?"""

    def test_k_sweep(self, all_tensor_data):
        """Sweep k from 1 to 15 for role classification."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["role"] for p in all_tensor_data]
        for d in range(3):
            mn, mx = coords[:, d].min(), coords[:, d].max()
            if mx - mn > 1e-12:
                coords[:, d] = (coords[:, d] - mn) / (mx - mn)

        print(f"\n  k-sweep for role classification:")
        best_k, best_acc = 1, 0.0
        for k in [1, 3, 5, 7, 9, 11, 15]:
            result = knn_leave_one_out(coords, labels, k=k)
            marker = " <<<" if result["accuracy"] > best_acc else ""
            if result["accuracy"] > best_acc:
                best_k, best_acc = k, result["accuracy"]
            print(f"    k={k:>2}: {result['accuracy']:.1%}{marker}")
        print(f"  Best: k={best_k}, accuracy={best_acc:.1%}")


class TestReport:
    def test_full_report(self, all_tensor_data):
        n = len(all_tensor_data)
        roles = sorted(set(p["role"] for p in all_tensor_data))
        families = sorted(set(p["family"] for p in all_tensor_data))

        # Compute all results
        coords_3d = np.array([[p["coords"]["complexity"],
                              p["coords"]["predictability"],
                              p["coords"]["locality"]] for p in all_tensor_data])
        coords_2d = coords_3d[:, :2].copy()
        coords_c = coords_3d[:, 0:1].copy()
        coords_p = coords_3d[:, 1:2].copy()

        role_labels = [p["role"] for p in all_tensor_data]
        fam_labels = [p["family"] for p in all_tensor_data]
        arch_labels = [p["arch"] for p in all_tensor_data]

        # Normalize
        def norm(c):
            c = c.copy()
            for d in range(c.shape[1]):
                mn, mx = c[:, d].min(), c[:, d].max()
                if mx - mn > 1e-12:
                    c[:, d] = (c[:, d] - mn) / (mx - mn)
            return c

        r_3d = knn_leave_one_out(norm(coords_3d), role_labels, k=5)
        r_2d = knn_leave_one_out(norm(coords_2d), role_labels, k=5)
        r_c = knn_leave_one_out(norm(coords_c), role_labels, k=5)
        r_p = knn_leave_one_out(norm(coords_p), role_labels, k=5)

        f_3d = knn_leave_one_out(norm(coords_3d), fam_labels, k=5)
        a_3d = knn_leave_one_out(norm(coords_3d), arch_labels, k=5)

        role_random = 1.0 / len(roles)
        fam_random = 1.0 / len(families)

        print(f"""
================================================================================
PHASE 0.15: GHOST COORDINATE CLASSIFIER
================================================================================

Tensors: {n}
Roles: {len(roles)}
Families: {len(families)}
Classifier: k-NN (k=5), leave-one-out cross-validation
Features: ghost coordinates (transition graph only)
body_opened: false

--- Role Classification (11 types) ---
  Random baseline: {role_random:.1%}

  Dimensions used       Accuracy    Lift
  ------------------    --------    ----
  complexity only       {r_c['accuracy']:.1%}       {r_c['accuracy']/role_random:.1f}x
  predictability only   {r_p['accuracy']:.1%}       {r_p['accuracy']/role_random:.1f}x
  complexity + pred     {r_2d['accuracy']:.1%}       {r_2d['accuracy']/role_random:.1f}x
  all three (+ loc)     {r_3d['accuracy']:.1%}       {r_3d['accuracy']/role_random:.1f}x

--- Family Classification (3 types) ---
  Random baseline: {fam_random:.1%}
  3D accuracy: {f_3d['accuracy']:.1%}  (lift: {f_3d['accuracy']/fam_random:.1f}x)

--- Architecture Classification (2 types) ---
  Random baseline: 50.0%
  3D accuracy: {a_3d['accuracy']:.1%}

--- Per-Role Detail (3D, k=5) ---""")
        for role in roles:
            acc = r_3d["per_class"].get(role, 0.0)
            n_role = sum(1 for l in role_labels if l == role)
            correct = int(acc * n_role)
            print(f"  {role:<16} {acc:>5.1%} ({correct}/{n_role})")

        print(f"""
================================================================================
VERDICT:
""")
        if r_3d["accuracy"] > 0.6:
            print(f"  Ghost coordinates PREDICT tensor function.")
            print(f"  {r_3d['accuracy']:.1%} accuracy on 11 roles = {r_3d['accuracy']/role_random:.1f}x random.")
            print(f"  The transition graph encodes computational behavior.")
            print(f"  Fixed thresholds failed (Phase 0.11: 22%).")
            print(f"  Learned classifier succeeds.")
        elif r_3d["accuracy"] > 0.4:
            print(f"  Partial role prediction. {r_3d['accuracy']:.1%} on 11 roles.")
            print(f"  Some roles cleanly separable, others overlap.")
        else:
            print(f"  Role prediction fails even with learned classifier.")
            print(f"  Ghost coordinates do not encode tensor function.")

        if f_3d["accuracy"] > 0.8:
            print(f"\n  Family classification: {f_3d['accuracy']:.1%} — strong separation.")
        if a_3d["accuracy"] > 0.8:
            print(f"  Architecture classification: {a_3d['accuracy']:.1%} — strong separation.")

        print("================================================================================")
