"""Phase 0.13: Boundary-Aware Ghost Coordinates

Phase 0.12 proved blind 256KB windows separate format, not tensor role.
The fix: use file headers to scan per-tensor, not per-arbitrary-window.

Data sources:
  Mamba-130M HXQ (safetensors): 96 U8 index tensors, 4 roles
    ssm_dt (24), attention/in_proj (24), attention_out (24), ssm_x (24)

  Qwen2.5-Coder-1.5B HXQ (safetensors): 196 U8 index tensors, 7 roles
    attn_q (28), attn_k (28), attn_v (28), attn_o (28),
    ffn_gate (28), ffn_up (28), ffn_down (28)

Total: 292 tensors, 11 roles, 2 architectures, 1 format (HXQ safetensors).

Ghost coordinates:
  complexity     = mean(TE, TR)
  predictability = MO
  locality       = AC

Test:
  1. Do roles cluster in ghost coordinate space?
  2. Which dimensions separate which roles?
  3. Do attention tensors occupy a different region than FFN?
  4. Does architecture matter (Mamba vs Transformer)?

WO-CRYSTAL-VAULT-01: Phase 0.13 — Boundary-Aware Ghost Coordinates
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
# Model files with per-tensor metadata
# ---------------------------------------------------------------------------

MAMBA_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
    "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
)

QWEN_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--qwen2.5-coder-1.5b-helix"
    "/snapshots/0a5c17fba5cc81018423eba394295ca8568caff2/model.safetensors"
)


def read_safetensors_index(path: Path) -> tuple[dict, int]:
    """Read safetensors header. Returns (tensor_dict, data_start)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    data_start = 8 + hlen
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return tensors, data_start


def classify_mamba_role(name: str) -> str:
    n = name.lower()
    if ".indices" not in n:
        return "skip"
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
    if ".indices" not in n:
        return "skip"
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
    """Map fine-grained role to coarse family."""
    if role.startswith("attn_"):
        return "attention"
    if role.startswith("ffn_"):
        return "ffn"
    if role.startswith("ssm_"):
        return "ssm"
    return role


# ---------------------------------------------------------------------------
# Ghost coordinates (from Phase 0.12, per-tensor version)
# ---------------------------------------------------------------------------

def ghost_coordinates_from_bytes(raw_bytes: bytes) -> dict:
    """Compute ghost coordinates from raw U8 index bytes.

    Same math as Phase 0.12 but on per-tensor data.
    body_opened = false. labels_used = none.
    """
    n = len(raw_bytes)
    if n < 64:
        return {"complexity": 0.0, "predictability": 0.0, "locality": 0.0,
                "te": 0.0, "tr": 0.0, "mo": 0.0, "ac": 0.0}

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Bigram matrix
    bigram_counts = np.zeros((256, 256), dtype=np.int64)
    for i in range(n - 1):
        bigram_counts[arr[i], arr[i + 1]] += 1
    total_bigrams = n - 1

    # TE
    bigram_probs = bigram_counts[bigram_counts > 0] / total_bigrams
    bigram_h = -float(np.sum(bigram_probs * np.log2(bigram_probs)))
    max_bigram_h = 2.0 * np.log2(256)
    te = bigram_h / max_bigram_h if max_bigram_h > 0 else 0.0

    # TR
    row_sums = bigram_counts.sum(axis=1)
    row_entropies = []
    for r in range(256):
        if row_sums[r] > 0:
            rp = bigram_counts[r][bigram_counts[r] > 0] / row_sums[r]
            row_entropies.append(-float(np.sum(rp * np.log2(rp))))
    mean_row_h = np.mean(row_entropies) if row_entropies else 0.0
    tr = mean_row_h / np.log2(256)

    # MO
    byte_counts = np.bincount(arr, minlength=256).astype(np.float64)
    probs = byte_counts[byte_counts > 0] / n
    unigram_h = -float(np.sum(probs * np.log2(probs)))
    mo = bigram_h / (2.0 * unigram_h) if unigram_h > 0 else 0.0

    # AC
    a = arr[:-1].astype(np.float64)
    b = arr[1:].astype(np.float64)
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    ac = float(np.mean((a - ma) * (b - mb)) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0

    complexity = (te + tr) / 2.0
    predictability = mo
    locality = ac

    return {
        "complexity": round(complexity, 6),
        "predictability": round(predictability, 6),
        "locality": round(locality, 6),
        "te": round(te, 6), "tr": round(tr, 6),
        "mo": round(mo, 6), "ac": round(ac, 6),
    }


# ---------------------------------------------------------------------------
# Fixture: scan all per-tensor data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_tensor_data():
    """Scan every U8 index tensor from Mamba and Qwen.

    Each point has:
      coords: ghost coordinate (complexity, predictability, locality)
      role: fine-grained role (attn_q, ffn_gate, ssm_dt, etc.)
      family: coarse family (attention, ffn, ssm)
      arch: architecture (mamba, transformer)
      layer: layer number
      name: full tensor name
    """
    points = []

    # Mamba
    if MAMBA_HXQ.exists():
        tensors, data_start = read_safetensors_index(MAMBA_HXQ)
        for name, info in tensors.items():
            if info["dtype"] != "U8":
                continue
            role = classify_mamba_role(name)
            if role == "skip" or role == "other":
                continue

            start, end = info["data_offsets"]
            with open(MAMBA_HXQ, "rb") as f:
                f.seek(data_start + start)
                raw = f.read(end - start)

            # Extract layer number
            layer = -1
            for part in name.split("."):
                if part.isdigit():
                    layer = int(part)
                    break

            coords = ghost_coordinates_from_bytes(raw)
            points.append({
                "name": name,
                "role": role,
                "family": role_family(role),
                "arch": "mamba",
                "layer": layer,
                "coords": coords,
                "size": len(raw),
            })

    # Qwen
    if QWEN_HXQ.exists():
        tensors, data_start = read_safetensors_index(QWEN_HXQ)
        for name, info in tensors.items():
            if info["dtype"] != "U8":
                continue
            role = classify_qwen_role(name)
            if role == "skip" or role == "other":
                continue

            start, end = info["data_offsets"]
            with open(QWEN_HXQ, "rb") as f:
                f.seek(data_start + start)
                raw = f.read(end - start)

            layer = -1
            for part in name.split("."):
                if part.isdigit():
                    layer = int(part)
                    break

            coords = ghost_coordinates_from_bytes(raw)
            points.append({
                "name": name,
                "role": role,
                "family": role_family(role),
                "arch": "transformer",
                "layer": layer,
                "coords": coords,
                "size": len(raw),
            })

    return points


# ===========================================================================
# Tests
# ===========================================================================


class TestDataAvailable:
    def test_sufficient_tensors(self, all_tensor_data):
        assert len(all_tensor_data) >= 100, f"Need >= 100 tensors, got {len(all_tensor_data)}"

    def test_both_architectures(self, all_tensor_data):
        archs = set(p["arch"] for p in all_tensor_data)
        assert len(archs) >= 2, f"Need >= 2 architectures, got {archs}"

    def test_multiple_roles(self, all_tensor_data):
        roles = set(p["role"] for p in all_tensor_data)
        assert len(roles) >= 5, f"Need >= 5 roles, got {roles}"
        print(f"\n  Roles found: {sorted(roles)}")
        role_counts = Counter(p["role"] for p in all_tensor_data)
        for role, count in sorted(role_counts.items()):
            print(f"    {role:<16} {count}")


class TestRoleClustering:
    """Do tensor roles naturally cluster in ghost coordinate space?"""

    def test_role_centroids(self, all_tensor_data):
        """Print centroid of each role in 3D ghost space."""
        by_role = defaultdict(list)
        for p in all_tensor_data:
            c = p["coords"]
            by_role[p["role"]].append([c["complexity"], c["predictability"], c["locality"]])

        print(f"\n  Role centroids (complexity, predictability, locality):")
        print(f"  {'role':<16} {'n':>4}  {'cmplx':>7} {'pred':>7} {'loc':>7}")
        print(f"  {'-'*16} {'-'*4}  {'-'*7} {'-'*7} {'-'*7}")
        for role in sorted(by_role.keys()):
            arr = np.array(by_role[role])
            m = arr.mean(axis=0)
            print(f"  {role:<16} {len(arr):>4}  {m[0]:>7.4f} {m[1]:>7.4f} {m[2]:>7.4f}")

    def test_family_centroids(self, all_tensor_data):
        """Print centroid of each family (attention, ffn, ssm)."""
        by_family = defaultdict(list)
        for p in all_tensor_data:
            c = p["coords"]
            by_family[p["family"]].append([c["complexity"], c["predictability"], c["locality"]])

        print(f"\n  Family centroids:")
        print(f"  {'family':<12} {'n':>4}  {'cmplx':>7} {'pred':>7} {'loc':>7}")
        for fam in sorted(by_family.keys()):
            arr = np.array(by_family[fam])
            m = arr.mean(axis=0)
            s = arr.std(axis=0)
            print(f"  {fam:<12} {len(arr):>4}  {m[0]:>7.4f} {m[1]:>7.4f} {m[2]:>7.4f}  "
                  f"std=({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")

    def test_f_ratio_by_role(self, all_tensor_data):
        """F-ratio per dimension for role separation."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        roles = [p["role"] for p in all_tensor_data]

        print(f"\n  F-ratio by role (between/within variance):")
        for dim, name in [(0, "complexity"), (1, "predictability"), (2, "locality")]:
            by_role = defaultdict(list)
            for i, r in enumerate(roles):
                by_role[r].append(coords[i, dim])
            all_vals = [v for vs in by_role.values() for v in vs]
            grand_mean = np.mean(all_vals)
            between = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_role.values()) / len(all_vals)
            within = sum(np.var(vs) * len(vs)
                        for vs in by_role.values()) / len(all_vals)
            f = between / within if within > 1e-12 else 0.0
            marker = "SEPARATES" if f > 1.0 else ("weak" if f > 0.3 else "NO")
            print(f"    {name:<16} F={f:>6.2f}  {marker}")

    def test_f_ratio_by_family(self, all_tensor_data):
        """F-ratio per dimension for family separation."""
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        families = [p["family"] for p in all_tensor_data]

        print(f"\n  F-ratio by family:")
        for dim, name in [(0, "complexity"), (1, "predictability"), (2, "locality")]:
            by_fam = defaultdict(list)
            for i, f in enumerate(families):
                by_fam[f].append(coords[i, dim])
            all_vals = [v for vs in by_fam.values() for v in vs]
            grand_mean = np.mean(all_vals)
            between = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_fam.values()) / len(all_vals)
            within = sum(np.var(vs) * len(vs)
                        for vs in by_fam.values()) / len(all_vals)
            f_val = between / within if within > 1e-12 else 0.0
            marker = "SEPARATES" if f_val > 1.0 else ("weak" if f_val > 0.3 else "NO")
            print(f"    {name:<16} F={f_val:>6.2f}  {marker}")


class TestSilhouetteByRole:
    """Silhouette scores for role and family clustering."""

    def _silhouette(self, coords, labels):
        unique = sorted(set(labels))
        if len(unique) < 2:
            return 0.0, {}
        label_idx = defaultdict(list)
        for i, l in enumerate(labels):
            label_idx[l].append(i)

        sils = []
        for i in range(len(coords)):
            my = labels[i]
            same = [j for j in label_idx[my] if j != i]
            a = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in same]) if same else 0.0
            b = float("inf")
            for ol in unique:
                if ol == my:
                    continue
                other = label_idx[ol]
                if other:
                    b = min(b, np.mean([np.linalg.norm(coords[i] - coords[j]) for j in other]))
            if b == float("inf"):
                b = 0.0
            s = (b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0
            sils.append(s)

        per_label = {}
        for l in unique:
            idx = label_idx[l]
            per_label[l] = float(np.mean([sils[i] for i in idx]))

        return float(np.mean(sils)), per_label

    def test_silhouette_by_role(self, all_tensor_data):
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["role"] for p in all_tensor_data]
        mean_sil, per_label = self._silhouette(coords, labels)

        print(f"\n  Silhouette by role: {mean_sil:.4f}")
        for label in sorted(per_label.keys()):
            n = sum(1 for l in labels if l == label)
            print(f"    {label:<16} sil={per_label[label]:>7.4f} (n={n})")

    def test_silhouette_by_family(self, all_tensor_data):
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["family"] for p in all_tensor_data]
        mean_sil, per_label = self._silhouette(coords, labels)

        print(f"\n  Silhouette by family: {mean_sil:.4f}")
        for label in sorted(per_label.keys()):
            n = sum(1 for l in labels if l == label)
            print(f"    {label:<12} sil={per_label[label]:>7.4f} (n={n})")

    def test_silhouette_by_arch(self, all_tensor_data):
        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])
        labels = [p["arch"] for p in all_tensor_data]
        mean_sil, per_label = self._silhouette(coords, labels)

        print(f"\n  Silhouette by architecture: {mean_sil:.4f}")
        for label in sorted(per_label.keys()):
            n = sum(1 for l in labels if l == label)
            print(f"    {label:<14} sil={per_label[label]:>7.4f} (n={n})")


class TestReport:
    def test_full_report(self, all_tensor_data):
        n = len(all_tensor_data)
        roles = sorted(set(p["role"] for p in all_tensor_data))
        families = sorted(set(p["family"] for p in all_tensor_data))
        archs = sorted(set(p["arch"] for p in all_tensor_data))

        coords = np.array([[p["coords"]["complexity"],
                           p["coords"]["predictability"],
                           p["coords"]["locality"]] for p in all_tensor_data])

        # Silhouettes
        def sil(labels):
            unique = sorted(set(labels))
            if len(unique) < 2:
                return 0.0
            label_idx = defaultdict(list)
            for i, l in enumerate(labels):
                label_idx[l].append(i)
            sils = []
            for i in range(len(coords)):
                my = labels[i]
                same = [j for j in label_idx[my] if j != i]
                a = np.mean([np.linalg.norm(coords[i] - coords[j]) for j in same]) if same else 0.0
                b = float("inf")
                for ol in unique:
                    if ol == my:
                        continue
                    other = label_idx[ol]
                    if other:
                        b = min(b, np.mean([np.linalg.norm(coords[i] - coords[j]) for j in other]))
                if b == float("inf"):
                    b = 0.0
                sils.append((b - a) / max(a, b) if max(a, b) > 1e-12 else 0.0)
            return float(np.mean(sils))

        sil_role = sil([p["role"] for p in all_tensor_data])
        sil_family = sil([p["family"] for p in all_tensor_data])
        sil_arch = sil([p["arch"] for p in all_tensor_data])

        # F-ratios for family
        f_ratios = []
        for dim, name in [(0, "complexity"), (1, "predictability"), (2, "locality")]:
            by_fam = defaultdict(list)
            for i, p in enumerate(all_tensor_data):
                by_fam[p["family"]].append(coords[i, dim])
            all_vals = [v for vs in by_fam.values() for v in vs]
            grand_mean = np.mean(all_vals)
            between = sum(len(vs) * (np.mean(vs) - grand_mean) ** 2
                         for vs in by_fam.values()) / len(all_vals)
            within = sum(np.var(vs) * len(vs)
                        for vs in by_fam.values()) / len(all_vals)
            f_val = between / within if within > 1e-12 else 0.0
            f_ratios.append((name, f_val))

        print(f"""
================================================================================
PHASE 0.13: BOUNDARY-AWARE GHOST COORDINATES
================================================================================

Tensors: {n}
Roles:   {len(roles)} ({', '.join(roles)})
Families: {len(families)} ({', '.join(families)})
Architectures: {len(archs)} ({', '.join(archs)})

scan_unit: per_tensor (from safetensors header)
labels_used_for_coordinates: NONE
body_opened: false

--- Coordinate Ranges ---
  complexity:     [{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]
  predictability: [{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]
  locality:       [{coords[:,2].min():.4f}, {coords[:,2].max():.4f}]

--- Silhouette Scores ---
  By role (11 types):    {sil_role:>7.4f}  {'CLUSTERS' if sil_role > 0.25 else 'WEAK' if sil_role > 0.0 else 'NONE'}
  By family (3 types):   {sil_family:>7.4f}  {'CLUSTERS' if sil_family > 0.25 else 'WEAK' if sil_family > 0.0 else 'NONE'}
  By architecture:       {sil_arch:>7.4f}  {'CLUSTERS' if sil_arch > 0.25 else 'WEAK' if sil_arch > 0.0 else 'NONE'}

--- Family F-ratios (between/within variance) ---""")
        for name, f in f_ratios:
            marker = "SEPARATES" if f > 1.0 else ("weak" if f > 0.3 else "NO")
            print(f"  {name:<16} F={f:>6.2f}  {marker}")

        print(f"""
================================================================================
VERDICT:
""")
        if sil_role > 0.25 or sil_family > 0.25:
            print("  Tensor roles form NATURAL CLUSTERS in ghost coordinate space.")
            print("  Labels are post-processing — the geometry exists before naming.")
            print("  Phase 0.12's failure was scan granularity, not signal quality.")
            best = max(f_ratios, key=lambda x: x[1])
            print(f"  Best separating dimension: {best[0]} (F={best[1]:.2f})")
        elif sil_role > 0.0 or sil_family > 0.0:
            print("  Weak but real clustering. Some roles separate, others overlap.")
            best = max(f_ratios, key=lambda x: x[1])
            print(f"  Best separating dimension: {best[0]} (F={best[1]:.2f})")
        else:
            print("  No natural clustering even with boundary-aware scanning.")
            print("  Transition structure does not separate tensor roles.")

        if sil_arch > 0.25:
            print(f"\n  Architecture clusters (sil={sil_arch:.3f}):")
            print("  Mamba and Transformer occupy different regions.")
        print("================================================================================")
