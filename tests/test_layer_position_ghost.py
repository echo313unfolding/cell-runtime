"""Phase 0.14: Layer-Position Ghost Test

Phase 0.13 found locality carries NO role signal (F<0.22).
The advisor's hypothesis: locality may encode WHERE in the network,
not WHAT role the tensor has.

If ghost coordinates track layer position, then Ghost is not just
classifying tensors — it is mapping model geometry.

Test:
  For each tensor with known layer_id (0..N):
    Correlate complexity vs layer_id
    Correlate predictability vs layer_id
    Correlate locality vs layer_id

  Per-role: do coordinates drift systematically across layers?
  Per-architecture: does Mamba drift differently from Transformer?

Data: same 292 tensors from Phase 0.13 (Mamba-130M + Qwen-1.5B HXQ).

WO-CRYSTAL-VAULT-01: Phase 0.14 — Layer-Position Ghost Test
"""
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sp_stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Model files + scanning (shared with Phase 0.13)
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


def ghost_coordinates_from_bytes(raw_bytes: bytes) -> dict:
    n = len(raw_bytes)
    if n < 64:
        return {"complexity": 0.0, "predictability": 0.0, "locality": 0.0,
                "te": 0.0, "tr": 0.0, "mo": 0.0, "ac": 0.0}

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
        "te": round(te, 6), "tr": round(tr, 6),
        "mo": round(mo, 6), "ac": round(ac, 6),
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

            layer = -1
            for part in name.split("."):
                if part.isdigit():
                    layer = int(part)
                    break
            if layer < 0:
                continue

            start, end = info["data_offsets"]
            with open(model_path, "rb") as f:
                f.seek(data_start + start)
                raw = f.read(end - start)

            coords = ghost_coordinates_from_bytes(raw)
            points.append({
                "name": name, "role": role, "arch": arch,
                "layer": layer, "coords": coords, "size": len(raw),
            })
    return points


# ===========================================================================
# Tests
# ===========================================================================


class TestDataAvailable:
    def test_sufficient_tensors(self, all_tensor_data):
        assert len(all_tensor_data) >= 100

    def test_layer_range(self, all_tensor_data):
        layers = set(p["layer"] for p in all_tensor_data)
        print(f"\n  Layer range: {min(layers)}..{max(layers)} ({len(layers)} unique)")
        assert len(layers) >= 10


class TestLayerCorrelation:
    """Core test: do ghost coordinates track layer position?"""

    def test_all_correlations(self, all_tensor_data):
        """Pearson and Spearman correlation of each coordinate vs layer_id."""
        layers = np.array([p["layer"] for p in all_tensor_data], dtype=np.float64)
        cmplx = np.array([p["coords"]["complexity"] for p in all_tensor_data])
        pred = np.array([p["coords"]["predictability"] for p in all_tensor_data])
        loc = np.array([p["coords"]["locality"] for p in all_tensor_data])

        print(f"\n  Ghost coordinate vs layer_id (all tensors, n={len(layers)}):")
        print(f"  {'dimension':<16} {'pearson':>8} {'spearman':>9} {'p-value':>9}")
        print(f"  {'-'*16} {'-'*8} {'-'*9} {'-'*9}")

        for name, vals in [("complexity", cmplx), ("predictability", pred), ("locality", loc)]:
            r_p = float(np.corrcoef(layers, vals)[0, 1])
            r_s, p_val = sp_stats.spearmanr(layers, vals)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            print(f"  {name:<16} {r_p:>+8.4f} {float(r_s):>+9.4f} {float(p_val):>9.2e} {sig}")

    def test_per_arch_correlation(self, all_tensor_data):
        """Same correlation, split by architecture."""
        for arch in ["mamba", "transformer"]:
            subset = [p for p in all_tensor_data if p["arch"] == arch]
            if len(subset) < 10:
                continue
            layers = np.array([p["layer"] for p in subset], dtype=np.float64)
            cmplx = np.array([p["coords"]["complexity"] for p in subset])
            pred = np.array([p["coords"]["predictability"] for p in subset])
            loc = np.array([p["coords"]["locality"] for p in subset])

            print(f"\n  {arch} (n={len(subset)}):")
            for name, vals in [("complexity", cmplx), ("predictability", pred), ("locality", loc)]:
                r_s, p_val = sp_stats.spearmanr(layers, vals)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                print(f"    {name:<16} spearman={float(r_s):>+.4f}  p={float(p_val):.2e} {sig}")


class TestPerRoleDrift:
    """Does each role drift systematically across layers?"""

    def test_role_layer_drift(self, all_tensor_data):
        """For each role, correlate ghost coordinates with layer position."""
        by_role = defaultdict(list)
        for p in all_tensor_data:
            by_role[p["role"]].append(p)

        print(f"\n  Per-role layer drift (Spearman correlation):")
        print(f"  {'role':<16} {'n':>4}  {'cmplx↔layer':>12} {'pred↔layer':>12} {'loc↔layer':>12}")
        print(f"  {'-'*16} {'-'*4}  {'-'*12} {'-'*12} {'-'*12}")

        drift_results = []
        for role in sorted(by_role.keys()):
            pts = by_role[role]
            if len(pts) < 5:
                continue
            layers = np.array([p["layer"] for p in pts], dtype=np.float64)
            cmplx = np.array([p["coords"]["complexity"] for p in pts])
            pred = np.array([p["coords"]["predictability"] for p in pts])
            loc = np.array([p["coords"]["locality"] for p in pts])

            rs_c, p_c = sp_stats.spearmanr(layers, cmplx)
            rs_p, p_p = sp_stats.spearmanr(layers, pred)
            rs_l, p_l = sp_stats.spearmanr(layers, loc)

            def fmt(r, p):
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                return f"{float(r):>+.3f}{sig:>4}"

            print(f"  {role:<16} {len(pts):>4}  {fmt(rs_c, p_c):>12} {fmt(rs_p, p_p):>12} {fmt(rs_l, p_l):>12}")
            drift_results.append({
                "role": role, "n": len(pts),
                "cmplx_r": float(rs_c), "cmplx_p": float(p_c),
                "pred_r": float(rs_p), "pred_p": float(p_p),
                "loc_r": float(rs_l), "loc_p": float(p_l),
            })

        # Count significant drifts
        sig_drifts = sum(1 for d in drift_results
                        if d["cmplx_p"] < 0.05 or d["pred_p"] < 0.05 or d["loc_p"] < 0.05)
        print(f"\n  Roles with significant layer drift: {sig_drifts}/{len(drift_results)}")


class TestReport:
    def test_full_report(self, all_tensor_data):
        layers_all = np.array([p["layer"] for p in all_tensor_data], dtype=np.float64)
        cmplx_all = np.array([p["coords"]["complexity"] for p in all_tensor_data])
        pred_all = np.array([p["coords"]["predictability"] for p in all_tensor_data])
        loc_all = np.array([p["coords"]["locality"] for p in all_tensor_data])

        rs_c, pc = sp_stats.spearmanr(layers_all, cmplx_all)
        rs_p, pp = sp_stats.spearmanr(layers_all, pred_all)
        rs_l, pl = sp_stats.spearmanr(layers_all, loc_all)

        # Per-arch
        arch_results = {}
        for arch in ["mamba", "transformer"]:
            subset = [p for p in all_tensor_data if p["arch"] == arch]
            if len(subset) < 10:
                continue
            ly = np.array([p["layer"] for p in subset], dtype=np.float64)
            c = np.array([p["coords"]["complexity"] for p in subset])
            p = np.array([p["coords"]["predictability"] for p in subset])
            l = np.array([p["coords"]["locality"] for p in subset])
            arch_results[arch] = {
                "n": len(subset),
                "layers": f"{int(ly.min())}..{int(ly.max())}",
                "cmplx": sp_stats.spearmanr(ly, c),
                "pred": sp_stats.spearmanr(ly, p),
                "loc": sp_stats.spearmanr(ly, l),
            }

        # Per-role drift count
        by_role = defaultdict(list)
        for pt in all_tensor_data:
            by_role[pt["role"]].append(pt)
        sig_roles = 0
        total_roles = 0
        strongest_drift = ("", "", 0.0)
        for role, pts in by_role.items():
            if len(pts) < 5:
                continue
            total_roles += 1
            ly = np.array([p["layer"] for p in pts], dtype=np.float64)
            for dim_name, dim_vals in [("complexity", [p["coords"]["complexity"] for p in pts]),
                                        ("predictability", [p["coords"]["predictability"] for p in pts]),
                                        ("locality", [p["coords"]["locality"] for p in pts])]:
                r, pv = sp_stats.spearmanr(ly, dim_vals)
                if pv < 0.05:
                    sig_roles += 1
                    if abs(r) > abs(strongest_drift[2]):
                        strongest_drift = (role, dim_name, float(r))
                    break  # Count role once

        print(f"""
================================================================================
PHASE 0.14: LAYER-POSITION GHOST TEST
================================================================================

Tensors: {len(all_tensor_data)}
Architectures: {len(arch_results)} ({', '.join(arch_results.keys())})
Layer range: {int(layers_all.min())}..{int(layers_all.max())}

body_opened: false
labels_used_for_coordinates: NONE

--- Global correlation (all tensors, all roles pooled) ---
  {'dimension':<16} {'spearman':>9} {'p-value':>9}
  {"complexity":<16} {float(rs_c):>+9.4f} {float(pc):>9.2e} {'***' if pc < 0.001 else ''}
  {"predictability":<16} {float(rs_p):>+9.4f} {float(pp):>9.2e} {'***' if pp < 0.001 else ''}
  {"locality":<16} {float(rs_l):>+9.4f} {float(pl):>9.2e} {'***' if pl < 0.001 else ''}

--- Per-architecture ---""")
        for arch, res in arch_results.items():
            rc, pc = res["cmplx"]
            rp, pp = res["pred"]
            rl, pl = res["loc"]
            print(f"  {arch} (n={res['n']}, layers {res['layers']}):")
            print(f"    complexity:     rho={float(rc):>+.4f}  p={float(pc):.2e}")
            print(f"    predictability: rho={float(rp):>+.4f}  p={float(pp):.2e}")
            print(f"    locality:       rho={float(rl):>+.4f}  p={float(pl):.2e}")

        print(f"""
--- Per-role drift ---
  Roles with significant (p<0.05) layer drift: {sig_roles}/{total_roles}
  Strongest drift: {strongest_drift[0]} {strongest_drift[1]} rho={strongest_drift[2]:+.4f}

================================================================================
VERDICT:
""")
        any_global_sig = pc < 0.05 or pp < 0.05 or pl < 0.05
        any_strong = abs(rs_c) > 0.3 or abs(rs_p) > 0.3 or abs(rs_l) > 0.3

        if any_strong and any_global_sig:
            best_dim = max([("complexity", abs(rs_c)), ("predictability", abs(rs_p)),
                           ("locality", abs(rs_l))], key=lambda x: x[1])
            print(f"  Ghost coordinates TRACK layer position.")
            print(f"  Best dimension: {best_dim[0]} (|rho|={best_dim[1]:.3f})")
            print(f"  Ghost is mapping model GEOMETRY, not just classifying tensors.")
        elif any_global_sig:
            print(f"  Weak but significant layer tracking detected.")
            print(f"  Ghost coordinates drift with layer depth but effect is small.")
        elif sig_roles > total_roles * 0.5:
            print(f"  Layer tracking is ROLE-SPECIFIC, not global.")
            print(f"  {sig_roles}/{total_roles} roles show significant drift individually.")
            print(f"  Ghost sees layer position within specific tensor types.")
        else:
            print(f"  Ghost coordinates do NOT track layer position.")
            print(f"  The coordinate system classifies tensor TYPE, not POSITION.")

        print("================================================================================")
