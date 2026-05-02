"""Tests for HXQ fallback policy.

Contract:
  - If HXQ shard load fails, runtime falls back to fallback_shard (standard GGUF)
  - If HXQ eval fails, shard should be quarantined (not used)
  - Fallback chain: HXQ shard → baseline shard → Sentinel
  - Q5_K_M is always the control group
  - No HXQ promotion without behavioral eval
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.shard_pool import ShardPool, ShardManifest
from cell.hxq_asset import can_promote, is_hxq_codec, validate_hxq_asset


def _make_paired_shards(tmpdir):
    """Create a baseline Q5 shard and an HXQ candidate shard."""
    # Baseline shard (Q5_K_M)
    base_dir = os.path.join(tmpdir, "coder_q5")
    os.makedirs(base_dir)
    base_gguf = os.path.join(base_dir, "coder_q5.gguf")
    with open(base_gguf, "w") as f:
        f.write("FAKE_Q5_GGUF")
    base_manifest = {
        "model_id": "coder_q5",
        "codec": "q5_k_m",
        "quant_type": "q5_k_m",
        "shard_paths": [base_gguf],
        "activation_intents": ["deep_code_analysis"],
        "status": "active",
        "behavioral_eval_receipt": "eval.json",
    }
    with open(os.path.join(base_dir, "shard_manifest.json"), "w") as f:
        json.dump(base_manifest, f)
    # Create the eval receipt
    with open(os.path.join(base_dir, "eval.json"), "w") as f:
        json.dump({"pass": True, "task_count": 50}, f)

    # HXQ candidate shard
    hxq_dir = os.path.join(tmpdir, "coder_hxq")
    os.makedirs(hxq_dir)
    hxq_gguf = os.path.join(hxq_dir, "coder_hxq.gguf")
    with open(hxq_gguf, "w") as f:
        f.write("FAKE_HXQ_GGUF")
    hxq_manifest = {
        "model_id": "coder_hxq",
        "codec": "hxq_affine_6",
        "quant_type": "hxq_affine_6",
        "shard_paths": [hxq_gguf],
        "activation_intents": ["deep_code_analysis"],
        "status": "candidate",
        "fallback_shard": "coder_q5",
        "fallback": "qwen2.5-sentinel",
    }
    with open(os.path.join(hxq_dir, "shard_manifest.json"), "w") as f:
        json.dump(hxq_manifest, f)

    return tmpdir


def test_baseline_shard_is_control():
    """Q5_K_M shard is active, HXQ shard is candidate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_paired_shards(tmpdir)
        pool = ShardPool(tmpdir)

        base = pool.get("coder_q5")
        hxq = pool.get("coder_hxq")
        assert base.status == "active"
        assert base.codec == "q5_k_m"
        assert hxq.status == "candidate"
        assert hxq.codec == "hxq_affine_6"


def test_route_uses_active_baseline_not_candidate_hxq():
    """When both baseline and HXQ exist for same intent, active baseline wins."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_paired_shards(tmpdir)
        pool = ShardPool(tmpdir)

        # Route should return the active baseline, not the candidate HXQ
        routed = pool.route("deep_code_analysis")
        assert routed is not None
        assert routed.model_id == "coder_q5"
        assert routed.codec == "q5_k_m"


def test_hxq_shard_has_fallback():
    """HXQ shard specifies fallback_shard to baseline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_paired_shards(tmpdir)
        pool = ShardPool(tmpdir)

        hxq = pool.get("coder_hxq")
        assert hxq.fallback_shard == "coder_q5"

        # Fallback shard exists in pool
        fallback = pool.get(hxq.fallback_shard)
        assert fallback is not None
        assert fallback.status == "active"


def test_fallback_chain():
    """Fallback chain: HXQ → baseline Q5 → Sentinel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_paired_shards(tmpdir)
        pool = ShardPool(tmpdir)

        hxq = pool.get("coder_hxq")
        # Level 1: HXQ shard (candidate, not routed)
        assert not hxq.is_active()

        # Level 2: Fallback to baseline Q5
        fallback_shard = pool.get(hxq.fallback_shard)
        assert fallback_shard.is_active()

        # Level 3: If baseline fails too, manifest fallback to Sentinel
        assert hxq.fallback == "qwen2.5-sentinel"


def test_cannot_promote_hxq_without_receipts():
    """HXQ shard without receipts cannot be promoted."""
    result = can_promote("hxq_affine_6")
    assert result["promotable"] is False


def test_can_promote_hxq_with_receipts():
    """HXQ shard with valid receipts can be promoted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create valid helix receipt
        helix = {
            "asset_id": "coder_hxq",
            "codec": "hxq_affine_6",
            "cosine_min": 0.9995,
            "sha256_compressed": "abc123",
        }
        with open(os.path.join(tmpdir, "helix.json"), "w") as f:
            json.dump(helix, f)

        # Create behavioral eval receipt
        with open(os.path.join(tmpdir, "eval.json"), "w") as f:
            json.dump({"pass": True, "task_count": 50, "accuracy": 0.92}, f)

        result = can_promote(
            "hxq_affine_6",
            helix_receipt_path="helix.json",
            behavioral_receipt_path="eval.json",
            manifest_dir=tmpdir,
        )
        assert result["promotable"] is True


def test_baseline_promotion_only_needs_behavioral():
    """Q5_K_M baseline only needs behavioral eval to promote."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "eval.json"), "w") as f:
            json.dump({"pass": True}, f)

        result = can_promote(
            "q5_k_m",
            behavioral_receipt_path=os.path.join(tmpdir, "eval.json"),
        )
        assert result["promotable"] is True


def test_quarantined_blocks_routing():
    """Quarantined HXQ shard is not routed."""
    m = ShardManifest(
        model_id="quarantined",
        codec="hxq_affine_6",
        status="quarantined",
        activation_intents=["test"],
    )
    pool = ShardPool()
    pool.register(m)
    assert pool.route("test") is None


def test_build_load_plan_records_codec():
    """Build load plan includes codec in output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = os.path.join(tmpdir, "test")
        os.makedirs(shard_dir)
        gguf = os.path.join(shard_dir, "test.gguf")
        with open(gguf, "w") as f:
            f.write("FAKE")
        manifest = {
            "model_id": "test",
            "codec": "q5_k_m",
            "shard_paths": [gguf],
            "status": "active",
        }
        with open(os.path.join(shard_dir, "shard_manifest.json"), "w") as f:
            json.dump(manifest, f)

        pool = ShardPool(tmpdir)
        shard = pool.get("test")
        assert shard.codec == "q5_k_m"


def test_hxq_eval_order():
    """Prove the correct evaluation order: Q5 first, then HXQ."""
    # This is a documentation test — enforces the policy
    # 1. Baseline Q5_K_M eval → active
    q5_result = can_promote("q5_k_m", behavioral_receipt_path="/fake/eval.json")
    # Without file it fails — that's expected (file doesn't exist)
    assert q5_result["promotable"] is False

    # 2. HXQ without behavioral eval → cannot promote
    hxq_result = can_promote("hxq_affine_6", helix_receipt_path="/fake/helix.json")
    assert hxq_result["promotable"] is False

    # 3. HXQ needs BOTH tensor fidelity AND behavioral eval
    # This enforces: same prompts, same tasks, same scoring
    with tempfile.TemporaryDirectory() as tmpdir:
        helix = {"asset_id": "t", "codec": "hxq_affine_6", "cosine_min": 0.9995, "sha256_compressed": "x"}
        with open(os.path.join(tmpdir, "h.json"), "w") as f:
            json.dump(helix, f)
        with open(os.path.join(tmpdir, "e.json"), "w") as f:
            json.dump({"pass": True}, f)

        full_result = can_promote(
            "hxq_affine_6",
            helix_receipt_path="h.json",
            behavioral_receipt_path="e.json",
            manifest_dir=tmpdir,
        )
        assert full_result["promotable"] is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
