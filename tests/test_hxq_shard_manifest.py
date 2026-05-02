"""Tests for HXQ asset validation on shard/cartridge manifests.

Contract:
  - Shard manifests have codec field
  - Baseline codecs (q5_k_m, q8_0) do not require HXQ validation
  - HXQ codecs require tensor fidelity receipt + behavioral eval receipt
  - HXQ assets cannot be promoted without both receipts
  - Quarantined status blocks routing
  - is_hxq_codec/is_baseline_codec correctly classify
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.hxq_asset import (
    is_hxq_codec, is_baseline_codec, validate_hxq_asset,
    can_promote, select_codec_for_target, HXQAssetReceipt,
    BASELINE_CODECS, HXQ_CODECS,
)
from cell.shard_pool import ShardManifest, ShardPool


def test_is_hxq_codec():
    """HXQ codec classification."""
    assert is_hxq_codec("hxq_affine_6")
    assert is_hxq_codec("hxq_affine_g128")
    assert not is_hxq_codec("q5_k_m")
    assert not is_hxq_codec("q8_0")
    assert not is_hxq_codec("")


def test_is_baseline_codec():
    """Baseline codec classification."""
    assert is_baseline_codec("q5_k_m")
    assert is_baseline_codec("q6_k")
    assert is_baseline_codec("q8_0")
    assert is_baseline_codec("fp16")
    assert not is_baseline_codec("hxq_affine_6")
    assert not is_baseline_codec("")


def test_shard_manifest_has_codec():
    """ShardManifest includes codec field."""
    m = ShardManifest(model_id="test", codec="q5_k_m", quant_type="q5_k_m")
    assert m.codec == "q5_k_m"


def test_shard_manifest_codec_from_file():
    """ShardManifest.from_file reads codec field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = os.path.join(tmpdir, "test_shard")
        os.makedirs(shard_dir)
        gguf = os.path.join(shard_dir, "test.gguf")
        with open(gguf, "w") as f:
            f.write("FAKE")

        manifest = {
            "model_id": "test_shard",
            "codec": "hxq_affine_6",
            "quant_type": "hxq_affine_6",
            "shard_paths": [gguf],
            "status": "candidate",
            "helix_codec_receipt": "helix_receipt.json",
            "behavioral_eval_receipt": "eval_receipt.json",
            "fallback_shard": "test_shard_q5",
        }
        manifest_path = os.path.join(shard_dir, "shard_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        m = ShardManifest.from_file(manifest_path)
        assert m.codec == "hxq_affine_6"
        assert m.helix_codec_receipt == "helix_receipt.json"
        assert m.behavioral_eval_receipt == "eval_receipt.json"
        assert m.fallback_shard == "test_shard_q5"


def test_shard_manifest_codec_defaults_to_quant_type():
    """If codec not set in manifest, defaults to quant_type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = os.path.join(tmpdir, "test")
        os.makedirs(shard_dir)
        manifest = {
            "model_id": "test",
            "quant_type": "q5_k_m",
            "shard_paths": [],
            "status": "candidate",
        }
        path = os.path.join(shard_dir, "shard_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f)

        m = ShardManifest.from_file(path)
        assert m.codec == "q5_k_m"


def test_validate_baseline_skips_hxq_check():
    """Baseline codecs skip HXQ validation."""
    result = validate_hxq_asset("/tmp", "q5_k_m")
    assert result["valid"] is True
    assert "Not HXQ" in result.get("note", "")


def test_validate_hxq_missing_receipts():
    """HXQ codec without receipts fails validation."""
    result = validate_hxq_asset("/tmp", "hxq_affine_6")
    assert result["valid"] is False
    assert len(result["issues"]) >= 2  # missing both receipts


def test_validate_hxq_with_valid_receipts():
    """HXQ codec with valid receipts passes validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create helix receipt
        helix_receipt = {
            "asset_id": "test_asset",
            "codec": "hxq_affine_6",
            "cosine_min": 0.9995,
            "sha256_compressed": "abc123",
        }
        helix_path = os.path.join(tmpdir, "helix_receipt.json")
        with open(helix_path, "w") as f:
            json.dump(helix_receipt, f)

        # Create behavioral receipt
        behavioral_path = os.path.join(tmpdir, "eval_receipt.json")
        with open(behavioral_path, "w") as f:
            json.dump({"pass": True}, f)

        result = validate_hxq_asset(
            tmpdir, "hxq_affine_6",
            helix_receipt_path="helix_receipt.json",
            behavioral_receipt_path="eval_receipt.json",
        )
        assert result["valid"] is True
        assert len(result["issues"]) == 0


def test_validate_hxq_low_cosine_fails():
    """HXQ with cosine below threshold fails validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        helix_receipt = {
            "asset_id": "test",
            "codec": "hxq_affine_6",
            "cosine_min": 0.990,  # below 0.998 threshold
            "sha256_compressed": "abc",
        }
        with open(os.path.join(tmpdir, "helix.json"), "w") as f:
            json.dump(helix_receipt, f)
        with open(os.path.join(tmpdir, "eval.json"), "w") as f:
            json.dump({"pass": True}, f)

        result = validate_hxq_asset(
            tmpdir, "hxq_affine_6",
            helix_receipt_path="helix.json",
            behavioral_receipt_path="eval.json",
        )
        assert result["valid"] is False
        assert any("fidelity" in i.lower() for i in result["issues"])


def test_can_promote_baseline():
    """Baseline codec can promote with behavioral eval only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = os.path.join(tmpdir, "eval.json")
        with open(eval_path, "w") as f:
            json.dump({"pass": True}, f)

        result = can_promote("q5_k_m", behavioral_receipt_path=eval_path)
        assert result["promotable"] is True


def test_can_promote_baseline_without_eval():
    """Baseline codec cannot promote without behavioral eval."""
    result = can_promote("q5_k_m")
    assert result["promotable"] is False


def test_can_promote_hxq_complete():
    """HXQ codec can promote with both receipts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        helix_receipt = {
            "asset_id": "test",
            "codec": "hxq_affine_6",
            "cosine_min": 0.9995,
            "sha256_compressed": "abc",
        }
        with open(os.path.join(tmpdir, "helix.json"), "w") as f:
            json.dump(helix_receipt, f)
        with open(os.path.join(tmpdir, "eval.json"), "w") as f:
            json.dump({"pass": True}, f)

        result = can_promote(
            "hxq_affine_6",
            helix_receipt_path="helix.json",
            behavioral_receipt_path="eval.json",
            manifest_dir=tmpdir,
        )
        assert result["promotable"] is True


def test_can_promote_hxq_missing_helix():
    """HXQ codec cannot promote without tensor fidelity receipt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "eval.json"), "w") as f:
            json.dump({"pass": True}, f)

        result = can_promote(
            "hxq_affine_6",
            behavioral_receipt_path="eval.json",
            manifest_dir=tmpdir,
        )
        assert result["promotable"] is False


def test_can_promote_prompt_pack():
    """Non-model codecs (prompt_pack) are always promotable."""
    result = can_promote("prompt_pack")
    assert result["promotable"] is True


def test_select_codec_for_target():
    """Codec recommendation for deployment targets."""
    assert select_codec_for_target("gpu_edge") == "hxq_affine_6"
    assert select_codec_for_target("cpu_fallback") == "hxq_affine_g128"
    assert select_codec_for_target("baseline") == "q5_k_m"
    assert select_codec_for_target("pod") == "q8_0"


def test_hxq_receipt_from_file():
    """HXQAssetReceipt loads from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "asset_id": "test_shard_001",
            "asset_type": "llm_weight_shard",
            "codec": "hxq_affine_6",
            "group_size": 128,
            "sha256_original": "aaa",
            "sha256_compressed": "bbb",
            "cosine_min": 0.9994,
            "ppl_baseline": 8.22,
            "ppl_compressed": 8.30,
            "ppl_delta_pct": 0.97,
            "behavioral_eval_pass": True,
            "runtime_status": "candidate",
        }
        path = os.path.join(tmpdir, "receipt.json")
        with open(path, "w") as f:
            json.dump(data, f)

        receipt = HXQAssetReceipt.from_file(path)
        assert receipt.asset_id == "test_shard_001"
        assert receipt.codec == "hxq_affine_6"
        assert receipt.cosine_min == 0.9994
        assert receipt.tensor_fidelity_pass()  # 0.9994 >= 0.998


def test_hxq_receipt_fidelity_fail():
    """Tensor fidelity check fails when cosine is too low."""
    receipt = HXQAssetReceipt(asset_id="test", cosine_min=0.990)
    assert not receipt.tensor_fidelity_pass()


def test_quarantined_shard_not_routed():
    """Quarantined shard is not returned by route."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = os.path.join(tmpdir, "quarantined_shard")
        os.makedirs(shard_dir)
        gguf = os.path.join(shard_dir, "model.gguf")
        with open(gguf, "w") as f:
            f.write("FAKE")
        manifest = {
            "model_id": "quarantined_shard",
            "shard_paths": [gguf],
            "activation_intents": ["test_intent"],
            "status": "quarantined",
            "codec": "hxq_affine_6",
        }
        with open(os.path.join(shard_dir, "shard_manifest.json"), "w") as f:
            json.dump(manifest, f)

        pool = ShardPool(tmpdir)
        assert pool.route("test_intent") is None


def test_list_shards_includes_codec():
    """list_shards includes codec and fallback_shard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = os.path.join(tmpdir, "test_shard")
        os.makedirs(shard_dir)
        gguf = os.path.join(shard_dir, "model.gguf")
        with open(gguf, "w") as f:
            f.write("FAKE")
        manifest = {
            "model_id": "test_shard",
            "shard_paths": [gguf],
            "activation_intents": ["test"],
            "status": "active",
            "codec": "hxq_affine_6",
            "fallback_shard": "test_shard_q5",
        }
        with open(os.path.join(shard_dir, "shard_manifest.json"), "w") as f:
            json.dump(manifest, f)

        pool = ShardPool(tmpdir)
        listing = pool.list_shards()
        assert len(listing) == 1
        assert listing[0]["codec"] == "hxq_affine_6"
        assert listing[0]["fallback_shard"] == "test_shard_q5"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
