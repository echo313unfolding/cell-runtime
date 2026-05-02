"""Tests for shard pool — monolithic model sharding manifests.

Contract:
  - ShardPool loads manifests from shard directories
  - Routes intents to correct shard
  - Candidate shards are NOT routed
  - Resource checks verify fit
  - Build load plan returns llama-server args
  - Load events are logged
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.shard_pool import ShardPool, ShardManifest


def _make_shard(tmpdir, model_id, intents, status="active",
                vram_mb=2500, ram_mb=8000, gpu_layers=20):
    """Create a minimal shard with a fake GGUF file."""
    shard_dir = os.path.join(tmpdir, model_id)
    os.makedirs(shard_dir, exist_ok=True)

    # Create a fake GGUF file
    gguf_path = os.path.join(shard_dir, f"{model_id}.gguf")
    with open(gguf_path, "w") as f:
        f.write("FAKE_GGUF_DATA")

    manifest = {
        "model_id": model_id,
        "role": "specialist",
        "backend": "llama_cpp",
        "load_mode": "cold",
        "shard_paths": [gguf_path],
        "quant_type": "q5_k_m",
        "offload_policy": {
            "gpu_layers": gpu_layers,
            "mmap": True,
        },
        "required_vram_mb": vram_mb,
        "required_ram_mb": ram_mb,
        "context_size": 4096,
        "activation_intents": intents,
        "fallback": "sentinel",
        "idle_unload_s": 300,
        "status": status,
        "system_prompt": "Test specialist.",
    }

    with open(os.path.join(shard_dir, "shard_manifest.json"), "w") as f:
        json.dump(manifest, f)

    return shard_dir


def test_load_manifests():
    """ShardPool loads manifests from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "shard_a", ["intent_a"])
        _make_shard(tmpdir, "shard_b", ["intent_b"])

        pool = ShardPool(tmpdir)
        assert len(pool) == 2
        assert pool.get("shard_a") is not None
        assert pool.get("shard_b") is not None


def test_route_by_intent():
    """Route matches intent to correct shard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "coder", ["deep_code_analysis", "patch_gen"])
        _make_shard(tmpdir, "research", ["research_task"])

        pool = ShardPool(tmpdir)
        shard = pool.route("deep_code_analysis")
        assert shard is not None
        assert shard.model_id == "coder"


def test_candidate_not_routed():
    """Candidate shard is not returned by route."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "cand", ["some_intent"], status="candidate")
        pool = ShardPool(tmpdir)
        assert pool.route("some_intent") is None


def test_disabled_not_routed():
    """Disabled shard is not returned by route."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "disabled", ["other_intent"], status="disabled")
        pool = ShardPool(tmpdir)
        assert pool.route("other_intent") is None


def test_check_resources_fits():
    """Resource check passes when resources are sufficient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "small", ["test"], vram_mb=2000, ram_mb=4000)
        pool = ShardPool(tmpdir)

        result = pool.check_resources("small", available_vram_mb=4096, available_ram_mb=16384)
        assert result["fits"] is True


def test_check_resources_vram_insufficient():
    """Resource check fails when VRAM is insufficient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "big", ["test"], vram_mb=8000, ram_mb=4000)
        pool = ShardPool(tmpdir)

        result = pool.check_resources("big", available_vram_mb=4096)
        assert result["fits"] is False
        assert "VRAM" in result.get("error", "")


def test_check_resources_ram_insufficient():
    """Resource check fails when RAM is insufficient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "rambig", ["test"], vram_mb=1000, ram_mb=32000)
        pool = ShardPool(tmpdir)

        result = pool.check_resources("rambig", available_vram_mb=4096, available_ram_mb=16384)
        assert result["fits"] is False
        assert "RAM" in result.get("error", "")


def test_check_resources_unknown():
    """Resource check for unknown shard returns error."""
    pool = ShardPool()
    result = pool.check_resources("nonexistent")
    assert result["fits"] is False


def test_build_load_plan():
    """Build load plan returns llama-server args."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "coder", ["test"], gpu_layers=20)
        pool = ShardPool(tmpdir)

        plan = pool.build_load_plan("coder", port=8090)
        assert "error" not in plan
        assert plan["model_id"] == "coder"
        assert "--model" in plan["llama_args"]
        assert "--n-gpu-layers" in plan["llama_args"]
        assert "20" in plan["llama_args"]
        assert "--port" in plan["llama_args"]
        assert "8090" in plan["llama_args"]


def test_build_load_plan_unknown():
    """Build load plan for unknown shard returns error."""
    pool = ShardPool()
    plan = pool.build_load_plan("nonexistent")
    assert "error" in plan


def test_build_load_plan_candidate():
    """Build load plan for candidate shard returns error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "cand", ["test"], status="candidate")
        pool = ShardPool(tmpdir)
        plan = pool.build_load_plan("cand")
        assert "error" in plan


def test_manifest_fields():
    """ShardManifest exposes key fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "test_model", ["intent1"], gpu_layers=15)
        pool = ShardPool(tmpdir)
        shard = pool.get("test_model")

        assert shard.gpu_layers() == 15
        assert shard.uses_mmap() is True
        assert shard.primary_gguf() is not None
        assert shard.all_shards_exist() is True
        assert shard.total_size_mb() >= 0  # fake GGUF is tiny, rounds to 0.0 MB


def test_load_log():
    """Load events are recorded."""
    pool = ShardPool()
    pool.record_load("test_model", "load", wall_time_s=5.2)
    pool.record_load("test_model", "unload", wall_time_s=0.1)

    log = pool.get_load_log()
    assert len(log) == 2
    assert log[0]["event"] == "load"
    assert log[0]["wall_time_s"] == 5.2
    assert log[1]["event"] == "unload"


def test_list_shards():
    """list_shards returns metadata for all shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "a", ["ia"])
        _make_shard(tmpdir, "b", ["ib"], status="candidate")
        pool = ShardPool(tmpdir)

        listing = pool.list_shards()
        assert len(listing) == 2
        ids = {s["model_id"] for s in listing}
        assert "a" in ids
        assert "b" in ids


def test_intent_map():
    """intent_map returns complete mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_shard(tmpdir, "coder", ["deep_code", "patch_gen"])
        pool = ShardPool(tmpdir)

        imap = pool.intent_map()
        assert imap["deep_code"] == "coder"
        assert imap["patch_gen"] == "coder"


def test_register_directly():
    """register() adds a shard without directory scan."""
    pool = ShardPool()
    m = ShardManifest(
        model_id="direct_shard",
        activation_intents=["direct_intent"],
        status="active",
        shard_paths=["/tmp/fake.gguf"],
    )
    pool.register(m)
    assert pool.get("direct_shard") is not None


def test_real_shards_load():
    """Real shard manifests in cell-runtime/shards/ load correctly."""
    shard_dir = str(Path(__file__).parent.parent / "shards")
    if not os.path.isdir(shard_dir):
        return  # skip

    pool = ShardPool(shard_dir)
    assert len(pool) >= 2
    assert pool.get("qwen_coder_7b_q5_local") is not None
    assert pool.get("qwen_coder_14b_split_cpu_gpu") is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
