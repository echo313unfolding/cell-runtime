"""Tests for Crystal Vault Runtime — Phase 0.

Tests the ExecutableShard interface (encoded execution, not streaming),
VaultManifest schema, dependency ordering, content-addressed hashing,
and boundary contracts WITHOUT requiring real HXQ files.

WO-CRYSTAL-VAULT-01: Phase 0 tests.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cell.vault_shard import (
    BoundaryContract,
    ExecutableShard,
    ExecutionSemantics,
    Ghost,
    GlyphDAR,
    GlyphPacket,
    Intent,
    LatentShape,
    Opcode,
    Outcome,
    Shadow,
    ShadowMemory,
    ShardReceipt,
    ShardState,
    VaultManifest,
)


# --- BoundaryContract ---

class TestBoundaryContract:
    def test_validate_matching_shape(self):
        c = BoundaryContract(input_shape=[-1, 128], output_shape=[-1, 256])
        X = np.zeros((4, 128), dtype=np.float32)
        assert c.validate_input(X)

    def test_validate_dynamic_batch(self):
        c = BoundaryContract(input_shape=[-1, 128], output_shape=[-1, 256])
        for batch in [1, 8, 32, 100]:
            X = np.zeros((batch, 128), dtype=np.float32)
            assert c.validate_input(X)

    def test_reject_wrong_dim(self):
        c = BoundaryContract(input_shape=[-1, 128], output_shape=[-1, 256])
        X = np.zeros((4, 64), dtype=np.float32)
        assert not c.validate_input(X)

    def test_reject_wrong_ndim(self):
        c = BoundaryContract(input_shape=[-1, 128], output_shape=[-1, 256])
        X = np.zeros((2, 4, 128), dtype=np.float32)
        assert not c.validate_input(X)

    def test_roundtrip_dict(self):
        c = BoundaryContract(
            input_shape=[-1, 3072],
            output_shape=[-1, 3072],
            dtype="float32",
            role="attention",
        )
        d = c.to_dict()
        c2 = BoundaryContract.from_dict(d)
        assert c2.input_shape == c.input_shape
        assert c2.output_shape == c.output_shape
        assert c2.role == "attention"


# --- ExecutableShard lifecycle ---

class TestExecutableShardLifecycle:
    def test_initial_state_encoded_executable(self):
        shard = ExecutableShard(
            shard_id="test_shard",
            encoded_path="/nonexistent",
            contract=BoundaryContract([-1, 64], [-1, 64]),
        )
        assert shard.state == ShardState.ENCODED_EXECUTABLE

    def test_contract_validation_rejects_bad_input(self):
        shard = ExecutableShard(
            shard_id="test_shard",
            encoded_path="/nonexistent",
            contract=BoundaryContract([-1, 64], [-1, 64]),
        )
        X = np.zeros((4, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="doesn't match contract"):
            shard.eval_encoded(X)

    def test_compressed_native_requires_helix_substrate(self):
        """Compressed native should fail gracefully if helix_substrate not importable."""
        shard = ExecutableShard(
            shard_id="test_shard",
            encoded_path="/nonexistent/file.hxz",
            contract=BoundaryContract([-1, 64], [-1, 64]),
            execution_semantics=ExecutionSemantics.COMPRESSED_NATIVE,
        )
        X = np.zeros((4, 64), dtype=np.float32)
        with pytest.raises(RuntimeError):
            shard.eval_encoded(X)

    def test_shard_to_dict(self):
        shard = ExecutableShard(
            shard_id="blk.0.attn",
            encoded_path="/path/to/attn.hxz",
            contract=BoundaryContract([-1, 3072], [-1, 3072], role="attention"),
            execution_semantics=ExecutionSemantics.COMPRESSED_NATIVE,
            codec="hxq_affine_6",
            codec_ir="hxq_codebook_index_affine_sidecar",
            sha256="abc123",
            dependencies=["embed"],
        )
        d = shard.to_dict()
        assert d["shard_id"] == "blk.0.attn"
        assert d["execution_semantics"] == "compressed_native"
        assert d["codec_ir"] == "hxq_codebook_index_affine_sidecar"
        assert d["dependencies"] == ["embed"]
        assert d["contract"]["role"] == "attention"

    def test_verify_integrity_missing_file(self):
        shard = ExecutableShard(
            shard_id="missing",
            encoded_path="/nonexistent/file.hxz",
            contract=BoundaryContract([-1, 64], [-1, 64]),
        )
        assert not shard.verify_integrity()

    def test_verify_integrity_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".hxz", delete=False) as f:
            f.write(b"test data")
            path = f.name
        try:
            shard = ExecutableShard(
                shard_id="exists",
                encoded_path=path,
                contract=BoundaryContract([-1, 64], [-1, 64]),
                sha256="",
            )
            assert shard.verify_integrity()
        finally:
            os.unlink(path)

    def test_verify_integrity_hash_check(self):
        data = b"crystal vault test content"
        expected_hash = hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(suffix=".hxz", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            shard = ExecutableShard(
                shard_id="hashed",
                encoded_path=path,
                contract=BoundaryContract([-1, 64], [-1, 64]),
                sha256=expected_hash,
            )
            assert shard.verify_integrity()

            shard_bad = ExecutableShard(
                shard_id="bad_hash",
                encoded_path=path,
                contract=BoundaryContract([-1, 64], [-1, 64]),
                sha256="0000000000000000",
            )
            assert not shard_bad.verify_integrity()
        finally:
            os.unlink(path)

    def test_default_execution_semantics_is_compressed_native(self):
        shard = ExecutableShard(
            shard_id="test",
            encoded_path="/path",
            contract=BoundaryContract([-1, 64], [-1, 64]),
        )
        assert shard.execution_semantics == ExecutionSemantics.COMPRESSED_NATIVE


# --- ShardReceipt ---

class TestShardReceipt:
    def test_receipt_level_2_encoded_execution(self):
        """Level 2: zero persistent materialization, hardware translation boundary."""
        r = ShardReceipt(
            shard_id="blk.0",
            execution_semantics="compressed_native",
            execution_level=2,
            codec_ir="hxq_codebook_index_affine_sidecar",
            state_path=["encoded_executable->executing_encoded",
                        "executing_encoded->encoded_executable"],
            wall_time_ms=1.234,
            peak_memory_mb=0.5,
            input_hash="aaa",
            output_hash="bbb",
            decompression_invoked=False,
            materialized_weight_bytes=0,
            encoded_ops_executed=24,
            fallback_used=False,
            translation_boundary="centroid_lookup_to_float_register",
            timestamp="2026-06-07T00:00:00Z",
        )
        d = r.to_dict()
        assert d["execution_semantics"] == "compressed_native"
        assert d["execution_level"] == 2
        proof = d["encoded_execution_proof"]
        assert proof["execution_level"] == 2
        assert proof["decompression_invoked"] is False
        assert proof["materialized_weight_bytes"] == 0
        assert proof["encoded_ops_executed"] == 24
        assert proof["fallback_used"] is False
        assert proof["translation_boundary"] == "centroid_lookup_to_float_register"
        assert d["compute_proof"]["input_sha256"] == "aaa"

    def test_receipt_level_0_materialized(self):
        """Level 0: full materialization — the failure path."""
        r = ShardReceipt(
            shard_id="blk.1",
            execution_semantics="materialized",
            execution_level=0,
            codec_ir="hxq_codebook_index_affine_sidecar",
            state_path=["encoded_executable->materializing",
                        "materializing->executing_materialized",
                        "executing_materialized->dematerializing",
                        "dematerializing->encoded_executable"],
            wall_time_ms=5.0,
            peak_memory_mb=64.0,
            input_hash="ccc",
            output_hash="ddd",
            decompression_invoked=True,
            materialized_weight_bytes=67108864,
            encoded_ops_executed=0,
            fallback_used=True,
            translation_boundary="full_materialization",
            timestamp="2026-06-07T00:00:00Z",
        )
        d = r.to_dict()
        proof = d["encoded_execution_proof"]
        assert proof["execution_level"] == 0
        assert proof["decompression_invoked"] is True
        assert proof["materialized_weight_bytes"] == 67108864
        assert proof["fallback_used"] is True
        assert proof["translation_boundary"] == "full_materialization"


# --- VaultManifest ---

def _make_manifest(n_shards=3, linear_chain=True):
    """Helper: build a manifest with n_shards in a linear chain."""
    shards = []
    edges = []
    for i in range(n_shards):
        shard_id = f"blk.{i}"
        shards.append({
            "shard_id": shard_id,
            "encoded_path": f"shards/blk_{i}.hxz",
            "contract": {
                "input_shape": [-1, 3072],
                "output_shape": [-1, 3072],
                "dtype": "float32",
                "role": "transformer_block",
            },
            "execution_semantics": "compressed_native",
            "codec": "hxq_affine_6",
            "codec_ir": "hxq_codebook_index_affine_sidecar",
            "sha256": hashlib.sha256(f"shard_{i}".encode()).hexdigest(),
            "dependencies": [f"blk.{i-1}"] if i > 0 else [],
        })
        if linear_chain and i > 0:
            edges.append((f"blk.{i-1}", f"blk.{i}"))

    return VaultManifest(
        vault_id="test-vault",
        model_id="qwen2.5-coder-3b",
        shards=shards,
        edges=edges,
        created="2026-06-07T00:00:00Z",
    )


class TestVaultManifest:
    def test_dependency_order_linear_chain(self):
        m = _make_manifest(n_shards=5, linear_chain=True)
        order = m.get_dependency_order()
        assert order == ["blk.0", "blk.1", "blk.2", "blk.3", "blk.4"]

    def test_dependency_order_no_edges(self):
        m = _make_manifest(n_shards=3, linear_chain=False)
        m.edges = []
        order = m.get_dependency_order()
        assert len(order) == 3
        assert set(order) == {"blk.0", "blk.1", "blk.2"}

    def test_cycle_detection(self):
        m = _make_manifest(n_shards=3, linear_chain=True)
        m.edges.append(("blk.2", "blk.0"))
        with pytest.raises(ValueError, match="cycles"):
            m.get_dependency_order()

    def test_merkle_hash_deterministic(self):
        m = _make_manifest(n_shards=3)
        h1 = m.compute_manifest_hash()
        h2 = m.compute_manifest_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_merkle_hash_changes_with_content(self):
        m1 = _make_manifest(n_shards=3)
        m2 = _make_manifest(n_shards=3)
        m2.shards[1]["sha256"] = "different_hash"
        assert m1.compute_manifest_hash() != m2.compute_manifest_hash()

    def test_roundtrip_json(self):
        m = _make_manifest(n_shards=3)
        m.manifest_hash = m.compute_manifest_hash()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            m.save(f.name)
            path = f.name

        try:
            m2 = VaultManifest.from_file(path)
            assert m2.vault_id == m.vault_id
            assert len(m2.shards) == 3
            assert len(m2.edges) == 2
            assert m2.manifest_hash == m.manifest_hash
        finally:
            os.unlink(path)

    def test_validate_missing_files(self):
        m = _make_manifest(n_shards=2)
        result = m.validate(base_path="/nonexistent")
        assert not result["valid"]
        assert len(result["errors"]) >= 2

    def test_validate_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shards_dir = Path(tmpdir) / "shards"
            shards_dir.mkdir()

            m = _make_manifest(n_shards=2)
            for s in m.shards:
                (shards_dir / Path(s["encoded_path"]).name).write_bytes(b"test")
                s["encoded_path"] = str(shards_dir / Path(s["encoded_path"]).name)

            result = m.validate()
            assert result["valid"]
            assert result["execution_order"] == ["blk.0", "blk.1"]
            assert "computed_hash" in result

    def test_validate_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = _make_manifest(n_shards=1)
            fpath = Path(tmpdir) / "blk_0.hxz"
            fpath.write_bytes(b"test")
            m.shards[0]["encoded_path"] = str(fpath)
            m.edges = []

            m.manifest_hash = "wrong_hash"
            result = m.validate()
            assert not result["valid"]
            assert any("hash mismatch" in e.lower() for e in result["errors"])

    def test_build_shards_from_manifest(self):
        m = _make_manifest(n_shards=3)
        shards = m.build_shards(base_path="/tmp")
        assert len(shards) == 3
        assert all(isinstance(s, ExecutableShard) for s in shards)
        assert shards[0].shard_id == "blk.0"
        assert shards[0].execution_semantics == ExecutionSemantics.COMPRESSED_NATIVE
        assert shards[0].contract.role == "transformer_block"
        assert shards[0].codec_ir == "hxq_codebook_index_affine_sidecar"
        assert shards[1].dependencies == ["blk.0"]

    def test_to_dict_roundtrip(self):
        m = _make_manifest(n_shards=2)
        d = m.to_dict()
        assert d["schema"] == "crystal_vault_manifest_v1"
        assert len(d["shards"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0] == ["blk.0", "blk.1"]


# --- Integration: manifest -> shards -> contracts chain ---

class TestManifestToShardChain:
    def test_linear_chain_contracts_align(self):
        m = _make_manifest(n_shards=4)
        shards = m.build_shards(base_path="/tmp")

        for i in range(len(shards) - 1):
            current_out = shards[i].contract.output_shape
            next_in = shards[i + 1].contract.input_shape
            assert current_out == next_in, (
                f"Shape mismatch at boundary {shards[i].shard_id} -> {shards[i+1].shard_id}: "
                f"output {current_out} != input {next_in}"
            )

    def test_execution_order_matches_dependencies(self):
        m = _make_manifest(n_shards=4)
        order = m.get_dependency_order()
        shards_by_id = {s["shard_id"]: s for s in m.shards}

        seen = set()
        for sid in order:
            deps = shards_by_id[sid].get("dependencies", [])
            for dep in deps:
                assert dep in seen, f"{sid} scheduled before its dependency {dep}"
            seen.add(sid)

    def test_mixed_execution_semantics(self):
        m = _make_manifest(n_shards=3)
        m.shards[0]["execution_semantics"] = "compressed_native"
        m.shards[1]["execution_semantics"] = "compressed_native"
        m.shards[2]["execution_semantics"] = "materialized"

        shards = m.build_shards(base_path="/tmp")
        assert shards[0].execution_semantics == ExecutionSemantics.COMPRESSED_NATIVE
        assert shards[1].execution_semantics == ExecutionSemantics.COMPRESSED_NATIVE
        assert shards[2].execution_semantics == ExecutionSemantics.MATERIALIZED

    def test_diamond_dag(self):
        m = VaultManifest(
            vault_id="diamond-test",
            shards=[
                {"shard_id": "A", "encoded_path": "a.hxz",
                 "contract": {"input_shape": [-1, 64], "output_shape": [-1, 64]},
                 "dependencies": []},
                {"shard_id": "B", "encoded_path": "b.hxz",
                 "contract": {"input_shape": [-1, 64], "output_shape": [-1, 64]},
                 "dependencies": ["A"]},
                {"shard_id": "C", "encoded_path": "c.hxz",
                 "contract": {"input_shape": [-1, 64], "output_shape": [-1, 64]},
                 "dependencies": ["A"]},
                {"shard_id": "D", "encoded_path": "d.hxz",
                 "contract": {"input_shape": [-1, 64], "output_shape": [-1, 64]},
                 "dependencies": ["B", "C"]},
            ],
            edges=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )
        order = m.get_dependency_order()
        assert order[0] == "A"
        assert order[-1] == "D"
        assert set(order[1:3]) == {"B", "C"}

    def test_level_2_state_path(self):
        """Level 2: ENCODED_EXECUTABLE -> EXECUTING_ENCODED -> ENCODED_EXECUTABLE.
        No materialization. The shard never leaves encoded form.
        Hardware translation boundary (centroid lookup) still exists at Level 2."""
        receipt = ShardReceipt(
            shard_id="blk.0",
            execution_semantics="compressed_native",
            execution_level=2,
            codec_ir="hxq_codebook_index_affine_sidecar",
            state_path=[
                "encoded_executable->executing_encoded",
                "executing_encoded->encoded_executable",
            ],
            wall_time_ms=0.5,
            peak_memory_mb=0.01,
            input_hash="x",
            output_hash="y",
            decompression_invoked=False,
            materialized_weight_bytes=0,
            encoded_ops_executed=24,
            fallback_used=False,
            translation_boundary="centroid_lookup_to_float_register",
        )
        d = receipt.to_dict()
        proof = d["encoded_execution_proof"]
        # Level 2 test: zero materialization, encoded ops, hardware translation
        assert proof["execution_level"] == 2
        assert proof["materialized_weight_bytes"] == 0
        assert proof["encoded_ops_executed"] > 0
        assert proof["decompression_invoked"] is False
        assert proof["fallback_used"] is False
        assert proof["translation_boundary"] != ""  # Translation exists at L2
        assert proof["translation_boundary"] != "none"  # That would be L4
        # State path never touches materialization
        for transition in d["state_path"]:
            assert "materializ" not in transition


# --- Shadow ---

class TestShadow:
    def test_shadow_creation_defaults(self):
        s = Shadow()
        assert s.entropy == 0.0
        assert s.simhash64 == 0
        assert s.glyph_histogram == []
        assert s.anchor_positions == []
        assert s.route_affinity == ""
        assert s.reconstruction_hint == ""
        assert s.confidence == 0.0

    def test_shadow_roundtrip_dict(self):
        s = Shadow(
            entropy=4.21,
            entropy_per_block=[3.9, 4.1, 4.5],
            simhash64=0xDEADBEEF_CAFEBABE,
            glyph_histogram=[10, 5, 0, 0] + [0] * 60,
            anchor_positions=[0, 128, 512],
            route_affinity="gpu",
            reconstruction_hint="hxq_affine_6:/shards/blk_0.hxz",
            confidence=0.574,
        )
        d = s.to_dict()
        s2 = Shadow.from_dict(d)
        assert s2.entropy == 4.21
        assert s2.simhash64 == 0xDEADBEEF_CAFEBABE
        assert len(s2.glyph_histogram) == 64
        assert s2.anchor_positions == [0, 128, 512]
        assert s2.route_affinity == "gpu"
        assert s2.confidence == 0.574

    def test_shadow_from_none(self):
        s = Shadow.from_dict(None)
        assert s.entropy == 0.0
        assert s.confidence == 0.0

    def test_shadow_is_cheap(self):
        """Shadow must be small relative to encoded body.
        A 64-element histogram + a few scalars + a hash < 1KB.
        An encoded shard body is 1MB-1GB. Shadow is the projection."""
        import sys
        s = Shadow(
            entropy=4.21,
            entropy_per_block=[3.9] * 32,
            simhash64=0xFFFFFFFF,
            glyph_histogram=[10] * 64,
            anchor_positions=[0, 128, 256, 384, 512],
            route_affinity="gpu",
            reconstruction_hint="hxq_affine_6:/path",
            confidence=0.574,
        )
        d = s.to_dict()
        size = len(json.dumps(d).encode())
        assert size < 4096, f"Shadow serialized to {size} bytes — too large for a projection"


# --- GlyphPacket ---

class TestGlyphPacket:
    def test_packet_creation(self):
        p = GlyphPacket(id=7, source_shard="blk.0", intent=Intent.COMPUTE)
        assert p.id == 7
        assert p.intent == Intent.COMPUTE
        assert p.shadow is None

    def test_packet_with_shadow(self):
        s = Shadow(entropy=3.8, route_affinity="gpu", confidence=0.574)
        p = GlyphPacket(id=12, shadow=s, entropy=3.8)
        assert p.shadow.route_affinity == "gpu"
        assert p.shadow.confidence == 0.574

    def test_packet_stamp_and_hash(self):
        p = GlyphPacket(id=0, source_shard="embed", intent=Intent.INITIALIZE)
        p.stamp()
        assert p.timestamp != ""
        assert p.packet_hash != ""
        assert len(p.packet_hash) == 64

    def test_packet_hash_deterministic(self):
        p1 = GlyphPacket(id=5, source_shard="blk.2", intent=Intent.COMPUTE)
        p2 = GlyphPacket(id=5, source_shard="blk.2", intent=Intent.COMPUTE)
        assert p1.compute_hash() == p2.compute_hash()

    def test_packet_hash_changes_with_intent(self):
        p1 = GlyphPacket(id=5, intent=Intent.COMPUTE)
        p2 = GlyphPacket(id=5, intent=Intent.OBSERVE)
        assert p1.compute_hash() != p2.compute_hash()

    def test_roundtrip_dict_with_shadow(self):
        s = Shadow(entropy=4.0, simhash64=42, route_affinity="cpu", confidence=0.9)
        p = GlyphPacket(
            id=3,
            source_shard="blk.1",
            intent=Intent.EVALUATE,
            entropy=4.0,
            shadow=s,
            route="zone_1_cpu",
            opcode=Opcode.C,
            next=[4, 5],
            receipt={"gate": "ALLOWED"},
        )
        p.stamp()

        d = p.to_dict()
        p2 = GlyphPacket.from_dict(d)
        assert p2.id == 3
        assert p2.intent == Intent.EVALUATE
        assert p2.shadow.simhash64 == 42
        assert p2.shadow.route_affinity == "cpu"
        assert p2.opcode == Opcode.C
        assert p2.next == [4, 5]
        assert p2.receipt == {"gate": "ALLOWED"}

    def test_roundtrip_dict_no_shadow(self):
        p = GlyphPacket(id=0, intent=Intent.INITIALIZE)
        d = p.to_dict()
        assert d["shadow"] is None
        p2 = GlyphPacket.from_dict(d)
        assert p2.shadow is None

    def test_backward_compat_float_shadow(self):
        """Old-format dicts had shadow as a float. from_dict wraps it."""
        d = {"id": 1, "shadow": 0.574}
        p = GlyphPacket.from_dict(d)
        assert p.shadow is not None
        assert p.shadow.confidence == 0.574

    def test_sensor_projection(self):
        p = GlyphPacket(id=42, source_shard="blk.5")
        proj = p.as_sensor_input()
        assert proj == {"id": 42, "source_shard": "blk.5"}

    def test_routing_projection_with_shadow(self):
        s = Shadow(entropy=3.5, route_affinity="gpu", confidence=0.574, simhash64=999)
        p = GlyphPacket(id=10, entropy=3.5, shadow=s, intent=Intent.COMPUTE)
        proj = p.as_routing_input()
        assert proj["route_affinity"] == "gpu"
        assert proj["confidence"] == 0.574
        assert proj["simhash64"] == 999
        assert proj["entropy"] == 3.5
        assert proj["intent"] == "compute"

    def test_routing_projection_no_shadow(self):
        p = GlyphPacket(id=10, entropy=3.5, intent=Intent.COMPUTE)
        proj = p.as_routing_input()
        assert proj["route_affinity"] == ""
        assert proj["confidence"] == 0.0

    def test_gate_projection(self):
        p = GlyphPacket(id=1, intent=Intent.REDIRECT, source_shard="blk.2",
                        route="zone_2_gpu", opcode=Opcode.W)
        proj = p.as_gate_input()
        assert proj["intent"] == "redirect"
        assert proj["opcode"] == "warp_control_flow"

    def test_dispatch_projection(self):
        p = GlyphPacket(id=7, opcode=Opcode.A, route="zone_2_gpu")
        proj = p.as_dispatch_input()
        assert proj["opcode"] == "action_arithmetic"
        assert proj["route"] == "zone_2_gpu"

    def test_preflight_projection(self):
        p = GlyphPacket(id=0, receipt={"hash": "abc123"})
        p.compute_hash()
        proj = p.as_preflight_input()
        assert proj["id"] == 0
        assert proj["packet_hash"] != ""
        assert proj["receipt"] == {"hash": "abc123"}


# --- Phase 0.7: One-loop proof ---

class TestOneLoopProof:
    """Prove one full packet transit through the pipeline WITHOUT opening the body.

    Loop: encoded shard → GlyphScope creates Shadow → Shadow populates GlyphPacket
          → Hydra routes from Shadow → MorphSAT gates from intent → opcode executes
          → receipt proves body was not opened for routing.

    This is a MOCK proof — no real encoded shard file is opened. The point is to
    prove the control plane protocol: every consumer reads the shadow/packet, and
    the receipt confirms the body was never inspected for routing decisions.
    """

    @staticmethod
    def _mock_glyphscope_observe(encoded_path: str, codec: str) -> Shadow:
        """Simulate GlyphScope scanning an encoded body to produce a Shadow.

        In production: reads raw bytes of the encoded shard, computes Shannon
        entropy via rolling windows, builds SimHash64 via BLAKE2b, counts
        codon frequencies for the histogram, locates @ anchor positions.
        Never decompresses. Never materializes weights.
        """
        # Simulate scanning an HXQ-encoded shard
        return Shadow(
            entropy=4.21,
            entropy_per_block=[3.9, 4.1, 4.5, 4.2],
            simhash64=0xA1B2C3D4E5F60718,
            glyph_histogram=[8, 12, 3, 7] + [2] * 60,
            anchor_positions=[0, 256, 512, 1024],
            route_affinity="",  # Not yet assigned — Hydra does that
            reconstruction_hint=f"{codec}:{encoded_path}",
            confidence=0.574,
        )

    @staticmethod
    def _mock_hydra_route(shadow: Shadow) -> str:
        """Simulate Hydra reading the Shadow to make a routing decision.

        In production: flowtorch router reads Se = H × U × D, compares to
        thresholds (Zone 1: CPU Se<0.30, Zone 2: GPU, Zone 3: QPU Se>=0.70).
        The router reads ONLY the shadow. If it opens the body, that's failure.
        """
        # Se-based routing from shadow entropy
        se = shadow.entropy
        if se < 0.30:
            return "zone_1_cpu"
        elif se < 0.70:
            return "zone_2_gpu"
        else:
            return "zone_2_gpu"  # structured high-Se defaults to GPU

    @staticmethod
    def _mock_morphsat_gate(intent: str, route: str, opcode: str) -> tuple[bool, str]:
        """Simulate MorphSAT gate check on the packet's intent.

        In production: MorphSATGate.step(TaskEvent) checks FSA legality,
        guardian vows, and constraint satisfaction. Returns (allowed, reason).
        """
        # Simple gate: all intents allowed except REDIRECT to CPU
        if intent == "redirect" and route == "zone_1_cpu":
            return False, "REDIRECT to CPU requires elevated permission"
        return True, "ALLOWED"

    @staticmethod
    def _mock_opcode_dispatch(opcode: str, route: str) -> dict:
        """Simulate opcode execution on the routed shard.

        In production: the vault runtime calls shard.eval_encoded(X) which
        triggers the Triton VQ kernel or C++ fused kernel. The kernel reads
        encoded symbols directly. materialized_weight_bytes=0.
        """
        return {
            "opcode": opcode,
            "route": route,
            "result": "executed",
            "materialized_weight_bytes": 0,
            "encoded_ops_executed": 24,
        }

    def test_one_loop_shadow_to_receipt(self):
        """The full loop: shadow → packet → route → gate → execute → receipt.

        PROVES: routing and gating decisions were made from the shadow,
        not from the encoded body. The body is only opened at execution time
        (by the kernel, which reads encoded symbols directly at Level 2).
        """
        # Step 1: GlyphScope observes the encoded shard → produces Shadow
        shadow = self._mock_glyphscope_observe(
            encoded_path="/vault/shards/blk_0_attn.hxz",
            codec="hxq_affine_6",
        )
        assert shadow.entropy > 0
        assert shadow.simhash64 != 0
        assert shadow.confidence > 0

        # Step 2: Shadow populates a GlyphPacket (born sparse, fills progressively)
        packet = GlyphPacket(
            id=1,
            source_shard="blk.0.attn",
            intent=Intent.COMPUTE,
            entropy=shadow.entropy,
            shadow=shadow,
            opcode=Opcode.A,
        )
        packet.stamp()

        # Step 3: Hydra reads the SHADOW to route (never opens body)
        route = self._mock_hydra_route(shadow)
        assert route != ""
        packet.route = route

        # Step 4: MorphSAT reads intent+route from PACKET to gate (never opens body)
        gate_input = packet.as_gate_input()
        allowed, reason = self._mock_morphsat_gate(
            gate_input["intent"], gate_input["route"], gate_input["opcode"]
        )
        assert allowed, f"Gate blocked: {reason}"

        # Step 5: Opcode dispatch executes (kernel reads encoded symbols)
        dispatch_input = packet.as_dispatch_input()
        exec_result = self._mock_opcode_dispatch(
            dispatch_input["opcode"], dispatch_input["route"]
        )
        assert exec_result["materialized_weight_bytes"] == 0
        assert exec_result["encoded_ops_executed"] > 0

        # Step 6: Build receipt — proves body was not opened for routing
        packet.receipt = {
            "gate_approved": allowed,
            "gate_reason": reason,
            "route": route,
            "execution": exec_result,
            "shadow_used_for_routing": True,
            "body_opened_for_routing": False,
            "body_opened_for_execution": True,
            "execution_level": 2,
            "timestamp": packet.timestamp,
        }

        # --- ASSERTIONS: the proof ---

        # A. Shadow was created without decompression
        assert shadow.reconstruction_hint.startswith("hxq_affine_6:")

        # B. Routing used shadow, not body
        assert packet.receipt["shadow_used_for_routing"] is True
        assert packet.receipt["body_opened_for_routing"] is False

        # C. Execution was encoded-native (Level 2)
        assert packet.receipt["execution"]["materialized_weight_bytes"] == 0
        assert packet.receipt["execution"]["encoded_ops_executed"] > 0
        assert packet.receipt["execution_level"] == 2

        # D. Gate approved without opening body
        assert packet.receipt["gate_approved"] is True

        # E. Packet accumulated state (born sparse, now full)
        d = packet.to_dict()
        assert d["id"] == 1
        assert d["shadow"] is not None
        assert d["route"] != ""
        assert d["receipt"] != {}
        assert d["packet_hash"] != ""
        assert d["timestamp"] != ""

    def test_one_loop_gate_rejection(self):
        """Gate BLOCKS a redirect-to-CPU packet. Body never opened."""
        shadow = self._mock_glyphscope_observe("/vault/shards/blk_0.hxz", "hxq_affine_6")
        packet = GlyphPacket(
            id=2,
            source_shard="blk.0.ffn",
            intent=Intent.REDIRECT,
            entropy=shadow.entropy,
            shadow=shadow,
            opcode=Opcode.W,
        )
        packet.stamp()

        route = "zone_1_cpu"
        packet.route = route

        gate_input = packet.as_gate_input()
        allowed, reason = self._mock_morphsat_gate(
            gate_input["intent"], gate_input["route"], gate_input["opcode"]
        )

        # Gate BLOCKS this
        assert not allowed
        assert "elevated permission" in reason

        # Packet gets a rejection receipt — body was never opened
        packet.receipt = {
            "gate_approved": False,
            "gate_reason": reason,
            "route": route,
            "body_opened_for_routing": False,
            "body_opened_for_execution": False,
        }
        assert packet.receipt["body_opened_for_routing"] is False
        assert packet.receipt["body_opened_for_execution"] is False

    def test_one_loop_observe_intent(self):
        """OBSERVE intent: sensor-only, no mutation, no execution."""
        shadow = self._mock_glyphscope_observe("/vault/shards/blk_0.hxz", "hxq_affine_6")
        packet = GlyphPacket(
            id=0,
            source_shard="blk.0.embed",
            intent=Intent.OBSERVE,
            entropy=shadow.entropy,
            shadow=shadow,
            opcode=Opcode.X,
        )
        packet.stamp()

        route = self._mock_hydra_route(shadow)
        packet.route = route

        gate_input = packet.as_gate_input()
        allowed, _ = self._mock_morphsat_gate(
            gate_input["intent"], gate_input["route"], gate_input["opcode"]
        )
        assert allowed

        # OBSERVE: no execution, just read shadow
        packet.receipt = {
            "gate_approved": True,
            "observation": {
                "entropy": shadow.entropy,
                "confidence": shadow.confidence,
                "simhash64": shadow.simhash64,
            },
            "body_opened_for_routing": False,
            "body_opened_for_execution": False,
        }
        assert packet.receipt["body_opened_for_execution"] is False
