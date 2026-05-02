"""Phase 4 integration test — end-to-end specialist compute path.

Proves the full chain WITHOUT live model inference:
  1. Sentinel triage receives an alert → assembles context pack
  2. specialist_compute_route routes to correct cartridge
  3. Cartridge dispatches and builds augmented prompt (proposal-only)
  4. gate_decide checks permission (auto=True for testing)
  5. Receipt emitted with cartridge_id, artifacts_loaded, wall_time

No live models. No network. No GPU. All assertions are structural:
the right agents fire, the right cartridges load, the right receipts
come back, and the gatekeeper controls execution.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentRegistry, AgentResult, Permission
from cell.agents.sentinel_agent import SentinelTriageAgent
from cell.agents.specialist_compute_agent import (
    SpecialistComputeRouteAgent, ShardListAgent, ShardResourceCheckAgent,
    reset_shard_pool, get_shard_pool,
)
from cell.agents.cartridge_agent import (
    CartridgeDispatchAgent, CodeRepairAgent, CartridgeListAgent,
    get_cartridge_pool, reset_pool,
)
from cell.agents.policy_agent import GateDecideAgent
from cell.cartridge_pool import CartridgePool, CartridgeManifest
from cell.shard_pool import ShardPool, ShardManifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_test_cartridges(tmpdir):
    """Create minimal cartridge manifests for integration testing."""
    # code_parser_repair cartridge
    cart_dir = os.path.join(tmpdir, "code_parser_repair")
    os.makedirs(cart_dir)
    manifest = {
        "cartridge_id": "code_parser_repair_v1",
        "activation_intents": ["parser_repair", "tool_fix", "json_repair", "code_repair"],
        "base_model": "qwen2.5-sentinel",
        "artifact_type": "prompt_pack",
        "scope": "code_repair",
        "status": "active",
        "system_prompt": "You are a code repair specialist. Diagnose and propose fixes.",
        "examples_path": "examples.jsonl",
        "eval_receipt": "eval_receipt.json",
    }
    with open(os.path.join(cart_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(cart_dir, "examples.jsonl"), "w") as f:
        f.write('{"input": "SyntaxError: unexpected EOF", "output": "Missing closing bracket"}\n')
    with open(os.path.join(cart_dir, "eval_receipt.json"), "w") as f:
        json.dump({"pass": True, "task_count": 20, "accuracy": 0.85}, f)

    # rule_generation cartridge
    rule_dir = os.path.join(tmpdir, "rule_generation")
    os.makedirs(rule_dir)
    rule_manifest = {
        "cartridge_id": "rule_generation_v1",
        "activation_intents": ["yara_rule", "sigma_rule", "rule_generate", "detection_rule"],
        "base_model": "qwen2.5-sentinel",
        "artifact_type": "prompt_pack",
        "scope": "detection",
        "status": "active",
        "system_prompt": "You generate detection rules.",
    }
    with open(os.path.join(rule_dir, "manifest.json"), "w") as f:
        json.dump(rule_manifest, f)

    return tmpdir


def _make_test_shards(tmpdir):
    """Create shard manifests for integration testing."""
    shard_dir = os.path.join(tmpdir, "coder_7b_q5")
    os.makedirs(shard_dir)
    gguf = os.path.join(shard_dir, "coder_7b.gguf")
    with open(gguf, "w") as f:
        f.write("FAKE_GGUF")
    manifest = {
        "model_id": "coder_7b_q5",
        "codec": "q5_k_m",
        "shard_paths": [gguf],
        "activation_intents": ["deep_code_analysis", "multi_file_refactor"],
        "status": "active",
        "offload_policy": {"gpu_layers": 20, "mmap": True},
        "required_vram_mb": 2500,
        "required_ram_mb": 8000,
        "fallback": "qwen2.5-sentinel",
        "behavioral_eval_receipt": "eval.json",
    }
    with open(os.path.join(shard_dir, "shard_manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(shard_dir, "eval.json"), "w") as f:
        json.dump({"pass": True}, f)
    return tmpdir


def _build_registry(cartridge_dir, shard_dir):
    """Build a wired agent registry for integration testing."""
    # Reset shared pools
    reset_pool()
    reset_shard_pool()

    # Create pools with test dirs
    cart_pool = CartridgePool(cartridge_dir)
    shard_pool = ShardPool(shard_dir)

    # Wire the shared pools via module-level singletons
    import cell.agents.cartridge_agent as ca_mod
    import cell.agents.specialist_compute_agent as sc_mod
    ca_mod._pool = cart_pool
    sc_mod._shard_pool = shard_pool

    # Build registry
    registry = AgentRegistry()
    agents = [
        SentinelTriageAgent(),
        SpecialistComputeRouteAgent(),
        ShardListAgent(),
        ShardResourceCheckAgent(),
        CartridgeDispatchAgent(),
        CodeRepairAgent(),
        CartridgeListAgent(),
        GateDecideAgent(),
    ]
    for a in agents:
        registry.register(a)

    # Wire cross-references (mimics orchestrator.__init__)
    sentinel = registry.get("sentinel_triage")
    sentinel._agent_registry = registry

    for name in ["cartridge_dispatch", "code_repair", "specialist_compute_route"]:
        agent = registry.get(name)
        if agent:
            agent._orchestrator = None  # No live orchestrator — tests proposal path

    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sentinel_assembles_context_pack():
    """Sentinel triage builds a context pack even without live backends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        _make_test_shards(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("sentinel_triage", {
            "alert_text": "Suspicious process svchost.exe connecting to 185.220.101.1",
            "entity_id": "host-01",
        })

        assert result.ok
        assert "context_pack" in result.output
        pack = result.output["context_pack"]
        assert pack["alert"]["text"].startswith("Suspicious process")
        # Without live backends, SSM/RAG/graph return empty but don't crash
        assert "ssm_state" in pack
        assert "rag_context" in pack
        assert "graph_links" in pack


def test_sentinel_no_orchestrator_returns_unknown_verdict():
    """Without orchestrator, sentinel returns context pack with unknown verdict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("sentinel_triage", {
            "alert_text": "Failed login attempt root@10.0.0.1",
        })

        assert result.ok
        assert result.output["verdict"]["severity"] == "unknown"
        assert result.output["gate_fired"] is False


def test_specialist_route_cartridge_first():
    """specialist_compute_route returns cartridge when intent matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("specialist_compute_route", {
            "intent": "code_repair",
            "task": "Fix the JSON parser crash in auth handler",
        })

        assert result.ok
        assert result.output["route"] == "cartridge"
        assert result.output["cartridge_id"] == "code_parser_repair_v1"
        assert "code_repair" in result.output["reason"]


def test_specialist_route_shard_when_no_cartridge():
    """When no cartridge matches, specialist_compute_route falls to shard pool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        _make_test_shards(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("specialist_compute_route", {
            "intent": "deep_code_analysis",
            "task": "Analyze the full codebase for architectural debt",
        })

        assert result.ok
        assert result.output["route"] == "shard"
        assert result.output["model_id"] == "coder_7b_q5"


def test_specialist_route_fallback_sentinel():
    """When neither cartridge nor shard matches, fallback to Sentinel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("specialist_compute_route", {
            "intent": "unknown_intent_xyz",
            "task": "Something that no cartridge or shard handles",
        })

        assert result.ok
        assert result.output["route"] == "fallback"
        assert result.output["model_id"] == "qwen2.5-sentinel"


def test_cartridge_dispatch_builds_prompt():
    """CartridgeDispatchAgent loads cartridge and builds augmented prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("cartridge_dispatch", {
            "intent": "code_repair",
            "task": "Fix SyntaxError in line 42 of auth.py",
        })

        assert result.ok
        # Without orchestrator, returns assembled_prompt (no model output)
        assert "assembled_prompt" in result.output or "output" in result.output
        assert result.output["cartridge_id"] == "code_parser_repair_v1"
        assert "system_prompt" in result.output.get("artifacts_loaded", [])
        assert "examples" in result.output.get("artifacts_loaded", [])


def test_code_repair_agent_proposal_only():
    """CodeRepairAgent dispatches to cartridge and returns proposal, not execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("code_repair", {
            "error_log": "Traceback: JSONDecodeError at line 5",
            "code": "data = json.loads(raw)",
            "file_path": "src/handler.py",
        })

        assert result.ok
        # Proposal returned, not executed
        assert "repair_plan" in result.output or "assembled_prompt" in result.output.get("repair_plan", "")
        assert result.output["cartridge_id"] == "code_parser_repair_v1"


def test_gate_decide_auto_approve():
    """GateDecideAgent auto-approves with auto=True (testing mode)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("gate_decide", {
            "action": "shell",
            "detail": "ss -tlnp",
            "auto": True,
        })

        assert result.ok
        assert result.output["allowed"] is True
        assert result.output["action"] == "shell"


def test_gate_decide_non_privileged_auto_approves():
    """Non-privileged actions auto-approve without explicit auto flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("gate_decide", {
            "action": "read_file",
            "detail": "/etc/hosts",
        })

        assert result.ok
        assert result.output["allowed"] is True


def test_full_chain_cartridge_path():
    """Full chain: route → cartridge dispatch → gate → receipt.

    Proves:
    1. specialist_compute_route selects cartridge for code_repair intent
    2. cartridge_dispatch loads correct cartridge and builds prompt
    3. gate_decide approves the proposal
    4. All steps emit receipts with agent name and wall_time_s
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        # Step 1: Route
        route_result = registry.run("specialist_compute_route", {
            "intent": "code_repair",
            "task": "Fix parser crash in JSON handler",
        })
        assert route_result.ok
        assert route_result.output["route"] == "cartridge"
        assert route_result.receipt is not None
        assert route_result.receipt["agent"] == "specialist_compute_route"
        assert route_result.receipt["wall_time_s"] >= 0

        # Step 2: Dispatch to cartridge
        dispatch_result = registry.run("cartridge_dispatch", {
            "intent": "code_repair",
            "task": "Fix parser crash in JSON handler",
        })
        assert dispatch_result.ok
        assert dispatch_result.output["cartridge_id"] == "code_parser_repair_v1"
        assert dispatch_result.receipt is not None
        assert dispatch_result.receipt["agent"] == "cartridge_dispatch"

        # Step 3: Gate check (proposal is approved)
        gate_result = registry.run("gate_decide", {
            "action": "code_repair_proposal",
            "detail": "Propose fix for JSON handler crash",
            "auto": True,
        })
        assert gate_result.ok
        assert gate_result.output["allowed"] is True
        assert gate_result.receipt is not None
        assert gate_result.receipt["agent"] == "gate_decide"

        # Step 4: Verify all receipts have required fields
        for result in [route_result, dispatch_result, gate_result]:
            receipt = result.receipt
            assert "agent" in receipt
            assert "permission" in receipt
            assert "wall_time_s" in receipt
            assert "timestamp" in receipt
            assert receipt["error"] is None


def test_full_chain_shard_path():
    """Full chain: route → shard selected → gate → receipt.

    When intent requires deep analysis beyond cartridge capability,
    the router selects a shard and the gate controls execution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        _make_test_shards(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        # Step 1: Route — deep_code_analysis has no cartridge, falls to shard
        route_result = registry.run("specialist_compute_route", {
            "intent": "deep_code_analysis",
            "task": "Multi-file architectural review",
        })
        assert route_result.ok
        assert route_result.output["route"] == "shard"
        assert route_result.output["model_id"] == "coder_7b_q5"
        assert "plan" in route_result.output  # load plan present

        # Step 2: Gate check
        gate_result = registry.run("gate_decide", {
            "action": "shard_load",
            "detail": f"Load shard {route_result.output['model_id']} for deep_code_analysis",
            "auto": True,
        })
        assert gate_result.ok
        assert gate_result.output["allowed"] is True

        # Step 3: Verify receipts
        assert route_result.receipt["agent"] == "specialist_compute_route"
        assert gate_result.receipt["agent"] == "gate_decide"


def test_full_chain_fallback_path():
    """Full chain: unknown intent → fallback to Sentinel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        route_result = registry.run("specialist_compute_route", {
            "intent": "quantum_physics_analysis",
            "task": "Analyze quantum decoherence in sensor data",
        })
        assert route_result.ok
        assert route_result.output["route"] == "fallback"
        assert route_result.output["model_id"] == "qwen2.5-sentinel"


def test_cartridge_list_reports_status():
    """cartridge_list agent reports all cartridges with correct metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("cartridge_list", {})
        assert result.ok
        assert result.output["count"] == 2  # code_parser_repair + rule_generation
        carts = result.output["cartridges"]
        ids = {c["cartridge_id"] for c in carts}
        assert "code_parser_repair_v1" in ids
        assert "rule_generation_v1" in ids
        # Intent map covers all registered intents
        imap = result.output["intent_map"]
        assert imap["code_repair"] == "code_parser_repair_v1"
        assert imap["yara_rule"] == "rule_generation_v1"


def test_shard_list_reports_status():
    """shard_list agent reports shards with codec and resource info."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        _make_test_shards(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("shard_list", {})
        assert result.ok
        assert result.output["count"] >= 1
        shards = result.output["shards"]
        assert any(s["model_id"] == "coder_7b_q5" for s in shards)


def test_receipts_contain_permission_level():
    """All agent receipts include permission level for audit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        result = registry.run("specialist_compute_route", {
            "intent": "code_repair",
            "task": "test",
        })
        assert result.receipt["permission"] == "read"

        gate = registry.run("gate_decide", {
            "action": "shell",
            "auto": True,
        })
        assert gate.receipt["permission"] == "read"


def test_cartridge_unloads_after_dispatch():
    """Cartridge pool unloads artifacts after dispatch completes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        import cell.agents.cartridge_agent as ca_mod
        pool = ca_mod._pool

        # Before dispatch
        assert pool.loaded is None

        # Dispatch
        registry.run("cartridge_dispatch", {
            "intent": "code_repair",
            "task": "test unload",
        })

        # After dispatch — cartridge should be unloaded
        assert pool.loaded is None


def test_activation_log_records_dispatch():
    """Cartridge pool activation log records every dispatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = _make_test_cartridges(tmpdir)
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        registry = _build_registry(cart_dir, shard_dir)

        import cell.agents.cartridge_agent as ca_mod
        pool = ca_mod._pool

        registry.run("cartridge_dispatch", {
            "intent": "code_repair",
            "task": "test log 1",
        })
        registry.run("cartridge_dispatch", {
            "intent": "yara_rule",
            "task": "test log 2",
        })

        log = pool.get_activation_log()
        assert len(log) == 2
        assert log[0]["cartridge_id"] == "code_parser_repair_v1"
        assert log[0]["intent"] == "code_repair"
        assert log[1]["cartridge_id"] == "rule_generation_v1"
        assert log[1]["intent"] == "yara_rule"
        assert all("timestamp" in e for e in log)
        assert all("wall_time_s" in e for e in log)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
