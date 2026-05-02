"""Tests for cartridge pool — manifest loading, routing, dispatch.

Contract:
  - CartridgePool loads manifests from cartridge directories
  - Routes intents to the correct cartridge
  - Loads artifacts (system_prompt, examples, grammar, policy)
  - Builds augmented prompts from cartridge artifacts
  - Dispatches through orchestrator when available
  - Disabled/candidate cartridges are NOT activated
  - Unloads after dispatch
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.cartridge_pool import CartridgePool, CartridgeManifest


def _make_cartridge(tmpdir, cartridge_id, intents, status="active",
                    system_prompt="Test prompt", examples=None):
    """Create a minimal cartridge in a temp directory."""
    cart_dir = os.path.join(tmpdir, cartridge_id)
    os.makedirs(cart_dir, exist_ok=True)

    manifest = {
        "cartridge_id": cartridge_id,
        "type": "skill_cartridge",
        "activation_intents": intents,
        "base_model": "test_model",
        "artifact_type": "prompt_pack",
        "status": status,
        "scope": "test",
        "system_prompt": system_prompt,
        "fallback": "test_model",
    }

    if examples:
        manifest["examples_path"] = "examples.jsonl"
        with open(os.path.join(cart_dir, "examples.jsonl"), "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    with open(os.path.join(cart_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    return cart_dir


def test_load_manifests():
    """CartridgePool loads manifests from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "cart_a", ["intent_a"])
        _make_cartridge(tmpdir, "cart_b", ["intent_b"])

        pool = CartridgePool(tmpdir)
        assert len(pool) == 2
        assert pool.get("cart_a") is not None
        assert pool.get("cart_b") is not None


def test_route_by_intent():
    """Route matches intent to correct cartridge."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "repair", ["parser_repair", "tool_fix"])
        _make_cartridge(tmpdir, "rules", ["yara_rule", "sigma_rule"])

        pool = CartridgePool(tmpdir)
        cart = pool.route("parser_repair")
        assert cart is not None
        assert cart.cartridge_id == "repair"

        cart2 = pool.route("sigma_rule")
        assert cart2 is not None
        assert cart2.cartridge_id == "rules"


def test_route_unknown_intent():
    """Unknown intent returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "repair", ["parser_repair"])
        pool = CartridgePool(tmpdir)
        assert pool.route("nonexistent") is None


def test_disabled_cartridge_not_routed():
    """Disabled cartridge is not returned by route."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "disabled_cart", ["some_intent"], status="disabled")
        pool = CartridgePool(tmpdir)
        assert pool.route("some_intent") is None


def test_candidate_cartridge_not_routed():
    """Candidate cartridge is not returned by route."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "candidate_cart", ["test_intent"], status="candidate")
        pool = CartridgePool(tmpdir)
        assert pool.route("test_intent") is None


def test_load_cartridge_artifacts():
    """Loading a cartridge returns system prompt and examples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        examples = [
            {"input": "error X", "output": "fix Y"},
            {"input": "error A", "output": "fix B"},
        ]
        _make_cartridge(tmpdir, "repair", ["code_repair"],
                       system_prompt="You are a repair specialist.",
                       examples=examples)
        pool = CartridgePool(tmpdir)

        result = pool.load_cartridge("repair")
        assert "error" not in result
        assert result["cartridge_id"] == "repair"
        assert "system_prompt" in result["loaded"]
        assert "examples" in result["loaded"]


def test_load_unknown_cartridge():
    """Loading unknown cartridge returns error."""
    pool = CartridgePool()
    result = pool.load_cartridge("nonexistent")
    assert "error" in result


def test_build_prompt():
    """build_prompt assembles system_prompt + examples + task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        examples = [{"input": "test error", "output": "test fix"}]
        _make_cartridge(tmpdir, "repair", ["code_repair"],
                       system_prompt="You are a repair specialist.",
                       examples=examples)
        pool = CartridgePool(tmpdir)
        pool.load_cartridge("repair")

        prompt = pool.build_prompt("repair", "Fix this bug please")
        assert "You are a repair specialist." in prompt
        assert "test error" in prompt
        assert "Fix this bug please" in prompt


def test_dispatch_without_orchestrator():
    """Dispatch without orchestrator returns assembled prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "repair", ["code_repair"],
                       system_prompt="Repair specialist.")
        pool = CartridgePool(tmpdir)

        result = pool.dispatch("code_repair", "Fix the bug")
        assert "error" not in result
        assert result["cartridge_id"] == "repair"
        assert result["assembled_prompt"] is not None
        assert "Repair specialist." in result["assembled_prompt"]
        assert "Fix the bug" in result["assembled_prompt"]


def test_dispatch_unknown_intent():
    """Dispatch with unknown intent returns error."""
    pool = CartridgePool()
    result = pool.dispatch("nonexistent", "task")
    assert "error" in result


def test_unload_after_dispatch():
    """Cartridge is unloaded after dispatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "repair", ["code_repair"])
        pool = CartridgePool(tmpdir)

        pool.dispatch("code_repair", "task")
        assert pool.loaded is None


def test_activation_log():
    """Dispatch records activation in log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "repair", ["code_repair"])
        pool = CartridgePool(tmpdir)

        pool.dispatch("code_repair", "task")
        log = pool.get_activation_log()
        assert len(log) == 1
        assert log[0]["cartridge_id"] == "repair"
        assert log[0]["intent"] == "code_repair"
        assert "timestamp" in log[0]


def test_list_cartridges():
    """list_cartridges returns metadata for all registered cartridges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "cart_a", ["intent_a"])
        _make_cartridge(tmpdir, "cart_b", ["intent_b"], status="candidate")
        pool = CartridgePool(tmpdir)

        listing = pool.list_cartridges()
        assert len(listing) == 2
        ids = {c["cartridge_id"] for c in listing}
        assert "cart_a" in ids
        assert "cart_b" in ids
        # candidate shows status
        for c in listing:
            if c["cartridge_id"] == "cart_b":
                assert c["status"] == "candidate"


def test_intent_map():
    """intent_map returns complete intent → cartridge_id mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_cartridge(tmpdir, "repair", ["parser_repair", "tool_fix"])
        pool = CartridgePool(tmpdir)

        imap = pool.intent_map()
        assert imap["parser_repair"] == "repair"
        assert imap["tool_fix"] == "repair"


def test_manifest_from_file():
    """CartridgeManifest.from_file parses manifest.json correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cart_dir = os.path.join(tmpdir, "test_cart")
        os.makedirs(cart_dir)
        manifest = {
            "cartridge_id": "test_v1",
            "activation_intents": ["test_a", "test_b"],
            "base_model": "sentinel",
            "artifact_type": "lora_adapter",
            "artifact_path": "adapter.gguf-lora",
            "status": "active",
            "scope": "testing",
            "requires_ask_pass": True,
        }
        with open(os.path.join(cart_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        m = CartridgeManifest.from_file(os.path.join(cart_dir, "manifest.json"))
        assert m.cartridge_id == "test_v1"
        assert m.activation_intents == ["test_a", "test_b"]
        assert m.artifact_type == "lora_adapter"
        assert m.requires_ask_pass is True
        assert m.cartridge_dir == cart_dir


def test_register_cartridge_directly():
    """register() adds a cartridge without scanning directory."""
    pool = CartridgePool()
    m = CartridgeManifest(
        cartridge_id="direct_cart",
        cartridge_dir="/tmp",
        activation_intents=["direct_intent"],
        status="active",
    )
    pool.register(m)
    assert pool.get("direct_cart") is not None
    assert pool.route("direct_intent") is not None


def test_real_cartridges_load():
    """Real cartridge manifests in cell-runtime/cartridges/ load correctly."""
    cartridge_dir = str(Path(__file__).parent.parent / "cartridges")
    if not os.path.isdir(cartridge_dir):
        return  # skip if not present

    pool = CartridgePool(cartridge_dir)
    assert len(pool) >= 5

    # Check known cartridges
    assert pool.get("code_parser_repair_v1") is not None
    assert pool.get("rule_generation_v1") is not None
    assert pool.get("patch_review_v1") is not None
    assert pool.get("exploit_analysis_v1") is not None
    assert pool.get("repo_context_v1") is not None

    # Check routing
    assert pool.route("code_repair") is not None
    assert pool.route("yara_rule") is not None
    assert pool.route("patch_review") is not None
    assert pool.route("exploit_analysis") is not None
    assert pool.route("repo_context") is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
