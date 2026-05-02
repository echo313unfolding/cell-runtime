"""Phase 3 tests: agents wired into orchestrator tool registry.

Contract:
  - Agent tools are registered in the tool registry
  - smaLLM can call agent tools during tool loop
  - Each agent call validates input, enforces permission, emits receipt
  - Unknown agent is denied
  - Sentinel agent has orchestrator reference
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.tool_registry import create_default_registry, parse_tool_calls


def test_agent_tools_registered():
    """Agent tools are registered alongside built-in tools."""
    reg = create_default_registry()

    # Built-in tools
    assert reg.has("read_file")
    assert reg.has("grep")
    assert reg.has("shell")

    # Agent tools
    assert reg.has("rag_lookup")
    assert reg.has("graph_lookup")
    assert reg.has("graph_neighbors")
    assert reg.has("graph_stats")
    assert reg.has("ssm_get_state")
    assert reg.has("ssm_update_event")
    assert reg.has("gate_decide")
    assert reg.has("receipt_lookup")
    assert reg.has("receipt_write")
    assert reg.has("rag_search")


def test_agent_registry_attached():
    """Tool registry has agent_registry attribute."""
    reg = create_default_registry()
    assert hasattr(reg, "_agent_registry")
    assert reg._agent_registry is not None
    assert len(reg._agent_registry) >= 10


def test_rag_lookup_callable():
    """rag_lookup is callable through tool registry."""
    reg = create_default_registry()
    result = reg.execute("rag_lookup", {"query": "sentinel"})
    # Should return a dict (JSON-serialized) or string
    assert isinstance(result, str)
    # Should not be an error about unknown tool
    assert "unknown tool" not in result.lower()


def test_graph_stats_callable():
    """graph_stats is callable through tool registry."""
    reg = create_default_registry()
    result = reg.execute("graph_stats", {})
    assert isinstance(result, str)
    assert "unknown tool" not in result.lower()


def test_ssm_get_state_callable():
    """ssm_get_state is callable through tool registry."""
    reg = create_default_registry()
    result = reg.execute("ssm_get_state", {"entity_id": "IP:1.2.3.4"})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert "entity" in parsed


def test_gate_decide_callable():
    """gate_decide is callable and returns allow/deny."""
    reg = create_default_registry()
    # auto=True to skip user prompt in tests
    result = reg.execute("gate_decide", {"action": "read_file", "auto": True})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["allowed"] is True


def test_receipt_lookup_callable():
    """receipt_lookup is callable through tool registry."""
    reg = create_default_registry()
    result = reg.execute("receipt_lookup", {"receipt_id": "nonexistent"})
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert "found" in parsed


def test_unknown_agent_returns_error():
    """Unknown agent name returns error through tool registry."""
    reg = create_default_registry()
    result = reg.execute("nonexistent_agent", {})
    assert "unknown tool" in result.lower()


def test_agent_validates_input():
    """Agent validates required fields before executing."""
    reg = create_default_registry()
    # rag_lookup requires "query"
    result = reg.execute("rag_lookup", {})
    assert "error" in result.lower() or "missing" in result.lower()


def test_agent_tool_emits_receipt():
    """Agent tool calls include receipt in output."""
    reg = create_default_registry()
    result = reg.execute("ssm_get_state", {"entity_id": "test"})
    parsed = json.loads(result)
    assert "_receipt" in parsed
    assert parsed["_receipt"]["agent"] == "ssm_get_state"
    assert "wall_time_s" in parsed["_receipt"]


def test_tool_prompt_includes_agents():
    """format_for_prompt() includes agent tools."""
    reg = create_default_registry()
    prompt = reg.format_for_prompt()
    assert "rag_lookup" in prompt
    assert "graph_lookup" in prompt
    assert "ssm_get_state" in prompt
    assert "sentinel_triage" in prompt


def test_parse_tool_calls_for_agent():
    """parse_tool_calls works for agent tool calls from model output."""
    model_output = '''I'll look up this entity in the graph.

```tool_call
{"name": "graph_lookup", "arguments": {"entity": "IP:203.0.113.5"}}
```

Let me also check SSM state.

```tool_call
{"name": "ssm_get_state", "arguments": {"entity_id": "IP:203.0.113.5"}}
```
'''
    calls = parse_tool_calls(model_output)
    assert len(calls) == 2
    assert calls[0]["name"] == "graph_lookup"
    assert calls[0]["arguments"]["entity"] == "IP:203.0.113.5"
    assert calls[1]["name"] == "ssm_get_state"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
