"""Behavior smoke tests for the default chat model.

These tests verify that the default model produces sane output
on basic prompts. They do NOT require a running backend — they
test the router and system prompt configuration, not generation.

For live generation tests, run: tools/smoke_behavior_live.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.router import classify, route_model


def test_greeting_routes_to_default():
    """A greeting should route to the default model, not get misrouted."""
    intent = classify("hey echo")
    model = route_model(intent)
    assert intent == "general"
    assert model == "qwen2.5-coder"


def test_identity_routes_to_default():
    """'Who are you?' should stay on default, not route to coding."""
    intent = classify("who are you?")
    model = route_model(intent)
    assert model == "qwen2.5-coder"


def test_simple_math_routes_to_default():
    """Simple math stays on default."""
    intent = classify("what is 2 + 2?")
    model = route_model(intent)
    assert model == "qwen2.5-coder"


def test_code_request_routes_to_coder():
    """Code requests route to coder."""
    intent = classify("write a python function that reverses a string")
    model = route_model(intent)
    assert intent == "coding"
    assert model == "qwen2.5-coder"


def test_security_routes_to_sentinel():
    """Security triage routes to sentinel."""
    intent = classify("investigate the brute force SSH attack in auth.log")
    model = route_model(intent)
    assert intent == "security_triage"
    assert model == "qwen2.5-sentinel"


def test_smollm3_not_default():
    """SmolLM3 is demoted — it should not be the default for any intent."""
    for intent in ["general", "coding", "reasoning", "hybrid_experiment"]:
        model = route_model(intent)
        assert model != "smollm3", f"SmolLM3 is still default for {intent}"
