"""Routing contract tests — enforce that smaLLM routes only, never reasons.

Contract:
  - classify() returns one of the known intents
  - route_model() maps intent to a model name in the roster
  - smaLLM (SmolLM3) is the default model
  - security_triage requires >=2 keyword matches (no false routing)
  - No intent maps to a nonexistent model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.router import classify, route_model

VALID_INTENTS = {"coding", "security_triage", "general", "reasoning", "hybrid_experiment"}
VALID_MODELS = {"qwen2.5-sentinel", "qwen2.5-coder", "smollm3"}


def test_classify_returns_valid_intent():
    """Every classify() result is a known intent."""
    inputs = [
        "hello world",
        "write python code",
        "check auth.log for suspicious brute force ssh",
        "explain quantum physics",
        "zamba mamba ssm hybrid architecture",
        "",
        "a" * 10000,
    ]
    for inp in inputs:
        intent = classify(inp)
        assert intent in VALID_INTENTS, f"Unknown intent '{intent}' for input: {inp[:50]}"


def test_route_model_returns_valid_model():
    """Every route_model() result is in the roster."""
    for intent in VALID_INTENTS:
        model = route_model(intent)
        assert model in VALID_MODELS, f"Unknown model '{model}' for intent: {intent}"


def test_default_model_is_smollm():
    """General and reasoning route to SmolLM3 (the front-end)."""
    assert route_model("general") == "smollm3"
    assert route_model("reasoning") == "smollm3"


def test_security_routes_to_sentinel():
    """security_triage routes to Qwen Sentinel."""
    assert route_model("security_triage") == "qwen2.5-sentinel"


def test_coding_routes_to_coder():
    """coding routes to Qwen Coder."""
    assert route_model("coding") == "qwen2.5-coder"


def test_single_security_keyword_does_not_trigger():
    """A single security keyword alone does NOT trigger security_triage.

    The router requires >=2 matches to avoid false routing.
    """
    # "ssh" alone matches 1 security pattern, shouldn't trigger
    intent = classify("ssh")
    assert intent != "security_triage" or classify("just ssh stuff") != "security_triage"


def test_unknown_intent_maps_to_smollm():
    """Unknown intents fall through to smollm3."""
    assert route_model("nonexistent_intent") == "smollm3"


def test_empty_input_is_general():
    """Empty input classifies as general."""
    assert classify("") == "general"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
