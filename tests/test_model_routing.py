"""Tests for model routing with smaLLM front-end + Sentinel backend."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.router import classify, route_model


def test_general_routes_to_smollm():
    """General queries route to smaLLM (SmolLM3)."""
    # "weather" doesn't match any pattern set → general
    intent = classify("tell me about the solar system")
    model = route_model(intent)
    assert intent == "general"
    assert model == "smollm3"


def test_security_routes_to_sentinel():
    """Security alerts route to Qwen Sentinel (needs >=2 security keywords)."""
    # auth.log + failed login = 2 security matches → security_triage
    intent = classify("suspicious failed login attempt in auth.log")
    model = route_model(intent)
    assert intent == "security_triage"
    assert model == "qwen2.5-sentinel"


def test_coding_routes_to_coder():
    """Coding queries route to Qwen Coder."""
    intent = classify("write a python sort function")
    model = route_model(intent)
    assert intent == "coding"
    assert model in ("qwen2.5-coder", "qwen2.5-coder:latest")


def test_reasoning_routes_to_smollm():
    """Reasoning queries route to SmolLM3."""
    intent = classify("explain the theory of relativity step by step")
    model = route_model(intent)
    assert intent in ("reasoning", "general")
    assert model == "smollm3"


def test_security_keywords():
    """Security inputs with >=2 keyword matches trigger security_triage.

    The router requires >=2 security pattern matches to avoid false routing
    on single keyword mentions.
    """
    security_inputs = [
        "suspicious alert in syslog",             # suspicious + syslog
        "reverse shell backdoor on server",        # reverse shell + backdoor
        "malware alert detected by sentinel",        # malware + alert + sentinel
        "investigate IOC indicators in alert log", # investigate + ioc + alert
        "brute force ssh attack detected",         # brute force + ssh
    ]
    for inp in security_inputs:
        intent = classify(inp)
        assert intent == "security_triage", f"Expected security_triage for: {inp}, got: {intent}"


def test_coding_keywords():
    """Various coding keywords trigger coding intent."""
    coding_inputs = [
        "write a function that checks if a number is prime",
        "debug this python code",
        "refactor the database query",
        "implement binary search in rust",
    ]
    for inp in coding_inputs:
        intent = classify(inp)
        assert intent == "coding", f"Expected coding for: {inp}, got: {intent}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
