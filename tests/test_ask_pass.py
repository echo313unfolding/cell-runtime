"""Tests for the ask-pass privileged action gate."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.ask_pass import ask_pass, reset_session, get_audit_log, _audit_log


def setup_function():
    """Reset state before each test."""
    reset_session()
    _audit_log.clear()


def test_auto_approve_read_file():
    """Read-only tools auto-approve without prompting."""
    assert ask_pass("read_file", "/some/path") is True
    assert ask_pass("grep", "pattern") is True
    assert ask_pass("list_dir", "/home") is True
    assert ask_pass("web_search", "query") is True
    assert ask_pass("memory_search", "sentinel") is True
    log = get_audit_log()
    assert len(log) == 5
    assert all(e["decision"] == "auto_approve" for e in log)


def test_unknown_action_auto_approves():
    """Actions not in PRIVILEGED auto-approve."""
    assert ask_pass("some_new_tool", "args") is True


def test_shell_requires_approval():
    """Shell tool requires user approval."""
    with patch("builtins.input", return_value="y"):
        assert ask_pass("shell", '{"command": "ls"}') is True
    log = get_audit_log()
    assert log[-1]["decision"] == "approved"


def test_shell_denied():
    """Shell tool denied by user."""
    with patch("builtins.input", return_value="n"):
        assert ask_pass("shell", '{"command": "ls"}') is False
    log = get_audit_log()
    assert log[-1]["decision"] == "denied"


def test_delegate_requires_approval():
    """delegate_to_host requires user approval."""
    with patch("builtins.input", return_value="y"):
        assert ask_pass("delegate_to_host", '{"goal": "help"}') is True


def test_allow_all():
    """Typing 'a' approves all subsequent privileged actions."""
    with patch("builtins.input", return_value="a"):
        assert ask_pass("shell", "first") is True

    # Now auto-approves without prompting
    assert ask_pass("shell", "second") is True
    assert ask_pass("delegate_to_host", "third") is True

    log = get_audit_log()
    assert log[-2]["decision"] == "session_allow_all"
    assert log[-1]["decision"] == "session_allow_all"


def test_env_bypass():
    """CELL_ASK_PASS=0 bypasses all prompts."""
    os.environ["CELL_ASK_PASS"] = "0"
    try:
        assert ask_pass("shell", "command") is True
        log = get_audit_log()
        assert log[-1]["decision"] == "env_bypass"
    finally:
        del os.environ["CELL_ASK_PASS"]


def test_eof_denies():
    """EOFError (no stdin) denies the action."""
    with patch("builtins.input", side_effect=EOFError):
        assert ask_pass("shell", "command") is False
    log = get_audit_log()
    assert log[-1]["decision"] == "denied_no_input"


def test_auto_flag():
    """auto=True bypasses prompting for privileged actions."""
    assert ask_pass("shell", "command", auto=True) is True


def test_audit_log_records_detail():
    """Audit log includes truncated detail."""
    long_detail = "x" * 500
    assert ask_pass("read_file", long_detail) is True
    log = get_audit_log()
    assert len(log[-1]["detail"]) == 200


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
