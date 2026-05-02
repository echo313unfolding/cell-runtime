"""ask-pass — privileged action gate for the cell daemon.

Like Claude Code: when the local model wants to do something privileged,
prompt the user and wait for approval. No complex allowlists, no gatekeeper
model. Just ask.

Usage:
    from cell.ask_pass import ask_pass

    if ask_pass("shell", '{"command": "ss -tlnp"}'):
        result = registry.execute("shell", args)
    else:
        result = "Action denied by user."
"""
import json
import os
import sys
import time
from pathlib import Path

# Actions that require user approval
PRIVILEGED = {
    "shell",
    "delegate_to_host",
    "file_write",
}

# Actions that auto-approve (no prompt)
AUTO_APPROVE = {
    "read_file",
    "grep",
    "list_dir",
    "web_search",
    "memory_search",
}

# Session-level "allow all" flag (set by user saying "yes to all")
_allow_all = False

# Audit log
_audit_log: list[dict] = []


def ask_pass(action: str, detail: str = "", auto: bool = False) -> bool:
    """Prompt user for approval of a privileged action.

    Returns True if approved, False if denied.
    """
    global _allow_all

    # Not privileged → auto-approve
    if action in AUTO_APPROVE or action not in PRIVILEGED:
        _log(action, detail, "auto_approve")
        return True

    # Environment override (for testing or headless mode)
    if os.environ.get("CELL_ASK_PASS") == "0":
        _log(action, detail, "env_bypass")
        return True

    # Session-level "allow all"
    if _allow_all or auto:
        _log(action, detail, "session_allow_all")
        return True

    # Prompt
    print(f"\n--- ask-pass ---", file=sys.stderr)
    print(f"  Action: {action}", file=sys.stderr)
    if detail:
        # Truncate long details
        display = detail[:300] + "..." if len(detail) > 300 else detail
        print(f"  Detail: {display}", file=sys.stderr)

    try:
        response = input("  Allow? [y/N/a(ll)] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  Denied (no input).", file=sys.stderr)
        _log(action, detail, "denied_no_input")
        return False

    if response in ("a", "all"):
        _allow_all = True
        _log(action, detail, "approved_all")
        return True
    elif response in ("y", "yes"):
        _log(action, detail, "approved")
        return True
    else:
        print("  Denied.", file=sys.stderr)
        _log(action, detail, "denied")
        return False


def reset_session():
    """Reset session-level allow-all flag."""
    global _allow_all
    _allow_all = False


def get_audit_log() -> list[dict]:
    """Return the session audit log."""
    return list(_audit_log)


def _log(action: str, detail: str, decision: str):
    """Record an ask-pass decision."""
    _audit_log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "action": action,
        "detail": detail[:200],
        "decision": decision,
    })
