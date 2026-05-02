"""Receipt enforcement tests — every action must emit a receipt.

Contract:
  - Every AgentBase.run() attaches a receipt to the result
  - Receipt has: agent name, permission, wall_time_s, timestamp
  - Receipt records errors
  - ReceiptWriteAgent can write receipt files
  - ReceiptLookupAgent can find receipt files
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cell.agents.base import AgentBase, AgentResult, Permission
from cell.agents.receipt_agent import ReceiptWriteAgent, ReceiptLookupAgent


class _DummyAgent(AgentBase):
    name = "dummy"
    permission = Permission.READ
    timeout_s = 1
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> AgentResult:
        return AgentResult(output={"value": 42})


class _FailingAgent(AgentBase):
    name = "failing"
    permission = Permission.READ
    timeout_s = 1

    def execute(self, args: dict) -> AgentResult:
        raise ValueError("intentional failure")


def test_receipt_attached_on_success():
    """Successful agent run attaches receipt."""
    agent = _DummyAgent()
    result = agent.run({})
    assert result.ok
    assert result.receipt is not None
    assert result.receipt["agent"] == "dummy"
    assert result.receipt["permission"] == "read"
    assert result.receipt["wall_time_s"] >= 0
    assert "timestamp" in result.receipt
    assert result.receipt["error"] is None


def test_receipt_attached_on_failure():
    """Failed agent run also attaches receipt with error info."""
    agent = _FailingAgent()
    result = agent.run({})
    assert not result.ok
    assert result.receipt is not None
    assert result.receipt["agent"] == "failing"
    assert result.receipt["error"] is not None
    assert "intentional failure" in result.receipt["error"]


def test_receipt_has_timing():
    """Receipt wall_time_s is a positive number."""
    agent = _DummyAgent()
    result = agent.run({})
    assert isinstance(result.receipt["wall_time_s"], float)
    assert result.receipt["wall_time_s"] >= 0


def test_receipt_write_agent():
    """ReceiptWriteAgent writes a receipt file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkey-patch receipt dir
        import cell.agents.receipt_agent as mod
        old_dir = mod.RECEIPT_DIR
        mod.RECEIPT_DIR = tmpdir

        try:
            agent = ReceiptWriteAgent()
            result = agent.run({
                "action": "test_action",
                "result": {"status": "ok", "value": 123},
                "subdirectory": "test",
            })
            assert result.ok
            assert result.output["written"]

            # Verify the file exists and is valid JSON
            path = result.output["path"]
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["action"] == "test_action"
            assert data["result"]["value"] == 123
            assert "receipt_id" in data
            assert "timestamp" in data
        finally:
            mod.RECEIPT_DIR = old_dir


def test_receipt_lookup_agent():
    """ReceiptLookupAgent finds receipt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a test receipt
        receipt = {"receipt_id": "test_123", "action": "test"}
        receipt_path = Path(tmpdir) / "test_123.json"
        with open(receipt_path, "w") as f:
            json.dump(receipt, f)

        import cell.agents.receipt_agent as mod
        old_dir = mod.RECEIPT_DIR
        mod.RECEIPT_DIR = tmpdir

        try:
            agent = ReceiptLookupAgent()
            result = agent.run({"receipt_id": "test_123"})
            assert result.ok
            assert result.output["found"]
            assert result.output["receipt"]["receipt_id"] == "test_123"
        finally:
            mod.RECEIPT_DIR = old_dir


def test_receipt_lookup_not_found():
    """ReceiptLookupAgent returns found=False for missing receipt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import cell.agents.receipt_agent as mod
        old_dir = mod.RECEIPT_DIR
        mod.RECEIPT_DIR = tmpdir

        try:
            agent = ReceiptLookupAgent()
            result = agent.run({"receipt_id": "nonexistent"})
            assert result.ok
            assert not result.output["found"]
        finally:
            mod.RECEIPT_DIR = old_dir


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
