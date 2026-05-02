"""Receipt agent — audit trail read/write.

Answers: "What happened? What's the evidence?"

Every agent call, model inference, memory update, graph write, and command
emits a receipt. This agent queries and writes them.
"""
import json
import os
import time
from pathlib import Path

from cell.agents.base import AgentBase, AgentResult, Permission

RECEIPT_DIR = os.path.expanduser("~/receipts")


class ReceiptLookupAgent(AgentBase):
    name = "receipt_lookup"
    description = "Look up a receipt by ID or search by keyword."
    permission = Permission.READ
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "receipt_id": {"type": "string", "description": "Receipt ID or keyword to search"},
        },
        "required": ["receipt_id"],
    }

    def execute(self, args: dict) -> AgentResult:
        receipt_id = args["receipt_id"]
        receipt_dir = Path(RECEIPT_DIR)
        if not receipt_dir.exists():
            return AgentResult(error="Receipt directory not found")

        # Try exact file match first
        for pattern in [f"{receipt_id}.json", f"*{receipt_id}*.json"]:
            matches = list(receipt_dir.rglob(pattern))
            if matches:
                # Return the first match's content
                try:
                    with open(matches[0]) as f:
                        data = json.load(f)
                    return AgentResult(output={
                        "found": True,
                        "path": str(matches[0]),
                        "receipt": data,
                    })
                except (json.JSONDecodeError, OSError) as e:
                    return AgentResult(error=f"Failed to read receipt: {e}")

        return AgentResult(output={
            "found": False,
            "receipt_id": receipt_id,
        })


class ReceiptWriteAgent(AgentBase):
    name = "receipt_write"
    description = "Write a receipt for an event/decision. Every action should emit a receipt."
    permission = Permission.WRITE  # Write-level: auto-approve with log
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "What action was taken"},
            "result": {"type": "object", "description": "Result data to record"},
            "subdirectory": {
                "type": "string",
                "description": "Subdirectory under ~/receipts/ (default: cell/agents/)",
            },
        },
        "required": ["action", "result"],
    }

    def execute(self, args: dict) -> AgentResult:
        action = args["action"]
        result_data = args["result"]
        subdir = args.get("subdirectory", "cell/agents")

        now = time.strftime("%Y%m%dT%H%M%SZ")
        receipt_id = f"{action}_{now}"

        receipt = {
            "receipt_id": receipt_id,
            "action": action,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result": result_data,
        }

        out_dir = Path(RECEIPT_DIR) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{receipt_id}.json"
        with open(path, "w") as f:
            json.dump(receipt, f, indent=2, default=str)

        return AgentResult(output={
            "receipt_id": receipt_id,
            "path": str(path),
            "written": True,
        })
