"""SSM agent — sequential memory state queries and updates.

Answers: "What has this entity been doing over time?"

Backend: Sentinel SSM state (~/tools/sentinel/sentinel.db)
  - alerts: detection events
  - verdicts: classification decisions
  - investigations: case tracking
"""
import os
import sqlite3
import time

from cell.agents.base import AgentBase, AgentResult, Permission

SENTINEL_DB = os.path.expanduser("~/tools/sentinel/sentinel.db")


class SSMGetStateAgent(AgentBase):
    name = "ssm_get_state"
    description = "Get the current SSM state for an entity (IP, user, process). Returns recent events, verdicts, and trend."
    permission = Permission.READ
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "Entity identifier (e.g., IP:203.0.113.5, user:root)"},
            "limit": {"type": "integer", "description": "Max events to return (default: 10)"},
        },
        "required": ["entity_id"],
    }

    def execute(self, args: dict) -> AgentResult:
        entity_id = args["entity_id"]
        limit = args.get("limit", 10)

        if not os.path.exists(SENTINEL_DB):
            return AgentResult(output={
                "entity": entity_id,
                "found": False,
                "reason": "sentinel.db not available",
                "events": [],
                "verdicts": [],
            })

        conn = sqlite3.connect(f"file:{SENTINEL_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Query alerts mentioning this entity
            alerts = []
            try:
                cursor = conn.execute(
                    "SELECT * FROM alerts WHERE alert_text LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (f"%{entity_id}%", limit))
                alerts = [dict(row) for row in cursor]
            except sqlite3.OperationalError:
                pass

            # Query verdicts for this entity
            verdicts = []
            try:
                cursor = conn.execute(
                    "SELECT * FROM verdicts WHERE input_text LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (f"%{entity_id}%", limit))
                verdicts = [dict(row) for row in cursor]
            except sqlite3.OperationalError:
                pass

            # Compute trend
            event_count = len(alerts)
            verdict_list = [v.get("severity", "") for v in verdicts]

            return AgentResult(output={
                "entity": entity_id,
                "found": event_count > 0 or len(verdicts) > 0,
                "event_count": event_count,
                "recent_alerts": alerts[:5],
                "recent_verdicts": verdicts[:5],
                "verdict_summary": verdict_list[:10],
                "trend": "active" if event_count > 3 else "quiet",
            })
        finally:
            conn.close()


class SSMUpdateEventAgent(AgentBase):
    name = "ssm_update_event"
    description = "Record a new event in the SSM state store. Updates sequential memory for the entity."
    permission = Permission.WRITE  # Write-level: auto-approve with log
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "Entity identifier"},
            "event_type": {"type": "string", "description": "Event type (alert, verdict, observation)"},
            "event_data": {"type": "string", "description": "Event content/description"},
        },
        "required": ["entity_id", "event_type", "event_data"],
    }

    def execute(self, args: dict) -> AgentResult:
        entity_id = args["entity_id"]
        event_type = args["event_type"]
        event_data = args["event_data"]

        if not os.path.exists(SENTINEL_DB):
            return AgentResult(error="sentinel.db not available")

        conn = sqlite3.connect(SENTINEL_DB)
        try:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")

            if event_type == "alert":
                conn.execute(
                    "INSERT INTO alerts (timestamp, alert_text, source) VALUES (?, ?, ?)",
                    (timestamp, f"[{entity_id}] {event_data}", "ssm_agent"))
            elif event_type == "verdict":
                conn.execute(
                    "INSERT INTO verdicts (timestamp, input_text, severity) VALUES (?, ?, ?)",
                    (timestamp, f"[{entity_id}] {event_data}", "pending"))

            conn.commit()
            return AgentResult(output={
                "entity": entity_id,
                "event_type": event_type,
                "recorded": True,
                "timestamp": timestamp,
            })
        except sqlite3.OperationalError as e:
            return AgentResult(error=f"SSM write error: {e}")
        finally:
            conn.close()
