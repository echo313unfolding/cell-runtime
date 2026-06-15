"""Poetica receipt — audit trail for every compilation."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class Receipt:
    """Immutable record of a compilation."""
    source_hash: str
    target: str
    gate_level: int
    gate_policy: str
    decisions: List[Dict[str, Any]]
    output_hash: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "schema": "poetica.receipt.v1",
            "source_hash": self.source_hash,
            "target": self.target,
            "gate_level": self.gate_level,
            "gate_policy": self.gate_policy,
            "all_allowed": all(d["verdict"] == "ALLOW" for d in self.decisions),
            "decisions": self.decisions,
            "output_hash": self.output_hash,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def hash_output(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()
