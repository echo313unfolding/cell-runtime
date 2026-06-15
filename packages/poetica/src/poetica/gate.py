"""Poetica gate — capability-level access control for operations.

Every operation must pass the gate before code generation.
Fail-closed: unknown operations are rejected.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List


class GateLevel(IntEnum):
    """Capability levels. Higher = more powerful operations unlocked."""
    L1 = 1  # Pure: seed, emit, flow, bloom (no side effects)
    L2 = 2  # + compare, verify, if/when (logic)
    L3 = 3  # + pack, grow, learn (transformation)
    L4 = 4  # + lift, use (external interaction)
    L5 = 5  # + unrestricted (full access)


# Operations allowed at each level (cumulative)
_LEVEL_OPS = {
    1: {'seed', 'emit', 'flow', 'bloom', 'name', 'remember', 'text'},
    2: {'if', 'when', 'when_in', 'for'},
    3: {'pack', 'grow', 'learn'},
    4: {'lift', 'use'},
    5: set(),  # L5 allows everything
}

# Operations that touch external systems
_EXTERNAL_OPS = {'lift', 'use'}


def _ops_at_level(level: int) -> set:
    """Return all operations allowed at a given level."""
    allowed = set()
    for lvl in range(1, min(level, 5) + 1):
        allowed |= _LEVEL_OPS[lvl]
    return allowed


@dataclass
class GateDecision:
    """Record of a single gate check."""
    op: str
    verdict: str       # "ALLOW" or "REJECT"
    reason: str        # reason code
    level: int         # required level for this op
    timestamp: str = ""
    input_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "verdict": self.verdict,
            "reason": self.reason,
            "level": self.level,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
        }


class GateError(Exception):
    """Raised when the gate rejects an operation."""
    def __init__(self, decision: GateDecision):
        self.decision = decision
        super().__init__(f"REJECT op='{decision.op}': {decision.reason}")


class Gate:
    """Capability gate. Checks every operation against the current level.

    Args:
        level: Maximum capability level (1-5).
        allow_external: Whether to allow operations that touch external systems.

    Usage:
        gate = Gate(level=2)
        gate.check(ir)  # raises GateError if any op exceeds level 2
    """

    def __init__(self, level: int = 1, allow_external: bool = False):
        if level < 1 or level > 5:
            raise ValueError(f"Level must be 1-5, got {level}")
        self.level = level
        self.allow_external = allow_external
        self._allowed = _ops_at_level(level)

        # Policy hash for audit trail
        policy = json.dumps({"level": level, "allow_external": allow_external}, sort_keys=True)
        self.policy_hash = hashlib.sha256(policy.encode()).hexdigest()[:16]

    def check(self, ir: Dict[str, Any]) -> List[GateDecision]:
        """Check all operations in an IR plan. Raises GateError on first rejection."""
        decisions = []
        for op_spec in ir.get("ops", []):
            d = self._check_op(op_spec)
            decisions.append(d)
            if d.verdict == "REJECT":
                raise GateError(d)
        return decisions

    def check_all(self, ir: Dict[str, Any]) -> List[GateDecision]:
        """Check all operations, returning all decisions (does not raise)."""
        return [self._check_op(op) for op in ir.get("ops", [])]

    def _check_op(self, op_spec: Dict[str, Any]) -> GateDecision:
        op_name = op_spec.get("op", "")
        now = datetime.now(timezone.utc).isoformat()
        input_hash = hashlib.sha256(
            json.dumps(op_spec, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Find what level this op requires
        required_level = self._required_level(op_name)

        if required_level is None:
            return GateDecision(
                op=op_name, verdict="REJECT", reason="UNKNOWN-OP",
                level=0, timestamp=now, input_hash=input_hash,
            )

        if op_name in _EXTERNAL_OPS and not self.allow_external:
            return GateDecision(
                op=op_name, verdict="REJECT", reason="EXTERNAL-DENIED",
                level=required_level, timestamp=now, input_hash=input_hash,
            )

        if required_level > self.level:
            return GateDecision(
                op=op_name, verdict="REJECT", reason="LEVEL-EXCEEDED",
                level=required_level, timestamp=now, input_hash=input_hash,
            )

        return GateDecision(
            op=op_name, verdict="ALLOW", reason="OK",
            level=required_level, timestamp=now, input_hash=input_hash,
        )

    def _required_level(self, op_name: str) -> int | None:
        for lvl in range(1, 6):
            if op_name in _LEVEL_OPS[lvl]:
                return lvl
        if self.level == 5:
            return 5  # L5 allows everything
        return None
