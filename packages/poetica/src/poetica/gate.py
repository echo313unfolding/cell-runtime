"""Poetica gate — capability-level access control for operations."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List


class GateLevel(IntEnum):
    L1 = 1  # Pure: seed, emit, flow, bloom (no side effects)
    L2 = 2  # + logic: if, when, for, else
    L3 = 3  # + transform: pack, grow, learn
    L4 = 4  # + external: lift, use
    L5 = 5  # unrestricted


_LEVEL_OPS = {
    1: {'seed', 'emit', 'flow', 'bloom', 'name', 'remember', 'text'},
    2: {'if', 'when', 'when_in', 'for', 'else_when', 'else'},
    3: {'pack', 'grow', 'learn'},
    4: {'lift', 'use'},
    5: set(),
}

_EXTERNAL_OPS = {'lift', 'use'}


def _ops_at_level(level: int) -> set:
    allowed = set()
    for lvl in range(1, min(level, 5) + 1):
        allowed |= _LEVEL_OPS[lvl]
    return allowed


@dataclass
class GateDecision:
    op: str
    verdict: str
    reason: str
    level: int
    timestamp: str = ""
    input_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "op": self.op, "verdict": self.verdict, "reason": self.reason,
            "level": self.level, "timestamp": self.timestamp, "input_hash": self.input_hash,
        }


class GateError(Exception):
    def __init__(self, decision: GateDecision):
        self.decision = decision
        super().__init__(f"REJECT op='{decision.op}': {decision.reason}")


class Gate:
    def __init__(self, level: int = 1, allow_external: bool = False):
        if level < 1 or level > 5:
            raise ValueError(f"Level must be 1-5, got {level}")
        self.level = level
        self.allow_external = allow_external
        self._allowed = _ops_at_level(level)
        policy = json.dumps({"level": level, "allow_external": allow_external}, sort_keys=True)
        self.policy_hash = hashlib.sha256(policy.encode()).hexdigest()[:16]

    def check(self, ir: Dict[str, Any]) -> List[GateDecision]:
        decisions = []
        for op_spec in ir.get("ops", []):
            d = self._check_op(op_spec)
            decisions.append(d)
            if d.verdict == "REJECT":
                raise GateError(d)
        return decisions

    def check_all(self, ir: Dict[str, Any]) -> List[GateDecision]:
        return [self._check_op(op) for op in ir.get("ops", [])]

    def _check_op(self, op_spec: Dict[str, Any]) -> GateDecision:
        op_name = op_spec.get("op", "")
        now = datetime.now(timezone.utc).isoformat()
        input_hash = hashlib.sha256(
            json.dumps(op_spec, sort_keys=True).encode()
        ).hexdigest()[:16]

        required_level = self._required_level(op_name)

        if required_level is None:
            return GateDecision(op=op_name, verdict="REJECT", reason="UNKNOWN-OP",
                                level=0, timestamp=now, input_hash=input_hash)

        if op_name in _EXTERNAL_OPS and not self.allow_external:
            return GateDecision(op=op_name, verdict="REJECT", reason="EXTERNAL-DENIED",
                                level=required_level, timestamp=now, input_hash=input_hash)

        if required_level > self.level:
            return GateDecision(op=op_name, verdict="REJECT", reason="LEVEL-EXCEEDED",
                                level=required_level, timestamp=now, input_hash=input_hash)

        return GateDecision(op=op_name, verdict="ALLOW", reason="OK",
                            level=required_level, timestamp=now, input_hash=input_hash)

    def _required_level(self, op_name: str) -> int | None:
        for lvl in range(1, 6):
            if op_name in _LEVEL_OPS[lvl]:
                return lvl
        if self.level == 5:
            return 5
        return None
