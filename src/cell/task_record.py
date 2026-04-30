"""Task record — shared handoff format for the local model cascade.

Every worker (Qwen, Sentinel, Zamba2) reads and writes this format.
The CPU router creates it, the GPU worker fills it, the next worker
can pick it up without losing context.

Usage:
    from tools.task_record import TaskRecord

    # Router creates
    task = TaskRecord.create(
        intent="security_triage",
        input_text="Alert: reverse shell detected on port 4444",
    )

    # Worker fills
    task.set_result(
        model="qwen2.5-sentinel",
        verdict="escalate",
        confidence=0.92,
        reasoning="Reverse shell on non-standard port...",
    )

    # Save / load
    task.save("~/receipts/tasks/")
    task = TaskRecord.load("~/receipts/tasks/task_abc123.json")

    # Escalate
    task.escalate("needs frontier model analysis")
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_id() -> str:
    """Short unique task ID from timestamp + random bits."""
    raw = f"{time.time():.6f}:{os.getpid()}:{os.urandom(4).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# Intent categories for routing
INTENTS = {
    "coding":           "Code generation, editing, debugging",
    "security_triage":  "Security alert analysis and classification",
    "general":          "General assistance, Q&A, summarization",
    "reasoning":        "Multi-step reasoning, math, logic",
    "hybrid_experiment":"SSM/hybrid architecture experiments",
}

# Model → intent mapping (default routing table)
DEFAULT_ROUTES = {
    "coding":           "qwen2.5-coder-3b",
    "security_triage":  "qwen2.5-sentinel",
    "general":          "qwen2.5-coder-3b",
    "reasoning":        "qwen2.5-coder-3b",
    "hybrid_experiment":"zamba2-2.7b",
}


@dataclass
class TaskRecord:
    task_id: str
    created_at: str
    intent: str
    input_text: str

    # Routing
    routed_model: str = ""
    route_reason: str = ""

    # Result (filled by worker)
    model: str = ""
    verdict: str = ""         # benign/suspicious/escalate OR free-form answer
    confidence: float = 0.0
    reasoning: str = ""
    output_text: str = ""
    tool_calls: list = field(default_factory=list)

    # Timing
    started_at: str = ""
    completed_at: str = ""
    elapsed_ms: float = 0.0

    # Swap cost (filled by orchestrator)
    swap_cost_s: float = 0.0
    swap_from: str = ""

    # Escalation
    escalated: bool = False
    escalation_reason: str = ""
    escalated_to: str = ""

    # Evidence
    receipt_refs: list = field(default_factory=list)

    @classmethod
    def create(cls, intent: str, input_text: str, routed_model: str = "") -> "TaskRecord":
        if not routed_model:
            routed_model = DEFAULT_ROUTES.get(intent, "qwen2.5-coder-3b")
        return cls(
            task_id=_task_id(),
            created_at=_now_iso(),
            intent=intent,
            input_text=input_text,
            routed_model=routed_model,
        )

    def start(self):
        self.started_at = _now_iso()

    def set_result(self, model: str, verdict: str = "", confidence: float = 0.0,
                   reasoning: str = "", output_text: str = "", tool_calls: list = None,
                   receipt_ref: str = ""):
        self.model = model
        self.verdict = verdict
        self.confidence = confidence
        self.reasoning = reasoning
        self.output_text = output_text
        if tool_calls:
            self.tool_calls = tool_calls
        if receipt_ref:
            self.receipt_refs.append(receipt_ref)
        self.completed_at = _now_iso()
        if self.started_at:
            t0 = datetime.fromisoformat(self.started_at)
            t1 = datetime.fromisoformat(self.completed_at)
            self.elapsed_ms = (t1 - t0).total_seconds() * 1000

    def escalate(self, reason: str, escalated_to: str = "tier3_claude"):
        self.escalated = True
        self.escalation_reason = reason
        self.escalated_to = escalated_to

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "TaskRecord":
        with open(os.path.expanduser(path)) as f:
            return cls.from_dict(json.load(f))

    def save(self, directory: str = "~/receipts/tasks") -> str:
        d = Path(os.path.expanduser(directory))
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"task_{self.task_id}.json"
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(path)

    def summary(self) -> str:
        status = "ESCALATED" if self.escalated else ("DONE" if self.completed_at else "PENDING")
        v = f" → {self.verdict}" if self.verdict else ""
        c = f" ({self.confidence:.0%})" if self.confidence else ""
        t = f" [{self.elapsed_ms:.0f}ms]" if self.elapsed_ms else ""
        return f"[{self.task_id}] {self.intent} {status}{v}{c}{t} model={self.model or self.routed_model}"
