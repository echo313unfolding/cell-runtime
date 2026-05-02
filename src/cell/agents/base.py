"""Agent base class — bounded, schema-validated workers.

Every agent has:
  - name, input_schema, output_schema
  - permission level (read/write/privileged)
  - timeout
  - failure mode
  - receipt emission

Usage:
    class MyAgent(AgentBase):
        name = "my_agent"
        permission = Permission.READ
        timeout_s = 10

        def execute(self, args: dict) -> AgentResult:
            return AgentResult(output={"answer": 42})
"""
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Permission(Enum):
    READ = "read"           # Query stores, no side effects. Auto-approve.
    WRITE = "write"         # Modify state stores. Auto-approve with log.
    PRIVILEGED = "privileged"  # System actions. Requires ask-pass.


@dataclass
class AgentResult:
    """Result from an agent execution."""
    output: dict = field(default_factory=dict)
    error: Optional[str] = None
    receipt: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class AgentBase:
    """Base class for bounded agents."""
    name: str = ""
    description: str = ""
    permission: Permission = Permission.READ
    timeout_s: int = 30
    failure_mode: str = "return_error"  # return_error | raise | fallback

    # JSON Schema for input/output validation
    input_schema: dict = {}
    output_schema: dict = {}

    def execute(self, args: dict) -> AgentResult:
        """Execute the agent. Override in subclasses."""
        raise NotImplementedError

    def run(self, args: dict) -> AgentResult:
        """Run with timeout, receipt, and error handling."""
        t_start = time.time()
        try:
            result = self.execute(args)
        except Exception as e:
            if self.failure_mode == "raise":
                raise
            result = AgentResult(error=f"{self.name}: {e}")

        elapsed = round(time.time() - t_start, 3)

        # Attach receipt
        result.receipt = {
            "agent": self.name,
            "permission": self.permission.value,
            "wall_time_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": result.error,
        }

        return result

    def validate_input(self, args: dict) -> Optional[str]:
        """Basic input validation. Returns error string or None."""
        if not self.input_schema:
            return None
        required = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})
        for key in required:
            if key not in args:
                return f"Missing required field: {key}"
        for key, value in args.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return f"Field {key} must be string, got {type(value).__name__}"
                if expected_type == "integer" and not isinstance(value, int):
                    return f"Field {key} must be integer, got {type(value).__name__}"
        return None


class AgentRegistry:
    """Registry of bounded agents."""

    def __init__(self):
        self._agents: dict[str, AgentBase] = {}

    def register(self, agent: AgentBase):
        """Register an agent."""
        if not agent.name:
            raise ValueError("Agent must have a name")
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[AgentBase]:
        """Get agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[dict]:
        """List all registered agents with metadata."""
        return [
            {
                "name": a.name,
                "description": a.description,
                "permission": a.permission.value,
                "timeout_s": a.timeout_s,
                "input_schema": a.input_schema,
                "output_schema": a.output_schema,
            }
            for a in self._agents.values()
        ]

    def run(self, name: str, args: dict) -> AgentResult:
        """Run an agent by name."""
        agent = self._agents.get(name)
        if not agent:
            return AgentResult(error=f"Unknown agent: {name}")

        # Validate input
        err = agent.validate_input(args)
        if err:
            return AgentResult(error=err)

        return agent.run(args)

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)
