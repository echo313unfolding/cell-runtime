"""Policy agent — permission check and action gating.

Answers: "Is this action allowed given current policy?"

Wraps ask_pass with policy awareness. For agents that need to check
permission before proposing an action.
"""
from cell.agents.base import AgentBase, AgentResult, Permission
from cell.ask_pass import ask_pass, get_audit_log


class GateDecideAgent(AgentBase):
    name = "gate_decide"
    description = "Check if an action is allowed by current policy. Returns allow/deny decision."
    permission = Permission.READ  # Checking policy is read-only
    timeout_s = 30  # May block on user prompt
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action name to check"},
            "detail": {"type": "string", "description": "Action details for user prompt"},
            "auto": {
                "type": "boolean",
                "description": "If true, auto-approve without prompting (for testing)",
            },
        },
        "required": ["action"],
    }

    def execute(self, args: dict) -> AgentResult:
        action = args["action"]
        detail = args.get("detail", "")
        auto = args.get("auto", False)

        allowed = ask_pass(action, detail, auto=auto)

        return AgentResult(output={
            "action": action,
            "allowed": allowed,
            "audit_entry": get_audit_log()[-1] if get_audit_log() else None,
        })
