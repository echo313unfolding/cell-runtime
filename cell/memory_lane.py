"""Memory Lane — rolling state tracker for the capsule orchestrator.

Maintains a structured JSON state capsule that evolves with each
orchestrator event. Two modes:

1. Programmatic (always on): extracts structured facts from events
2. LLM-enhanced (optional): uses loaded Qwen to refine the capsule

The capsule is injected into lane system prompts to give models
session continuity.
"""

import json
import time
from dataclasses import dataclass, field


# State schema — matches the memory_lane_test.py format
EMPTY_STATE = {
    "files_mentioned": [],
    "errors_active": [],
    "errors_resolved": [],
    "current_blocker": None,
    "recent_actions": [],
    "session_summary": "",
    "turn_count": 0,
}

MAX_RECENT_ACTIONS = 10
MAX_ERRORS = 20

MEMORY_SYSTEM_PROMPT = """You are a session state tracker. Your ONLY job is to update a JSON state object based on the new event.

Rules:
1. Output ONLY valid JSON — no markdown fences, no explanation.
2. Keep lists deduplicated and concise (max 10 recent_actions, max 20 errors).
3. current_blocker: null if nothing blocking, or short string.
4. session_summary: 1-2 sentence summary of the session so far.
5. Preserve existing state and only modify what the new event changes.

Schema:
{
    "files_mentioned": ["files referenced in the session"],
    "errors_active": ["currently unresolved errors"],
    "errors_resolved": ["errors that were fixed"],
    "current_blocker": null or "description",
    "recent_actions": ["last N actions taken"],
    "session_summary": "brief session summary",
    "turn_count": N
}"""


@dataclass
class MemoryLane:
    """Rolling state tracker for capsule sessions."""

    state: dict = field(default_factory=lambda: dict(EMPTY_STATE))
    event_log: list = field(default_factory=list)
    llm_enabled: bool = True
    _llm_call: object = None  # Set by orchestrator: fn(system, user) -> str

    def reset(self):
        """Clear state for a new session."""
        self.state = dict(EMPTY_STATE)
        self.state["files_mentioned"] = []
        self.state["errors_active"] = []
        self.state["errors_resolved"] = []
        self.state["recent_actions"] = []
        self.event_log = []

    def get_capsule(self) -> dict:
        """Return current state capsule."""
        return dict(self.state)

    def get_capsule_prompt(self) -> str:
        """Format capsule for injection into a lane's system prompt."""
        if self.state["turn_count"] == 0:
            return ""
        capsule = self.get_capsule()
        return (
            "\n\n--- Session Memory ---\n"
            f"{json.dumps(capsule, indent=2)}\n"
            "--- End Session Memory ---"
        )

    def ingest(self, event: dict):
        """Process an orchestrator event and update state.

        Event dict should have:
            intent, model, output, user_input, tool_calls (optional)
        """
        self.event_log.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            **{k: v for k, v in event.items() if k != "output" or len(str(v)) < 500},
        })

        # Always do programmatic update
        self._programmatic_update(event)

        # Try LLM-enhanced update if enabled and callable
        if self.llm_enabled and self._llm_call:
            self._llm_update(event)

    def _programmatic_update(self, event: dict):
        """Extract structured facts from an event without LLM."""
        self.state["turn_count"] += 1

        intent = event.get("intent", "")
        output = str(event.get("output", ""))
        user_input = str(event.get("user_input", ""))
        model = event.get("model", "")

        # Track recent actions
        action = f"[{intent}] {user_input[:80]}" if user_input else f"[{intent}] (no input)"
        self.state["recent_actions"].append(action)
        self.state["recent_actions"] = self.state["recent_actions"][-MAX_RECENT_ACTIONS:]

        # Extract file mentions from output and input
        combined = output + " " + user_input
        for token in combined.split():
            cleaned = token.strip(".,;:()\"'`")
            if ("." in cleaned and "/" in cleaned) or cleaned.endswith((".py", ".cpp", ".json", ".gguf", ".sh", ".md", ".yaml", ".yml", ".toml", ".cfg", ".h", ".cuh", ".cu")):
                if cleaned not in self.state["files_mentioned"] and len(cleaned) < 200:
                    self.state["files_mentioned"].append(cleaned)

        # Detect errors in output
        error_keywords = ["error", "failed", "failure", "exception", "traceback", "segfault", "undefined reference"]
        output_lower = output.lower()
        if any(kw in output_lower for kw in error_keywords):
            # Grab first line with an error keyword
            for line in output.split("\n"):
                if any(kw in line.lower() for kw in error_keywords):
                    err = line.strip()[:120]
                    if err and err not in self.state["errors_active"]:
                        self.state["errors_active"].append(err)
                        self.state["errors_active"] = self.state["errors_active"][-MAX_ERRORS:]
                    break

        # Detect fixes
        fix_keywords = ["fixed", "resolved", "passes now", "works now", "success"]
        if any(kw in output_lower for kw in fix_keywords):
            # Move active errors to resolved if output mentions fix
            newly_resolved = []
            for err in self.state["errors_active"]:
                # Simple heuristic: if the fix output mentions keywords from the error
                err_words = set(err.lower().split())
                if len(err_words & set(output_lower.split())) >= 2:
                    newly_resolved.append(err)
            for err in newly_resolved:
                self.state["errors_active"].remove(err)
                if err not in self.state["errors_resolved"]:
                    self.state["errors_resolved"].append(err)

        # Update blocker
        if self.state["errors_active"]:
            self.state["current_blocker"] = self.state["errors_active"][-1]
        else:
            self.state["current_blocker"] = None

        # Simple session summary
        n = self.state["turn_count"]
        self.state["session_summary"] = (
            f"{n} turns. Last: {intent} via {model}."
        )

    def _llm_update(self, event: dict):
        """Use loaded LLM to refine the state capsule."""
        user_input = event.get("user_input", "")
        output = event.get("output", "")
        # Truncate output to avoid blowing context
        output_truncated = output[:500] if len(output) > 500 else output

        prompt = (
            f"Current state:\n{json.dumps(self.state, indent=2)}\n\n"
            f"New event (turn {self.state['turn_count']}):\n"
            f"User: {user_input[:200]}\n"
            f"Intent: {event.get('intent', '?')}\n"
            f"Model: {event.get('model', '?')}\n"
            f"Output (truncated): {output_truncated}\n\n"
            f"Output the updated state JSON only."
        )

        try:
            response = self._llm_call(MEMORY_SYSTEM_PROMPT, prompt)
            parsed = _parse_json(response)
            if parsed and isinstance(parsed, dict):
                # Validate schema — only accept if it has expected keys
                expected_keys = {"files_mentioned", "errors_active", "errors_resolved",
                                 "current_blocker", "recent_actions", "session_summary", "turn_count"}
                if expected_keys & set(parsed.keys()):
                    # Merge: LLM output wins but keep programmatic fields it might miss
                    self.state.update(parsed)
                    self.state["turn_count"] = max(
                        self.state.get("turn_count", 0),
                        event.get("_turn", self.state["turn_count"])
                    )
        except Exception:
            # LLM update failed — programmatic state stands
            pass

    def to_dict(self) -> dict:
        """Full state for serialization."""
        return {
            "state": self.state,
            "event_count": len(self.event_log),
            "llm_enabled": self.llm_enabled,
        }


def _parse_json(text: str) -> dict | None:
    """Try to extract JSON from model response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None
