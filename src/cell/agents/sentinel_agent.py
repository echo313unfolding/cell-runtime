"""Sentinel agent — security triage with context pack.

Answers: "Given all that context, what is the risk/verdict/tool call?"

This agent assembles a context pack from RAG, graph, and SSM agents,
then delegates to the Sentinel hybrid stack for the actual verdict.
"""
from cell.agents.base import AgentBase, AgentResult, Permission


class SentinelTriageAgent(AgentBase):
    name = "sentinel_triage"
    description = (
        "Run a security event through the full Sentinel pipeline with context pack. "
        "Assembles SSM state, RAG context, and graph links, then calls the "
        "Sentinel hybrid stack (SSM -> LLM -> gates) for a verdict."
    )
    permission = Permission.READ  # Read: the verdict is informational
    timeout_s = 60
    input_schema = {
        "type": "object",
        "properties": {
            "alert_text": {"type": "string", "description": "The alert or log entry to analyze"},
            "entity_id": {
                "type": "string",
                "description": "Entity to check state for (optional, extracted from alert if missing)",
            },
            "context_pack": {
                "type": "object",
                "description": "Pre-assembled context (optional — if missing, will be assembled from agents)",
            },
        },
        "required": ["alert_text"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "object"},
            "context_pack": {"type": "object"},
            "gate_fired": {"type": "boolean"},
        },
    }

    # Set by orchestrator at startup
    _agent_registry = None
    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        alert_text = args["alert_text"]
        entity_id = args.get("entity_id")
        context_pack = args.get("context_pack")

        # Assemble context pack if not provided
        if not context_pack:
            context_pack = self._assemble_context(alert_text, entity_id)

        # Delegate to orchestrator's Sentinel path if available
        if self._orchestrator:
            result = self._orchestrator.process(
                alert_text,
                force_model="qwen2.5-sentinel",
                use_tools=False,
            )
            return AgentResult(output={
                "verdict": result.get("verdict", {}),
                "output": result.get("output", ""),
                "model": result.get("model", ""),
                "specialist": result.get("specialist", ""),
                "gate_fired": result.get("gate_fired", False),
                "gate_rules": result.get("gate_rules", []),
                "context_pack": context_pack,
                "wall_time_s": result.get("wall_time_s", 0),
                "receipt": result.get("receipt", ""),
            })

        # No orchestrator — return context pack without verdict
        return AgentResult(output={
            "verdict": {"severity": "unknown", "note": "No orchestrator available"},
            "context_pack": context_pack,
            "gate_fired": False,
        })

    def _assemble_context(self, alert_text: str, entity_id: str = None) -> dict:
        """Assemble context pack from available agents."""
        pack = {
            "alert": {"text": alert_text},
            "ssm_state": {},
            "rag_context": [],
            "graph_links": [],
        }

        if not self._agent_registry:
            return pack

        # SSM state
        if entity_id:
            ssm = self._agent_registry.run("ssm_get_state", {"entity_id": entity_id})
            if ssm.ok:
                pack["ssm_state"] = ssm.output

        # RAG context
        rag = self._agent_registry.run("rag_lookup", {
            "query": alert_text[:100],
            "scope": "claims",
            "limit": 5,
        })
        if rag.ok:
            pack["rag_context"] = rag.output.get("results", [])

        # Graph links
        if entity_id:
            graph = self._agent_registry.run("graph_neighbors", {
                "entity_id": entity_id,
                "limit": 10,
            })
            if graph.ok:
                pack["graph_links"] = graph.output.get("neighbors", [])

        return pack
