"""Sentinel-specific MCP tools for the cell daemon.

These extend the base cell MCP server with security triage capabilities.
They delegate to the SentinelHybridAdapter which wraps the frozen hybrid stack.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp.types import Tool, TextContent, CallToolResult


def sentinel_tools() -> list[Tool]:
    """Return Sentinel-specific MCP tool definitions."""
    return [
        Tool(
            name="sentinel_triage",
            description=(
                "Run a security alert through the full Sentinel pipeline "
                "(SSM state -> Qwen LLM -> post-LLM gates). Returns structured verdict "
                "with severity, benign classification, recommended actions, and gate info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "alert_text": {
                        "type": "string",
                        "description": "The security alert or log entry to analyze",
                    },
                },
                "required": ["alert_text"],
            },
        ),
        Tool(
            name="sentinel_status",
            description=(
                "Show Sentinel state: whether the specialist is loaded, "
                "SSM state summary, gate policy, recent verdict counts."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def handle_sentinel_tool(name: str, arguments: dict, orchestrator) -> CallToolResult:
    """Handle a Sentinel MCP tool call."""
    if name == "sentinel_triage":
        return await _handle_triage(arguments, orchestrator)
    elif name == "sentinel_status":
        return await _handle_sentinel_status(orchestrator)
    return CallToolResult(content=[TextContent(type="text", text=f"Unknown sentinel tool: {name}")])


async def _handle_triage(arguments: dict, orchestrator) -> CallToolResult:
    """Run alert through the full Sentinel pipeline."""
    alert_text = arguments.get("alert_text", "")
    if not alert_text:
        return CallToolResult(content=[TextContent(type="text", text="Error: 'alert_text' required")])

    result = orchestrator.process(
        alert_text,
        force_model="qwen2.5-sentinel",
        use_tools=False,
    )

    if "error" in result:
        return CallToolResult(content=[TextContent(
            type="text", text=f"Sentinel error: {result['error']}"
        )])

    lines = [
        f"Verdict: {result.get('verdict', {}).get('severity', 'unknown')}",
        f"Benign: {result.get('verdict', {}).get('is_benign', 'unknown')}",
        f"Model: {result['model']}",
        f"Specialist: {result.get('specialist', 'generic')}",
        f"Gate fired: {result.get('gate_fired', False)}",
    ]
    if result.get('gate_rules'):
        lines.append(f"Gate rules: {result['gate_rules']}")
    lines.append(f"Wall time: {result['wall_time_s']:.1f}s")
    lines.append(f"Receipt: {result.get('receipt', 'none')}")
    lines.append(f"---")
    lines.append(result.get("output", ""))

    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])


async def _handle_sentinel_status(orchestrator) -> CallToolResult:
    """Show Sentinel specialist state."""
    status = orchestrator.status()
    specialists = status.get("specialists", [])
    sentinel_state = None
    for s in specialists:
        if s.get("name") == "sentinel_hybrid":
            sentinel_state = s
            break

    lines = [
        f"Loaded model: {status['loaded_model']}",
        f"Sentinel specialist: {'found' if sentinel_state else 'not registered'}",
    ]
    if sentinel_state:
        lines.append(f"  SSM initialized: {sentinel_state.get('ssm_initialized', False)}")
        lines.append(f"  Verdicts processed: {sentinel_state.get('verdict_count', 0)}")
        lines.append(f"  Gate policy: {sentinel_state.get('gate_policy', 'default')}")

    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])
