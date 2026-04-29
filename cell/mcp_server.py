#!/usr/bin/env python3
"""Capsule MCP Server — expose the 3-model local assistant to Claude Code.

Tools:
  capsule_status    — current model, roster, swap history
  capsule_classify  — classify intent and show routing (no generation)
  capsule_generate  — full pipeline: classify → route → swap → generate → log

Register:
  claude mcp add --scope user capsule /home/voidstr3m33/tools/capsule/mcp_wrapper.sh
"""
import json
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

# Add tools/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from capsule.orchestrator import Orchestrator

server = Server("capsule")


def _orch() -> Orchestrator:
    """Lazy singleton orchestrator."""
    if not hasattr(_orch, "_instance"):
        _orch._instance = Orchestrator()
    return _orch._instance


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="capsule_status",
            description=(
                "Show current state of the local 3-model assistant: which model is loaded, "
                "roster, swap policy, and swap history this session."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="capsule_classify",
            description=(
                "Classify user input into an intent (coding, security_triage, reasoning, general) "
                "and show which model would handle it. Does NOT generate or swap — just routing info. "
                "Use this to preview routing before committing to a generation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The user input text to classify",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="capsule_generate",
            description=(
                "Full local assistant pipeline: classify intent → route to model → swap if needed → "
                "generate response → log task receipt. Returns the model's response plus metadata "
                "(intent, model used, swap info, tok/s). Use for delegating work to local models. "
                "WARNING: This triggers Ollama model loading which takes 2-30s depending on swap state."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The prompt/question to send to the local model",
                    },
                    "force_model": {
                        "type": "string",
                        "description": "Override routing — force a specific model (smollm3, qwen2.5-sentinel, qwen2.5-coder:latest)",
                    },
                    "use_tools": {
                        "type": "boolean",
                        "description": "Enable tool use — model can call tools (read_file, grep, shell, web_search, memory_search). Default false.",
                        "default": False,
                    },
                },
                "required": ["text"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    try:
        if name == "capsule_status":
            return await _handle_status()
        elif name == "capsule_classify":
            return await _handle_classify(arguments)
        elif name == "capsule_generate":
            return await _handle_generate(arguments)
        else:
            return CallToolResult(content=[TextContent(
                type="text", text=f"Unknown tool: {name}"
            )])
    except Exception as e:
        return CallToolResult(content=[TextContent(
            type="text", text=f"Error in {name}: {e}"
        )])


async def _handle_status() -> CallToolResult:
    orch = _orch()
    result = orch.status()
    lines = [
        f"Loaded model: {result['loaded_model']}",
        f"Swap policy: {result['swap_policy']}",
        f"Roster: {', '.join(result['roster'])}",
        f"Swaps this session: {result['swap_count']}",
    ]
    if result['swap_history']:
        lines.append("\nRecent swaps:")
        for s in result['swap_history'][-5:]:
            lines.append(f"  {s.get('from','?')} → {s.get('to','?')} ({s.get('swap_time_s',0):.1f}s)")
    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])


async def _handle_classify(arguments: dict) -> CallToolResult:
    text = arguments.get("text", "")
    if not text:
        return CallToolResult(content=[TextContent(type="text", text="Error: 'text' is required")])
    orch = _orch()
    result = orch.classify_only(text)
    lines = [
        f"Intent: {result['intent']}",
        f"Routed model: {result['routed_model']}",
        f"Current model: {result['current_model']}",
        f"Would swap: {result['would_swap']}",
        f"Policy: {result['swap_policy']}",
    ]
    return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])


async def _handle_generate(arguments: dict) -> CallToolResult:
    text = arguments.get("text", "")
    if not text:
        return CallToolResult(content=[TextContent(type="text", text="Error: 'text' is required")])
    force_model = arguments.get("force_model")
    use_tools = arguments.get("use_tools", False)
    orch = _orch()
    result = orch.process(text, force_model=force_model, use_tools=use_tools)

    if "error" in result:
        return CallToolResult(content=[TextContent(
            type="text", text=f"Generation error: {result['error']}"
        )])

    swap_info = ""
    if result.get("swapped"):
        swap_info = f"\nSwapped: yes ({result.get('swap_time_s', 0):.1f}s)"
    else:
        swap_info = "\nSwapped: no (model already loaded)"

    header = (
        f"Intent: {result['intent']}\n"
        f"Model: {result['model']}{swap_info}\n"
        f"Tokens: {result.get('tok_s', 0)} tok/s, {result['wall_time_s']:.1f}s total\n"
        f"Receipt: {result['receipt']}\n"
        f"---\n"
    )
    body = header + result.get("output", "")

    # Surface escalation requests from local model
    escalations = result.get("escalations", [])
    if escalations:
        body += "\n\n=== ESCALATION REQUESTS ===\n"
        for i, esc in enumerate(escalations, 1):
            body += f"\n[{i}] {esc.get('request_type', 'general').upper()}: {esc.get('goal', '')}\n"
            if esc.get("files"):
                body += f"    Files: {', '.join(esc['files'])}\n"
            if esc.get("proposed_change"):
                body += f"    Proposed: {esc['proposed_change']}\n"
            if esc.get("reason"):
                body += f"    Reason: {esc['reason']}\n"
            body += f"    Priority: {esc.get('priority', 'normal')}\n"

    return CallToolResult(content=[TextContent(type="text", text=body)])


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
