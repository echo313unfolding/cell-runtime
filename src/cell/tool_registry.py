#!/usr/bin/env python3
"""Tool registry — pluggable tools that local models can call through the orchestrator.

Models emit tool calls as:
```tool_call
{"name": "tool_name", "arguments": {"key": "value"}}
```

The orchestrator parses these, looks up the tool in the registry,
executes it, and feeds the result back to the model.
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, handler: Callable, description: str,
                 args_schema: dict = None):
        """Register a tool that local models can call."""
        self._tools[name] = {
            "handler": handler,
            "description": description,
            "args_schema": args_schema or {},
        }

    def list_tools(self) -> list[dict]:
        """Return tool descriptions for inclusion in system prompts."""
        return [
            {"name": name, "description": info["description"],
             "args": info["args_schema"]}
            for name, info in self._tools.items()
        ]

    def format_for_prompt(self) -> str:
        """Format tool descriptions for injection into a system prompt."""
        if not self._tools:
            return ""
        lines = ["\nYou have these tools available:"]
        for name, info in self._tools.items():
            args_desc = ""
            if info["args_schema"]:
                args_desc = f" Args: {json.dumps(info['args_schema'])}"
            lines.append(f"- {name}: {info['description']}{args_desc}")
        lines.append("")
        lines.append("To use a tool, output:")
        lines.append("```tool_call")
        lines.append('{"name": "tool_name", "arguments": {"key": "value"}}')
        lines.append("```")
        lines.append("")
        lines.append("You can call multiple tools. After all tool results come back,")
        lines.append("provide your final answer to the user.")
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name. Returns result string."""
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        try:
            result = self._tools[name]["handler"](arguments)
            if isinstance(result, dict) or isinstance(result, list):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as e:
            return f"Tool error ({name}): {e}"

    def has(self, name: str) -> bool:
        return name in self._tools


def parse_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model output. Returns list of {name, arguments}."""
    calls = []
    pattern = r'```tool_call\s*\n?(.*?)\n?```'
    for match in re.findall(pattern, text, re.DOTALL | re.IGNORECASE):
        try:
            data = json.loads(match.strip())
            if "name" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue
    return calls


# ============================================================================
# Built-in tools
# ============================================================================

def _tool_read_file(args: dict) -> str:
    """Read a file. Limited to home directory for safety."""
    path = args.get("path", "")
    if not path:
        return "Error: 'path' required"
    path = os.path.expanduser(path)
    home = os.path.expanduser("~")
    real = os.path.realpath(path)
    if not real.startswith(home):
        return f"Error: path must be under {home}"
    if not os.path.exists(real):
        return f"Error: file not found: {path}"
    try:
        with open(real) as f:
            content = f.read(10000)  # limit to 10KB
        if len(content) == 10000:
            content += "\n... (truncated at 10KB)"
        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


def _tool_grep(args: dict) -> str:
    """Search file contents. Limited to home directory."""
    pattern = args.get("pattern", "")
    path = args.get("path", os.path.expanduser("~"))
    if not pattern:
        return "Error: 'pattern' required"
    path = os.path.expanduser(path)
    home = os.path.expanduser("~")
    real = os.path.realpath(path)
    if not real.startswith(home):
        return f"Error: path must be under {home}"
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.json", "--include=*.md",
             "--include=*.sh", "--include=*.txt", "-l", pattern, real],
            capture_output=True, text=True, timeout=10,
        )
        files = result.stdout.strip().split("\n")[:20]
        if not files or files == ['']:
            return "No matches found."
        return f"Matching files ({len(files)}):\n" + "\n".join(files)
    except subprocess.TimeoutExpired:
        return "Error: search timed out"
    except Exception as e:
        return f"Error: {e}"


def _tool_list_dir(args: dict) -> str:
    """List directory contents."""
    path = args.get("path", os.path.expanduser("~"))
    path = os.path.expanduser(path)
    home = os.path.expanduser("~")
    real = os.path.realpath(path)
    if not real.startswith(home):
        return f"Error: path must be under {home}"
    if not os.path.isdir(real):
        return f"Error: not a directory: {path}"
    try:
        entries = sorted(os.listdir(real))[:50]
        result = []
        for e in entries:
            full = os.path.join(real, e)
            if os.path.isdir(full):
                result.append(f"  {e}/")
            else:
                size = os.path.getsize(full)
                result.append(f"  {e}  ({size} bytes)")
        return f"Contents of {path} ({len(entries)} items):\n" + "\n".join(result)
    except Exception as e:
        return f"Error: {e}"


def _tool_shell(args: dict) -> str:
    """Run a shell command. RESTRICTED to read-only/safe commands."""
    command = args.get("command", "")
    if not command:
        return "Error: 'command' required"

    # Safety: block destructive commands and interpreter escapes
    blocked = ["rm ", "rm\t", "rmdir", "dd ", "mkfs", "> /", ">> /",
               "chmod", "chown", "sudo", "su ", "kill", "pkill",
               "shutdown", "reboot", "systemctl", "mv ", "cp ",
               "bash ", "bash\t", "sh ", "sh\t", "zsh",
               "python", "perl", "ruby", "node ", "node\t",
               "tee ", "tee\t", "unlink", "truncate",
               "$(", "`"]
    cmd_lower = command.lower().strip()
    for b in blocked:
        if b in cmd_lower:
            return f"Error: command blocked for safety (contains '{b.strip()}')"

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=15, cwd=os.path.expanduser("~"),
        )
        output = result.stdout[:5000]
        if result.stderr:
            output += f"\nSTDERR: {result.stderr[:1000]}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (15s limit)"
    except Exception as e:
        return f"Error: {e}"


def _tool_web_search(args: dict) -> str:
    """Simple web search via DuckDuckGo Lite (no API key needed)."""
    query = args.get("query", "")
    if not query:
        return "Error: 'query' required"
    try:
        import urllib.request
        import urllib.parse
        url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Extract text snippets from HTML (rough)
        # DDG Lite returns simple HTML with result links
        results = []
        for match in re.findall(r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>', html):
            url, title = match
            title = re.sub(r'<[^>]+>', '', title).strip()
            if title and url.startswith("http"):
                results.append(f"- {title}\n  {url}")
        if not results:
            # Fallback: extract any links
            for match in re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', html):
                url, title = match
                title = re.sub(r'<[^>]+>', '', title).strip()
                if title and len(title) > 5:
                    results.append(f"- {title}\n  {url}")
        return f"Search results for '{query}':\n" + "\n".join(results[:10]) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def _tool_memory_search(args: dict) -> str:
    """Search Claude Code memory files."""
    query = args.get("query", "")
    if not query:
        return "Error: 'query' required"
    memory_dir = Path(os.path.expanduser(
        "~/.claude/projects/-home-voidstr3m33/memory"))
    if not memory_dir.exists():
        return "Error: memory directory not found"
    results = []
    for f in sorted(memory_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        try:
            content = f.read_text()
            if re.search(query, content, re.IGNORECASE):
                # Get first matching line
                for line in content.split("\n"):
                    if re.search(query, line, re.IGNORECASE):
                        results.append(f"  {f.stem}: {line.strip()[:120]}")
                        break
        except Exception:
            continue
    if not results:
        return f"No memory files match '{query}'"
    return f"Memory matches for '{query}' ({len(results)}):\n" + "\n".join(results[:15])


ESCALATION_MARKER = "__ESCALATION__"


def _tool_delegate_to_host(args: dict) -> str:
    """Request help from the host (Claude Code). Does NOT execute — creates a request.

    The orchestrator captures this and surfaces it for human/host review.
    """
    request_type = args.get("request_type", "general")
    goal = args.get("goal", "")
    files = args.get("files", [])
    proposed_change = args.get("proposed_change", "")
    reason = args.get("reason", "")
    priority = args.get("priority", "normal")

    if not goal:
        return "Error: 'goal' is required — describe what you need done"

    # Return a structured escalation packet with a marker the orchestrator detects
    packet = {
        ESCALATION_MARKER: True,
        "request_type": request_type,
        "goal": goal,
        "files": files if isinstance(files, list) else [files],
        "proposed_change": proposed_change,
        "reason": reason,
        "priority": priority,
    }
    return json.dumps(packet)


def create_default_registry() -> ToolRegistry:
    """Create a registry with built-in safe tools."""
    reg = ToolRegistry()

    reg.register("read_file", _tool_read_file,
                 "Read a file (under home directory, max 10KB).",
                 {"path": "File path to read"})

    reg.register("grep", _tool_grep,
                 "Search for a pattern in files (py/json/md/sh/txt). Returns matching file paths.",
                 {"pattern": "Search pattern (regex)", "path": "Directory to search (default: ~)"})

    reg.register("list_dir", _tool_list_dir,
                 "List contents of a directory.",
                 {"path": "Directory path (default: ~)"})

    reg.register("shell", _tool_shell,
                 "Run a read-only shell command. Destructive commands are blocked.",
                 {"command": "Shell command to run"})

    reg.register("web_search", _tool_web_search,
                 "Search the web via DuckDuckGo.",
                 {"query": "Search query"})

    reg.register("memory_search", _tool_memory_search,
                 "Search Claude Code memory files by keyword.",
                 {"query": "Search term (regex OK)"})

    reg.register("delegate_to_host", _tool_delegate_to_host,
                 "Request help from Claude Code / the host AI for tasks you cannot do: "
                 "file edits, complex refactors, code review, multi-file changes, "
                 "or anything needing stronger reasoning. This does NOT execute the action — "
                 "it creates a request for the host to review and act on.",
                 {"request_type": "edit|review|refactor|explain|investigate",
                  "goal": "What you want done (required)",
                  "files": "List of file paths involved",
                  "proposed_change": "Your suggested change (diff, description, or pseudocode)",
                  "reason": "Why you're escalating instead of doing it yourself",
                  "priority": "low|normal|high"})

    # Register agent-backed tools
    _register_agent_tools(reg)

    return reg


def _register_agent_tools(reg: ToolRegistry):
    """Register bounded agent tools into the tool registry.

    These let smaLLM call RAG, graph, SSM, sentinel, gate, receipt,
    and cartridge agents as tools during its tool loop.
    """
    from cell.agents.base import AgentRegistry
    from cell.agents.rag_agent import RAGLookupAgent, RAGSearchAgent
    from cell.agents.graph_agent import GraphLookupAgent, GraphNeighborsAgent, GraphStatsAgent
    from cell.agents.ssm_agent import SSMGetStateAgent, SSMUpdateEventAgent
    from cell.agents.sentinel_agent import SentinelTriageAgent
    from cell.agents.policy_agent import GateDecideAgent
    from cell.agents.receipt_agent import ReceiptLookupAgent, ReceiptWriteAgent
    from cell.agents.cartridge_agent import (
        CartridgeDispatchAgent, CodeRepairAgent, RuleGenerateAgent,
        PatchReviewAgent, ExploitAnalysisAgent, CartridgeListAgent,
    )
    from cell.agents.specialist_compute_agent import (
        SpecialistComputeRouteAgent, ShardListAgent, ShardResourceCheckAgent,
    )

    agent_reg = AgentRegistry()
    for cls in [RAGLookupAgent, RAGSearchAgent, GraphLookupAgent,
                GraphNeighborsAgent, GraphStatsAgent, SSMGetStateAgent,
                SSMUpdateEventAgent, SentinelTriageAgent, GateDecideAgent,
                ReceiptLookupAgent, ReceiptWriteAgent,
                CartridgeDispatchAgent, CodeRepairAgent, RuleGenerateAgent,
                PatchReviewAgent, ExploitAnalysisAgent, CartridgeListAgent,
                SpecialistComputeRouteAgent, ShardListAgent,
                ShardResourceCheckAgent]:
        agent_reg.register(cls())

    def _make_agent_handler(agent_name):
        def handler(args):
            result = agent_reg.run(agent_name, args)
            if result.error:
                return f"Agent error: {result.error}"
            output = result.output
            if result.receipt:
                output["_receipt"] = result.receipt
            return output
        return handler

    for info in agent_reg.list_agents():
        name = info["name"]
        # Build args_schema from input_schema properties
        props = info.get("input_schema", {}).get("properties", {})
        args_schema = {k: v.get("description", k) for k, v in props.items()}
        reg.register(
            name,
            _make_agent_handler(name),
            info["description"],
            args_schema,
        )

    # Store the agent registry on the tool registry for external access
    reg._agent_registry = agent_reg
