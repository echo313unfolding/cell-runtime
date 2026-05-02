"""Cartridge agents — bounded tools that dispatch to skill cartridges.

Each agent maps to a specific set of cartridge intents. They route to the
CartridgePool, which loads the needed capability package (LoRA, grammar,
prompt pack, RAG index, etc.) and runs through the base model backend.

Cartridges PROPOSE. They do NOT execute. No shell, no file_write, no sudo.
Gatekeeper authorizes any action after cartridge proposes.
"""
from cell.agents.base import AgentBase, AgentResult, Permission
from cell.cartridge_pool import CartridgePool


# Shared pool — initialized once, used by all cartridge agents
_pool: CartridgePool | None = None


def get_cartridge_pool(cartridge_dir: str = None) -> CartridgePool:
    """Get or create the shared cartridge pool."""
    global _pool
    if _pool is None:
        import os
        if cartridge_dir is None:
            # Default: cell-runtime/cartridges/
            # __file__ is src/cell/agents/cartridge_agent.py → 4 levels up to cell-runtime/
            cartridge_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                "cartridges",
            )
        _pool = CartridgePool(cartridge_dir)
    return _pool


def reset_pool():
    """Reset the shared pool (for testing)."""
    global _pool
    _pool = None


class CartridgeDispatchAgent(AgentBase):
    """Generic cartridge dispatcher — routes any intent to the matching cartridge."""
    name = "cartridge_dispatch"
    description = (
        "Route a task to the appropriate skill cartridge by intent. "
        "Returns the cartridge's structured proposal. Does NOT execute anything."
    )
    permission = Permission.READ
    timeout_s = 120
    input_schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "Task intent (e.g., parser_repair, yara_rule, patch_review)",
            },
            "task": {
                "type": "string",
                "description": "Task description or payload",
            },
        },
        "required": ["intent", "task"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "cartridge_id": {"type": "string"},
            "output": {"type": "string"},
            "artifacts_loaded": {"type": "array"},
        },
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        intent = args["intent"]
        task = args["task"]
        pool = get_cartridge_pool()

        result = pool.dispatch(intent, task, orchestrator=self._orchestrator)

        if "error" in result:
            return AgentResult(error=result["error"])

        return AgentResult(output=result)


class CodeRepairAgent(AgentBase):
    """Code/parser repair — dispatches to code_parser_repair cartridge."""
    name = "code_repair"
    description = (
        "Given an error log and code excerpt, propose a fix via the code repair "
        "cartridge. Returns root cause, proposed fix, and risk assessment. "
        "Does NOT apply fixes."
    )
    permission = Permission.READ
    timeout_s = 120
    input_schema = {
        "type": "object",
        "properties": {
            "error_log": {"type": "string", "description": "Error output or traceback"},
            "code": {"type": "string", "description": "Relevant code that needs repair"},
            "file_path": {"type": "string", "description": "Path to the file (for context)"},
        },
        "required": ["error_log"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        error_log = args["error_log"]
        code = args.get("code", "")
        file_path = args.get("file_path", "")

        task = f"Error:\n```\n{error_log[:2000]}\n```\n"
        if code:
            task += f"\nRelevant code"
            if file_path:
                task += f" ({file_path})"
            task += f":\n```\n{code[:3000]}\n```\n"
        task += "\nProvide: 1) Root cause, 2) Proposed fix (diff or description), 3) Risk assessment."

        pool = get_cartridge_pool()
        result = pool.dispatch("code_repair", task, orchestrator=self._orchestrator)

        if "error" in result:
            return AgentResult(error=result["error"])

        return AgentResult(output={
            "repair_plan": result.get("output", result.get("assembled_prompt", "")),
            "cartridge_id": result.get("cartridge_id", ""),
            "artifacts_loaded": result.get("artifacts_loaded", []),
        })


class RuleGenerateAgent(AgentBase):
    """Detection rule generation — dispatches to rule_generation cartridge."""
    name = "rule_generate"
    description = (
        "Generate a YARA, Sigma, or custom detection rule from IOCs and policy. "
        "Returns the rule text as a proposal. Does NOT deploy the rule."
    )
    permission = Permission.READ
    timeout_s = 120
    input_schema = {
        "type": "object",
        "properties": {
            "iocs": {"type": "string", "description": "Indicators of compromise (IPs, hashes, patterns)"},
            "rule_type": {"type": "string", "description": "Rule format: yara, sigma, custom (default: yara)"},
            "policy_context": {"type": "string", "description": "Policy or threat context for the rule"},
        },
        "required": ["iocs"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        iocs = args["iocs"]
        rule_type = args.get("rule_type", "yara")
        policy = args.get("policy_context", "")

        task = f"Generate a {rule_type.upper()} detection rule for these IOCs.\n\n"
        task += f"IOCs:\n{iocs[:2000]}\n"
        if policy:
            task += f"\nPolicy context:\n{policy[:1000]}\n"
        task += f"\nOutput ONLY the {rule_type} rule. Include comments explaining each condition."

        # Map rule_type to cartridge intent
        intent = "yara_rule" if rule_type == "yara" else "sigma_rule" if rule_type == "sigma" else "rule_generate"

        pool = get_cartridge_pool()
        result = pool.dispatch(intent, task, orchestrator=self._orchestrator)

        if "error" in result:
            return AgentResult(error=result["error"])

        return AgentResult(output={
            "rule": result.get("output", result.get("assembled_prompt", "")),
            "rule_type": rule_type,
            "cartridge_id": result.get("cartridge_id", ""),
            "artifacts_loaded": result.get("artifacts_loaded", []),
        })


class PatchReviewAgent(AgentBase):
    """Patch/diff review — dispatches to patch_review cartridge."""
    name = "patch_review"
    description = (
        "Review a code diff/patch for correctness, security risks, and side effects. "
        "Returns structured review. Does NOT apply or reject the patch."
    )
    permission = Permission.READ
    timeout_s = 120
    input_schema = {
        "type": "object",
        "properties": {
            "diff": {"type": "string", "description": "The diff/patch to review"},
            "context": {"type": "string", "description": "What the patch is trying to fix/change"},
        },
        "required": ["diff"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        diff = args["diff"]
        context = args.get("context", "")

        task = "Review this patch for correctness, security risks, and side effects.\n\n"
        if context:
            task += f"Context: {context}\n\n"
        task += f"Diff:\n```diff\n{diff[:4000]}\n```\n"
        task += "\nProvide: 1) Correctness assessment, 2) Security risks, 3) Side effects, 4) Approve/reject recommendation."

        pool = get_cartridge_pool()
        result = pool.dispatch("patch_review", task, orchestrator=self._orchestrator)

        if "error" in result:
            return AgentResult(error=result["error"])

        return AgentResult(output={
            "review": result.get("output", result.get("assembled_prompt", "")),
            "cartridge_id": result.get("cartridge_id", ""),
            "artifacts_loaded": result.get("artifacts_loaded", []),
        })


class ExploitAnalysisAgent(AgentBase):
    """Exploit/vulnerability analysis — dispatches to exploit_analysis cartridge."""
    name = "exploit_analysis"
    description = (
        "Analyze code, logs, or indicators for exploit chains and attack vectors. "
        "Returns defensive analysis. Does NOT execute exploits."
    )
    permission = Permission.READ
    timeout_s = 120
    input_schema = {
        "type": "object",
        "properties": {
            "artifact": {"type": "string", "description": "Code, log, or indicator to analyze"},
            "artifact_type": {"type": "string", "description": "Type: code, log, ioc, binary_hash"},
            "context": {"type": "string", "description": "Threat context or alert details"},
        },
        "required": ["artifact"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        artifact = args["artifact"]
        artifact_type = args.get("artifact_type", "code")
        context = args.get("context", "")

        task = f"Analyze this {artifact_type} for exploit chains and attack vectors.\n\n"
        if context:
            task += f"Context: {context}\n\n"
        task += f"Artifact ({artifact_type}):\n```\n{artifact[:4000]}\n```\n"
        task += "\nProvide: 1) Attack vector, 2) Exploit chain (if applicable), 3) Impact assessment, 4) Mitigation recommendations."

        pool = get_cartridge_pool()
        result = pool.dispatch("exploit_analysis", task, orchestrator=self._orchestrator)

        if "error" in result:
            return AgentResult(error=result["error"])

        return AgentResult(output={
            "analysis": result.get("output", result.get("assembled_prompt", "")),
            "cartridge_id": result.get("cartridge_id", ""),
            "artifacts_loaded": result.get("artifacts_loaded", []),
        })


class CartridgeListAgent(AgentBase):
    """List available cartridges and their status."""
    name = "cartridge_list"
    description = "List all registered skill cartridges with status and intent mappings."
    permission = Permission.READ
    timeout_s = 10
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> AgentResult:
        pool = get_cartridge_pool()
        return AgentResult(output={
            "cartridges": pool.list_cartridges(),
            "intent_map": pool.intent_map(),
            "count": len(pool),
        })
