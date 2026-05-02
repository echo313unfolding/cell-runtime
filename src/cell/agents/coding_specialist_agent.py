"""Coding specialist agent — cold-load model for code analysis, repair, rule generation.

The coding specialist is a PROPOSER. It cannot execute shell, write files,
or escalate. It reads code, analyzes it, and returns structured proposals.

Sentinel or smaLLM can call the coding specialist when they need:
  - Script/code analysis
  - Tool/parser repair plans
  - YARA/Sigma rule generation
  - Patch review

The model is cold-loaded on demand and unloaded after idle timeout.
"""
import json
from cell.agents.base import AgentBase, AgentResult, Permission


class CodeAnalyzeAgent(AgentBase):
    name = "code_analyze"
    description = (
        "Analyze code or a script for issues, risks, patterns, and quality. "
        "Returns structured analysis. Does NOT execute the code."
    )
    permission = Permission.READ
    timeout_s = 120  # Cold-load may take time
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Code or script to analyze"},
            "context": {
                "type": "string",
                "description": "Why this code needs analysis (e.g., 'suspicious cron job', 'tool failure')",
            },
            "language": {
                "type": "string",
                "description": "Programming language (auto-detected if missing)",
            },
        },
        "required": ["code"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "risks": {"type": "array"},
            "model": {"type": "string"},
        },
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        code = args["code"]
        context = args.get("context", "")
        language = args.get("language", "")

        prompt = f"Analyze this code for issues, risks, and quality.\n"
        if context:
            prompt += f"Context: {context}\n"
        if language:
            prompt += f"Language: {language}\n"
        prompt += f"\n```\n{code[:4000]}\n```\n"
        prompt += "\nProvide: 1) Summary, 2) Risks/issues found, 3) Recommendations."

        return self._call_coder(prompt)

    def _call_coder(self, prompt: str) -> AgentResult:
        """Call the coding specialist model. Cold-loads if needed."""
        if not self._orchestrator:
            return AgentResult(error="No orchestrator available for coding specialist")

        result = self._orchestrator.process(
            prompt,
            force_model="qwen2.5-coder",
            use_tools=False,
        )

        if "error" in result:
            return AgentResult(error=f"Coding specialist error: {result['error']}")

        return AgentResult(output={
            "analysis": result.get("output", ""),
            "model": result.get("model", ""),
            "wall_time_s": result.get("wall_time_s", 0),
            "swapped": result.get("swapped", False),
            "receipt": result.get("receipt", ""),
        })


class CodeRepairPlanAgent(AgentBase):
    name = "code_repair_plan"
    description = (
        "Given an error log and file excerpt, propose a fix. "
        "Returns a repair plan (diff, pseudocode, or description). "
        "Does NOT apply the fix — that requires gatekeeper approval."
    )
    permission = Permission.READ
    timeout_s = 120
    input_schema = {
        "type": "object",
        "properties": {
            "error_log": {"type": "string", "description": "Error output or traceback"},
            "file_excerpt": {
                "type": "string",
                "description": "Relevant code that needs repair",
            },
            "file_path": {
                "type": "string",
                "description": "Path to the file (for context, not modification)",
            },
        },
        "required": ["error_log"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        error_log = args["error_log"]
        file_excerpt = args.get("file_excerpt", "")
        file_path = args.get("file_path", "")

        prompt = "Propose a fix for this error. Output ONLY the repair plan.\n\n"
        prompt += f"Error:\n```\n{error_log[:2000]}\n```\n"
        if file_excerpt:
            prompt += f"\nRelevant code"
            if file_path:
                prompt += f" ({file_path})"
            prompt += f":\n```\n{file_excerpt[:3000]}\n```\n"
        prompt += "\nProvide: 1) Root cause, 2) Proposed fix (diff or description), 3) Risk assessment."

        return self._call_coder(prompt)

    def _call_coder(self, prompt: str) -> AgentResult:
        if not self._orchestrator:
            return AgentResult(error="No orchestrator available for coding specialist")

        result = self._orchestrator.process(
            prompt, force_model="qwen2.5-coder", use_tools=False)

        if "error" in result:
            return AgentResult(error=f"Coding specialist error: {result['error']}")

        return AgentResult(output={
            "repair_plan": result.get("output", ""),
            "model": result.get("model", ""),
            "wall_time_s": result.get("wall_time_s", 0),
            "receipt": result.get("receipt", ""),
        })


class RuleGenerateAgent(AgentBase):
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
            "rule_type": {
                "type": "string",
                "description": "Rule format: yara, sigma, custom (default: yara)",
            },
            "policy_context": {
                "type": "string",
                "description": "Policy or threat context for the rule",
            },
        },
        "required": ["iocs"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        iocs = args["iocs"]
        rule_type = args.get("rule_type", "yara")
        policy = args.get("policy_context", "")

        prompt = f"Generate a {rule_type.upper()} detection rule for these IOCs.\n\n"
        prompt += f"IOCs:\n{iocs[:2000]}\n"
        if policy:
            prompt += f"\nPolicy context:\n{policy[:1000]}\n"
        prompt += f"\nOutput ONLY the {rule_type} rule. Include comments explaining each condition."

        if not self._orchestrator:
            return AgentResult(error="No orchestrator available for coding specialist")

        result = self._orchestrator.process(
            prompt, force_model="qwen2.5-coder", use_tools=False)

        if "error" in result:
            return AgentResult(error=f"Rule generation error: {result['error']}")

        return AgentResult(output={
            "rule": result.get("output", ""),
            "rule_type": rule_type,
            "model": result.get("model", ""),
            "receipt": result.get("receipt", ""),
        })


class PatchReviewAgent(AgentBase):
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
            "context": {
                "type": "string",
                "description": "What the patch is trying to fix/change",
            },
        },
        "required": ["diff"],
    }

    _orchestrator = None

    def execute(self, args: dict) -> AgentResult:
        diff = args["diff"]
        context = args.get("context", "")

        prompt = "Review this patch for correctness, security risks, and side effects.\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += f"Diff:\n```diff\n{diff[:4000]}\n```\n"
        prompt += "\nProvide: 1) Correctness assessment, 2) Security risks, 3) Side effects, 4) Approve/reject recommendation."

        if not self._orchestrator:
            return AgentResult(error="No orchestrator available for coding specialist")

        result = self._orchestrator.process(
            prompt, force_model="qwen2.5-coder", use_tools=False)

        if "error" in result:
            return AgentResult(error=f"Patch review error: {result['error']}")

        return AgentResult(output={
            "review": result.get("output", ""),
            "model": result.get("model", ""),
            "receipt": result.get("receipt", ""),
        })
