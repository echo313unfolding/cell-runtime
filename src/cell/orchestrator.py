#!/usr/bin/env python3
"""Cell orchestrator — end-to-end local assistant pipeline.

Takes user input → classifies intent → swaps to right model → generates → logs.

Usage:
    python3 orchestrator.py "write a sort function"
    python3 orchestrator.py --classify-only "check auth.log"
    python3 orchestrator.py --status
    python3 orchestrator.py --force-model smollm3 "write code"
    python3 orchestrator.py --json "query"
    echo "explain quicksort" | python3 orchestrator.py
"""
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

# Add tools/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from cell.router import classify, route_model
from cell.task_record import TaskRecord
from cell.model_pool import ModelPool
from cell.tool_registry import (
    ToolRegistry, create_default_registry, parse_tool_calls, ESCALATION_MARKER,
)
from cell.memory_lane import MemoryLane
from cell.specialists import create_specialist_registry, find_specialist
from cell.ask_pass import ask_pass
from cell.context_compiler import compile_context, summarize_turn

CONFIG_PATH = Path(__file__).parent / "config.json"
MAX_TOOL_TURNS = 5
CONTEXT_MARGIN = 128          # tokens reserved beyond num_predict
TOOL_RESULT_CAP = 1200        # max chars per tool result (down from 3000)
FORCE_COMMIT_MSG = (
    "You have gathered enough information. Answer the original question now "
    "from what you have. Do not call any more tools."
)

# Tools injected per intent when use_tools=True.
# Keeps prompt small — only what the model needs for this query type.
#
# tools_are_evidence: when True, the intent's tools ARE the evidence
# mechanism (e.g., almanac_evidence queries a pre-computed table).
# The tools_when_evidence=off guard must not suppress them even if the
# compiler found receipt keyword hits — those hits are noise for this
# intent. Read by _should_suppress_tools().
INTENT_TOOLS = {
    "coding": {"tools": ["read_file", "grep", "list_dir", "shell"]},
    "security_triage": {"tools": ["read_file", "grep", "list_dir", "shell",
                                   "sentinel_triage", "rag_lookup"]},
    "general": {"tools": ["read_file", "grep", "list_dir", "memory_search"]},
    "reasoning": {"tools": ["read_file", "memory_search", "rag_lookup"]},
    "hybrid_experiment": {"tools": ["read_file", "grep", "list_dir", "memory_search"]},
    "almanac": {"tools": ["almanac_normalize", "almanac_evidence",
                           "vault_review", "read_file", "memory_search"],
                "tools_are_evidence": True},
}


class Orchestrator:
    def __init__(self, config_path: str = None):
        path = Path(config_path) if config_path else CONFIG_PATH
        with open(path) as f:
            self.config = json.load(f)
        self.pool = ModelPool(self.config)
        self.roster = self.config.get("roster", {})
        self.swap_policy = self.config.get("swap_policy", "sticky")
        self.default_model = self.config.get("default_model", "smollm3")
        self.gen_opts = self.config.get("generation", {})

        # Context strategy. Env var wins over config so an A/B harness can
        # flip it per-process without touching config.json.
        #   CELL_CONTEXT_COMPILER=0  → legacy capsule + raw history + full registry
        #   CELL_CONTEXT_COMPILER=1  → CPU-side compiler (default)
        ctx_cfg = self.config.get("context_compiler", {})
        env_flag = os.environ.get("CELL_CONTEXT_COMPILER")
        if env_flag is not None:
            self.context_compiler_enabled = env_flag.strip().lower() not in ("0", "false", "off", "")
        else:
            self.context_compiler_enabled = bool(ctx_cfg.get("enabled", True))
        self.context_token_budget = int(
            os.environ.get("CELL_CONTEXT_BUDGET", ctx_cfg.get("token_budget", 1200)))

        # v6 A/B controls. Config sets the default; env var overrides for
        # harness A/B runs.
        #   tools_when_evidence: "off" suppresses the tool loop when the
        #       compiler found receipt evidence. v6 receipt proved this
        #       cuts wall time 42% with zero retrieval loss.
        #   CELL_HINT_IN_SYSTEM=1: put the "evidence was retrieved" hint
        #       in the system prompt (experimental; caused degeneration).
        self.tools_when_evidence = os.environ.get(
            "CELL_TOOLS_WHEN_EVIDENCE",
            ctx_cfg.get("tools_when_evidence", "on")).strip().lower()
        self.hint_in_system = os.environ.get(
            "CELL_HINT_IN_SYSTEM", "0").strip().lower() in ("1", "true", "on")
        self.coding_exempt = os.environ.get(
            "CELL_CODING_EXEMPT", "0").strip().lower() not in ("0", "false", "off")

        self.tool_registry = create_default_registry()

        # Wire agents to orchestrator if available
        if hasattr(self.tool_registry, '_agent_registry'):
            self.agent_registry = self.tool_registry._agent_registry
            # Sentinel needs registry + orchestrator for context pack assembly
            sentinel_agent = self.agent_registry.get("sentinel_triage")
            if sentinel_agent:
                sentinel_agent._agent_registry = self.agent_registry
                sentinel_agent._orchestrator = self
            # Cartridge and compute agents need orchestrator for backend dispatch
            for name in ["cartridge_dispatch", "code_repair", "rule_generate",
                         "patch_review", "exploit_analysis",
                         "specialist_compute_route"]:
                agent = self.agent_registry.get(name)
                if agent:
                    agent._orchestrator = self
        else:
            self.agent_registry = None

        # Memory lane
        mem_cfg = self.config.get("memory_lane", {})
        self.memory = MemoryLane(
            llm_enabled=mem_cfg.get("llm_enhanced", False),
        )
        # Wire LLM callback — uses whatever model is currently loaded
        self.memory._llm_call = self._memory_llm_call

        # Specialist adapters (e.g., Sentinel hybrid stack)
        self.specialists = create_specialist_registry()

        # Receipt excerpts the compiler read on the previous turn of this
        # session. Follow-up questions ("what were the tok/s?") rarely
        # keyword-match the receipt they're about; compile_context falls
        # back to these when the fresh search is empty or weak.
        # Each entry: {"path": str, "name": str, "excerpt": str}.
        # The stored excerpt text is reused verbatim on carry-forward so
        # the byte prefix of the evidence block is identical across turns,
        # enabling llama-server's prompt cache.
        self._prior_excerpts: list[dict] = []
        self._prior_intent: str = None

        # Resolve logging paths
        log_cfg = self.config.get("logging", {})
        self.swap_log_path = Path(os.path.expanduser(
            log_cfg.get("swap_log", "~/receipts/cell/swap_log.jsonl")))
        self.task_dir = Path(os.path.expanduser(
            log_cfg.get("task_dir", "~/receipts/cell/tasks/")))
        self.swap_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(parents=True, exist_ok=True)

    def reset_session(self):
        """Start a fresh conversation: clear memory lane and carried excerpts."""
        self.memory.reset()
        self._prior_excerpts = []
        self._prior_intent = None

    def _lean_registry(self, intent: str) -> ToolRegistry:
        """Build a lean tool registry containing only tools relevant to this intent."""
        entry = INTENT_TOOLS.get(intent, INTENT_TOOLS["general"])
        tool_names = entry["tools"] if isinstance(entry, dict) else entry
        lean = ToolRegistry()
        for name in tool_names:
            if name in self.tool_registry._tools:
                info = self.tool_registry._tools[name]
                lean.register(name, info["handler"], info["description"],
                              info["args_schema"])
        # Always include delegate_to_host as escape hatch
        if "delegate_to_host" in self.tool_registry._tools:
            info = self.tool_registry._tools["delegate_to_host"]
            lean.register("delegate_to_host", info["handler"],
                          info["description"], info["args_schema"])
        return lean

    def _should_swap(self, current: str | None, target: str, intent: str) -> bool:
        """Decide whether to swap models based on policy."""
        if self.swap_policy == "always_swap":
            return current != target
        if self.swap_policy == "force_model":
            return False
        # sticky: check if current model handles this intent
        if current and current in self.roster:
            current_intents = self.roster[current].get("intents", [])
            if intent in current_intents:
                return False
        return True

    def _memory_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """LLM callback for memory lane — uses currently loaded model."""
        current = self.pool.which_loaded()
        if not current:
            return ""
        backend = self._backend_for(current)
        if backend in ("helix", "llama-server"):
            base_url = (self.config.get("llama_server_url", "http://localhost:8080")
                        if backend == "llama-server"
                        else self.config.get("helix_url", "http://localhost:8001"))
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            result = self._openai_chat(base_url, messages, max_tokens=512, temperature=0.0)
            return result.get("content", "")
        return ""

    def _backend_for(self, model: str) -> str:
        return self.roster.get(model, {}).get("backend", "ollama")

    def _openai_chat(self, base_url: str, messages: list,
                     max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        """Call an OpenAI-compatible chat endpoint (helix or llama-server)."""
        payload = {
            "messages": [{"role": m["role"], "content": m["content"]}
                         for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self.pool._http(
            f"{base_url}/v1/chat/completions", payload, timeout=300)
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})
        # Two different prompt numbers, and they answer different questions:
        #   usage.prompt_tokens  — the FULL context the model saw (what counts
        #                          against n_ctx)
        #   timings.prompt_n     — tokens llama-server actually EVALUATED this
        #                          call, i.e. not served from its prompt cache.
        #                          A constant system prompt is cached after the
        #                          first call, so prompt_n drops to ~question
        #                          size on later calls.
        # Context occupancy must come from usage; speed from timings. (The
        # first A/B run conflated them and under-reported legacy context 5x.)
        timings = data.get("timings") or {}
        full_prompt = usage.get("prompt_tokens") or timings.get("prompt_n", 0)
        evaluated = timings.get("prompt_n", full_prompt)
        return {
            "content": content,
            "eval_count": timings.get("predicted_n",
                                      usage.get("completion_tokens", 0)),
            "prompt_tokens": full_prompt,
            "prompt_evaluated": evaluated,
            "prompt_cached": max(0, full_prompt - evaluated),
            "prompt_ms": timings.get("prompt_ms", 0.0),
            "predicted_ms": timings.get("predicted_ms", 0.0),
        }

    def _tokenize(self, text: str) -> int:
        """Count tokens via llama-server /tokenize. Falls back to len/2.5."""
        base_url = self.config.get("llama_server_url", "http://localhost:8080")
        try:
            data = self.pool._http(
                f"{base_url}/tokenize",
                {"content": text}, timeout=5)
            tokens = data.get("tokens", [])
            return len(tokens) if isinstance(tokens, list) else int(tokens)
        except Exception:
            return int(len(text) / 2.5)

    def _generate(self, model: str, user_input: str, use_tools: bool = False,
                  history: list = None, tool_registry: ToolRegistry = None,
                  system_hint: str = "", num_predict_override: int = None,
                  system_prompt_override: str = None,
                  raw_user_input: str = None) -> dict:
        """Generate via the appropriate backend. If use_tools, run a multi-turn tool loop.

        Args:
            history: Optional list of prior {"role": ..., "content": ...} messages
                     for multi-turn chat. Inserted between system prompt and current user turn.
            tool_registry: Optional override registry (for selective tool loading).
                           Defaults to self.tool_registry.
            system_hint: Extra text appended to the system prompt (e.g., evidence hint).
            num_predict_override: If set, overrides generation.num_predict for this call.
            system_prompt_override: If set, replaces the model's default system prompt.
            raw_user_input: The original user question before augmentation.
                            Passed as trusted execution context to agents that
                            need it (e.g., almanac_evidence query correction).
                            Never derived from the augmented prompt.
        """
        active_registry = tool_registry or self.tool_registry
        backend = self._backend_for(model)
        system_prompt = system_prompt_override or self.roster.get(model, {}).get(
            "system_prompt", "You are a helpful assistant.")
        if system_hint:
            system_prompt += system_hint
        if use_tools:
            system_prompt += active_registry.format_for_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        base_predict = self.gen_opts.get("num_predict", 1024)
        opts = {
            "temperature": self.gen_opts.get("temperature", 0.0),
            "num_ctx": self.gen_opts.get("num_ctx", 4096),
            "num_predict": num_predict_override or base_predict,
        }

        n_ctx = opts["num_ctx"]
        num_predict = opts["num_predict"]

        total_eval_count = 0
        total_eval_duration = 0
        total_prompt_tokens = 0
        total_prompt_ms = 0.0
        first_prompt_tokens = None   # FULL context of the first call (vs n_ctx)
        first_prompt_evaluated = None  # of which, tokens not served from cache
        max_prompt_tokens = 0        # largest context any call in the loop saw
        timings_measured = False
        tool_calls_log = []
        data = None
        context_ceiling_hit = False
        stagnation_hit = False
        seen_call_hashes = set()     # (tool_name, sorted_args) → stagnation
        seen_result_hashes = set()   # result hash → no-new-info

        def _result(content: str, **extra) -> dict:
            out = {
                "content": content,
                "eval_count": total_eval_count,
                "eval_duration": total_eval_duration,
                "prompt_eval_duration": (
                    int(total_prompt_ms * 1e6) if timings_measured
                    else (data.get("prompt_eval_duration", 0)
                          if backend == "ollama" and data else 0)),
                "total_duration": (
                    data.get("total_duration", 0)
                    if backend == "ollama" and data else 0),
                "prompt_tokens": total_prompt_tokens,
                "first_prompt_tokens": first_prompt_tokens or 0,
                "first_prompt_evaluated": first_prompt_evaluated or 0,
                "max_prompt_tokens": max_prompt_tokens,
                "prompt_ms": round(total_prompt_ms, 1),
                "timings_measured": timings_measured,
                "tool_calls": tool_calls_log,
                "context_ceiling_hit": context_ceiling_hit,
                "stagnation": stagnation_hit,
            }
            out.update(extra)
            return out

        def _count_tokens(text: str) -> int:
            """Token count via /tokenize, or conservative char estimate."""
            if backend == "llama-server":
                return self._tokenize(text)
            return int(len(text) / 2.5)

        def _record_call(hx_or_data):
            """Update running accumulators after an _openai_chat or ollama call."""
            nonlocal total_eval_count, total_eval_duration, total_prompt_tokens
            nonlocal total_prompt_ms, first_prompt_tokens, first_prompt_evaluated
            nonlocal max_prompt_tokens, timings_measured, data
            if backend in ("helix", "llama-server"):
                hx = hx_or_data
                total_eval_count += hx["eval_count"]
                total_prompt_tokens += hx.get("prompt_tokens", 0)
                max_prompt_tokens = max(max_prompt_tokens, hx.get("prompt_tokens", 0))
                if first_prompt_tokens is None:
                    first_prompt_tokens = hx.get("prompt_tokens", 0)
                    first_prompt_evaluated = hx.get("prompt_evaluated", first_prompt_tokens)
                if hx.get("predicted_ms"):
                    total_eval_duration += int(hx["predicted_ms"] * 1e6)
                    total_prompt_ms += hx.get("prompt_ms", 0.0)
                    timings_measured = True
                else:
                    rated_toks = self.roster.get(model, {}).get("tok_s", 10)
                    est_dur_ns = int(hx["eval_count"] / rated_toks * 1e9) if rated_toks else 0
                    total_eval_duration += est_dur_ns
            else:
                data = hx_or_data
                total_eval_count += data.get("eval_count", 0)
                total_eval_duration += data.get("eval_duration", 0)

        def _call_model(msgs, max_tok=None):
            """Single model call through appropriate backend. Returns content string."""
            if max_tok is None:
                max_tok = num_predict
            if backend in ("helix", "llama-server"):
                base_url = (self.config.get("llama_server_url", "http://localhost:8080")
                            if backend == "llama-server"
                            else self.config.get("helix_url", "http://localhost:8001"))
                hx = self._openai_chat(base_url, msgs, max_tokens=max_tok,
                                       temperature=opts["temperature"])
                _record_call(hx)
                return hx["content"]
            else:
                payload = {
                    "model": model,
                    "messages": msgs,
                    "stream": False,
                    "options": {**opts, "num_predict": max_tok},
                }
                d = self.pool._api("api/chat", payload, timeout=300)
                _record_call(d)
                return d.get("message", {}).get("content", "")

        def _force_commit(reason: str) -> dict:
            """MorphSAT-style force commit: one final answer call, no tools."""
            nonlocal context_ceiling_hit, stagnation_hit
            if "ceiling" in reason:
                context_ceiling_hit = True
            if "stagnation" in reason or "no_new_info" in reason:
                stagnation_hit = True
            # Estimate how many tokens we can still generate.
            # Subtract ~64 for FORCE_COMMIT_MSG + any pending assistant
            # content already in messages that max_prompt_tokens doesn't
            # reflect yet.
            remaining = n_ctx - max_prompt_tokens - CONTEXT_MARGIN - 64
            commit_budget = max(32, min(remaining, num_predict))
            # Force-commit message tells the model to answer, not tool-call
            messages.append({"role": "user", "content": FORCE_COMMIT_MSG})
            try:
                answer = _call_model(messages, max_tok=commit_budget)
            except Exception:
                answer = f"[context ceiling reached: {reason}]"
            return _result(answer, force_commit=reason)

        for turn in range(MAX_TOOL_TURNS if use_tools else 1):
            content = _call_model(messages)

            if not use_tools:
                return _result(content)

            # Check for tool calls
            calls = parse_tool_calls(content)
            if not calls:
                # No tool calls — model is done
                return _result(content)

            # Execute tool calls and feed results back
            messages.append({"role": "assistant", "content": content})
            escalations = []
            # pending_tokens: tokens appended to messages THIS turn that
            # haven't been reflected in max_prompt_tokens yet (which only
            # updates on the next _call_model). Includes the assistant
            # fence (content) plus each tool result we append below.
            # The assistant fence is roughly the last call's predicted_n.
            pending_tokens = _count_tokens(content)
            for tc in calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})

                # Stagnation: repeated (tool_name, args) pair
                call_sig = hashlib.md5(
                    json.dumps([tool_name, tool_args], sort_keys=True
                               ).encode()).hexdigest()[:8]
                if call_sig in seen_call_hashes:
                    return _force_commit("stagnation_repeated_call")
                seen_call_hashes.add(call_sig)

                # ask-pass gate for privileged actions
                if not ask_pass(tool_name, json.dumps(tool_args)):
                    tool_calls_log.append({
                        "name": tool_name,
                        "arguments": tool_args,
                        "result": "Action denied by user.",
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Tool denied ({tool_name}): user declined the action.",
                    })
                    continue

                # Sanitize and inject trusted execution context for
                # almanac_evidence. Always strip model-supplied context
                # fields — even when raw_user_input is absent (e.g.,
                # direct _generate() callers), forged fields must not
                # reach the agent. Only inject trusted when we have it.
                if tool_name == "almanac_evidence":
                    tool_args.pop("user_question", None)
                    tool_args.pop("_trusted_user_question", None)
                    if raw_user_input is not None:
                        tool_args["_trusted_user_question"] = raw_user_input

                tool_result = active_registry.execute(tool_name, tool_args)

                # No-new-info: identical result from different call.
                # Skip for short results — empty dirs, grep misses, etc.
                # all hash equal and would false-positive.
                # Also skip error strings — identical errors from different
                # calls mean the model is hitting a constraint, not that
                # there's no new information.  The call-hash still catches
                # a genuine loop (same tool + same args).
                result_lower = tool_result.lower()
                if (len(tool_result) >= 40
                        and not result_lower.startswith(
                            ("error", "permission", "timeout",
                             "unknown tool", "tool error", "agent error"))):
                    result_hash = hashlib.md5(
                        tool_result.encode()).hexdigest()[:8]
                    if result_hash in seen_result_hashes:
                        tool_calls_log.append({
                            "name": tool_name,
                            "arguments": tool_args,
                            "result": tool_result,
                            "result_sha256": hashlib.sha256(
                                tool_result.encode()).hexdigest(),
                            "result_bytes": len(tool_result.encode()),
                        })
                        return _force_commit("no_new_info_repeated_result")
                    seen_result_hashes.add(result_hash)

                # Receipt stores the complete result + SHA-256.
                # Preview metadata added after model preview is computed.
                full_hash = hashlib.sha256(
                    tool_result.encode()).hexdigest()
                log_entry = {
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                    "result_sha256": full_hash,
                    "result_bytes": len(tool_result.encode()),
                }
                tool_calls_log.append(log_entry)

                # Check if this is an escalation request
                if tool_name == "delegate_to_host":
                    try:
                        packet = json.loads(tool_result)
                        if packet.get(ESCALATION_MARKER):
                            escalations.append(packet)
                            messages.append({
                                "role": "user",
                                "content": "Escalation request recorded. The host will review it. "
                                           "Continue with any remaining work you can do locally.",
                            })
                            continue
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Token ceiling: will this result + next generation fit?
                # pending_tokens accounts for the assistant fence AND all
                # tool results already appended this turn.
                result_tokens = _count_tokens(tool_result[:TOOL_RESULT_CAP])
                est_next = (max_prompt_tokens + pending_tokens
                            + result_tokens + num_predict)
                if est_next > n_ctx - CONTEXT_MARGIN:
                    # Try to fit a truncated version
                    available = (n_ctx - CONTEXT_MARGIN - max_prompt_tokens
                                 - pending_tokens - num_predict)
                    max_chars = int(available * 2.5)
                    if max_chars < 200:
                        return _force_commit("ceiling_no_room")
                    tool_result = tool_result[:max_chars] + "\n[truncated to fit context]"
                    result_tokens = _count_tokens(tool_result)

                model_preview = tool_result[:TOOL_RESULT_CAP]
                msg_text = f"Tool result ({tool_name}):\n{model_preview}"
                messages.append({"role": "user", "content": msg_text})
                pending_tokens += _count_tokens(msg_text)

                # Record what the model actually saw vs the full result.
                preview_bytes = len(model_preview.encode())
                if preview_bytes < log_entry["result_bytes"]:
                    log_entry["model_preview_chars"] = len(model_preview)
                    log_entry["model_preview_sha256"] = hashlib.sha256(
                        model_preview.encode()).hexdigest()

            # If there were escalations, include them in the result
            if escalations:
                return _result(content, escalations=escalations)

        # Max turns reached
        return _result(content, max_turns_reached=True)

    def _log_swap(self, swap_result: dict, task_id: str, intent: str):
        """Append swap event to JSONL log."""
        if not swap_result.get("swapped"):
            return
        entry = {
            "timestamp": swap_result.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            "from": swap_result.get("from", ""),
            "to": swap_result.get("to", ""),
            "swap_time_s": swap_result.get("swap_time_s", 0),
            "task_id": task_id,
            "reason": f"intent={intent}, policy={self.swap_policy}",
        }
        with open(self.swap_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def process(self, user_input: str, force_model: str = None,
                use_tools: bool = False, history: list = None) -> dict:
        """End-to-end: classify → route → swap → generate (with optional tool loop) → log."""
        t_start = time.time()
        cpu_start = time.process_time()
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 1. Classify
        intent = classify(user_input)

        # 2. Route
        if force_model:
            model = force_model
        else:
            model = route_model(intent)

        # 3. Check current model and decide swap
        current = self.pool.which_loaded()
        need_swap = self._should_swap(current, model, intent)

        # If sticky and current model handles it, use current model instead
        if not need_swap and current and current != model:
            model = current

        # 4. Create task record
        task = TaskRecord.create(intent=intent, input_text=user_input, routed_model=model)
        task.route_reason = f"cell.v0: {intent} → {model} (policy={self.swap_policy})"
        task.start()

        # 5. Swap if needed
        swap_result = {"swapped": False, "swap_time_s": 0}
        if need_swap:
            swap_result = self.pool.swap_to(model)
            task.swap_cost_s = swap_result.get("swap_time_s", 0)
            task.swap_from = swap_result.get("from", "")
            if swap_result.get("error"):
                task.set_result(
                    model=model,
                    reasoning=f"Swap failed: {swap_result['error']}",
                )
                task.save(str(self.task_dir))
                return {"error": swap_result["error"], "task": task.to_dict()}

        # 6. Check for specialist adapter
        specialist = find_specialist(self.specialists, intent) if not force_model else None
        specialist_used = False

        if specialist:
            # Specialist path: delegate to domain-specific pipeline
            # The specialist handles its own SSM state, LLM calls, and gating.
            llama_port = int(self.config.get("llama_server_url",
                            "http://localhost:8080").rsplit(":", 1)[-1])
            try:
                spec_result = specialist.handle(user_input, port=llama_port)
                specialist_used = True

                # Feed specialist output to generic memory lane for cross-lane awareness
                self.memory.ingest({
                    "intent": intent,
                    "model": model,
                    "user_input": user_input,
                    "output": spec_result.output,
                })

                task.set_result(
                    model=model,
                    output_text=spec_result.output,
                    verdict=spec_result.verdict.get("severity", ""),
                    reasoning=f"Specialist: {spec_result.specialist}",
                )

                # Build result with specialist receipt
                wall_time = round(time.time() - t_start, 3)
                cpu_time = round(time.process_time() - cpu_start, 3)
                task_dict = task.to_dict()
                task_dict["cost"] = {
                    "wall_time_s": wall_time,
                    "cpu_time_s": cpu_time,
                    "peak_memory_mb": round(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                    "timestamp_start": start_iso,
                    "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                task_dict["specialist"] = spec_result.receipt

                task_path = self.task_dir / f"task_{task.task_id}.json"
                with open(task_path, "w") as f:
                    json.dump(task_dict, f, indent=2)

                self._log_swap(swap_result, task.task_id, intent)

                return {
                    "task_id": task.task_id,
                    "intent": intent,
                    "model": model,
                    "swapped": swap_result.get("swapped", False),
                    "swap_time_s": swap_result.get("swap_time_s", 0),
                    "output": spec_result.output,
                    "specialist_adapter_used": True,
                    "specialist": spec_result.specialist,
                    "gate_fired": spec_result.gate_fired,
                    "gate_rules": spec_result.gate_rules,
                    "verdict": spec_result.verdict,
                    "wall_time_s": wall_time,
                    "receipt": str(task_path),
                    "memory_turn": self.memory.state.get("turn_count", 0),
                }
            except ImportError:
                # Specialist not available — fall through to generic path
                pass

        # 7. Build the prompt context. Two modes, selected by
        #    CELL_CONTEXT_COMPILER (env, default on):
        #      "1"  compiler  — CPU-side evidence packet, compressed history,
        #                       lean per-intent tool registry (step 7/8 as of 9b67b64)
        #      "0"  legacy    — memory capsule appended to input, raw history
        #                       passed through, full tool registry (pre-9b67b64)
        #    The legacy path exists so A/B runs can hit the same orchestrator,
        #    binary and model with only the context strategy varying.
        context_mode = "compiler" if self.context_compiler_enabled else "legacy"
        compiled = {"evidence": "", "sources": [], "tokens_est": 0, "searches": []}
        gen_history = None

        if context_mode == "compiler":
            history_summary = ""
            if history:
                # Compress prior turns into a compact summary (chronological;
                # compile_context keeps the tail if it has to trim)
                for i in range(0, len(history) - 1, 2):
                    if (history[i].get("role") == "user" and
                            i + 1 < len(history) and
                            history[i + 1].get("role") == "assistant"):
                        turn_sum = summarize_turn(
                            history[i]["content"],
                            history[i + 1]["content"],
                        )
                        if turn_sum:
                            history_summary += turn_sum + "\n"

            compiled = compile_context(
                user_input, intent,
                history_summary=history_summary,
                token_budget=self.context_token_budget,
                prior_excerpts=self._prior_excerpts if history else None,
            )
            self._prior_excerpts = compiled.get("excerpts", [])
            saved_prior_intent = self._prior_intent  # capture before overwrite
            self._prior_intent = intent

            # v6: suppress tool loop when receipt evidence is present
            # (CELL_TOOLS_WHEN_EVIDENCE=off).
            #
            # evidence_present means the compiler found actual receipt
            # excerpts, not just memory or FGIP hits. Memory/FGIP add
            # "memory" and "fgip.db" to sources but those aren't answers
            # to the question — only receipt excerpts are.
            #
            # Coding exempt: "fix this file" needs read_file/grep/shell
            # even when a receipt filename coincidentally matches a word
            # in the question ("fixture" → wp_runtime_001_fixture.json).
            # The classifier's code-keyword regexes are tightened (0007)
            # so receipt questions no longer false-positive as coding.
            receipt_sources = [s for s in compiled["sources"]
                               if s not in ("memory", "fgip.db")]
            evidence_present = bool(receipt_sources)
            # tools_are_evidence: intent's own tools ARE the evidence
            # mechanism (declared in INTENT_TOOLS metadata). Suppressing
            # them on a noise receipt hit would block the only path to
            # real data. coding_exempt is the legacy flag for the same
            # concept before the metadata was added.
            entry = INTENT_TOOLS.get(intent, {})
            tools_are_evidence = (
                (isinstance(entry, dict) and entry.get("tools_are_evidence"))
                or (self.coding_exempt and intent == "coding")
            )
            if (use_tools and evidence_present
                    and self.tools_when_evidence == "off"
                    and not tools_are_evidence):
                use_tools = False

            # Build augmented input: evidence FIRST, then question.
            # Evidence goes at the front so the carried excerpts form a
            # byte-stable prefix across T2/T3, letting llama-server's
            # prompt cache cover them.
            augmented_input = user_input
            hint = ""
            if compiled["evidence"]:
                if use_tools and evidence_present and not self.hint_in_system:
                    hint = ("\nEvidence above was retrieved for you; "
                            "call tools only if it is insufficient.\n")
                augmented_input = (
                    f"--- Evidence (retrieved from local files) ---\n"
                    f"{compiled['evidence']}\n"
                    f"--- End evidence ---\n{hint}\n"
                    f"{user_input}"
                )
            active_reg = self._lean_registry(intent) if use_tools else None
            # Raw history is NOT passed — it's compressed into the evidence block
        else:
            evidence_present = False
            saved_prior_intent = self._prior_intent
            self._prior_intent = intent
            capsule_ctx = self.memory.get_capsule_prompt()
            augmented_input = user_input + capsule_ctx if capsule_ctx else user_input
            active_reg = None            # _generate falls back to the full registry
            gen_history = history

        # 8. Generate (with optional tool loop)
        # When hint_in_system is on, the hint goes into the system prompt
        # instead of the user turn (v6 arm: measure whether system-prompt
        # placement reduces unnecessary tool calls).
        sys_hint = ""
        if self.hint_in_system and evidence_present and context_mode == "compiler":
            sys_hint = ("\nWhen evidence is provided in the user message, "
                        "use it directly. Only call tools if the evidence "
                        "is clearly insufficient.")

        # Intent-specific num_predict: coding needs more tokens for functions.
        # Carry coding intent forward on follow-ups so T2/T3 get the budget.
        predict_by_intent = self.gen_opts.get("num_predict_by_intent", {})
        effective_intent = intent
        if intent not in predict_by_intent and history and saved_prior_intent in predict_by_intent:
            effective_intent = saved_prior_intent
        num_predict_override = predict_by_intent.get(effective_intent)

        # Intent-specific system prompt (e.g., almanac addresses member as "you").
        prompt_by_intent = self.gen_opts.get("system_prompt_by_intent", {})
        sys_prompt_override = prompt_by_intent.get(intent)

        gen_result = self._generate(model, augmented_input, use_tools=use_tools,
                                    history=gen_history, tool_registry=active_reg,
                                    system_hint=sys_hint,
                                    num_predict_override=num_predict_override,
                                    system_prompt_override=sys_prompt_override,
                                    raw_user_input=user_input)

        # 9. Feed result to memory lane
        self.memory.ingest({
            "intent": intent,
            "model": model,
            "user_input": user_input,
            "output": gen_result.get("content", ""),
            "tool_calls": gen_result.get("tool_calls", []),
        })

        # 10. Record result
        eval_ns = gen_result.get("eval_duration", 0)
        eval_count = gen_result.get("eval_count", 0)
        tok_s = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0

        task.set_result(
            model=model,
            output_text=gen_result["content"],
            reasoning=f"Generated {eval_count} tokens at {tok_s:.1f} tok/s",
        )

        # 11. Save task with cost block
        wall_time = round(time.time() - t_start, 3)
        cpu_time = round(time.process_time() - cpu_start, 3)
        task_dict = task.to_dict()
        task_dict["cost"] = {
            "wall_time_s": wall_time,
            "cpu_time_s": cpu_time,
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tool_calls = gen_result.get("tool_calls", [])
        escalations = gen_result.get("escalations", [])
        task_dict["generation"] = {
            "eval_tokens": eval_count,
            "tok_s": round(tok_s, 2),
            "ttft_ns": gen_result.get("prompt_eval_duration", 0),
            "total_duration_ns": gen_result.get("total_duration", 0),
            # Measured by llama-server when timings_measured is True;
            # otherwise estimated from the roster tok_s.
            "prompt_tokens": gen_result.get("prompt_tokens", 0),
            "first_prompt_tokens": gen_result.get("first_prompt_tokens", 0),
            "first_prompt_evaluated": gen_result.get("first_prompt_evaluated", 0),
            "max_prompt_tokens": gen_result.get("max_prompt_tokens", 0),
            "prompt_ms": gen_result.get("prompt_ms", 0.0),
            "timings_measured": gen_result.get("timings_measured", False),
            "context_ceiling_hit": gen_result.get("context_ceiling_hit", False),
            "stagnation": gen_result.get("stagnation", False),
            "force_commit": gen_result.get("force_commit"),
        }
        task_dict["specialist_adapter_used"] = False
        task_dict["context_mode"] = context_mode
        task_dict["context_compiled"] = {
            "tokens_est": compiled["tokens_est"],
            "sources": compiled["sources"],
            "searches": compiled["searches"],
        }
        if tool_calls:
            task_dict["tool_calls"] = tool_calls
        if escalations:
            task_dict["escalations"] = escalations

        task_path = self.task_dir / f"task_{task.task_id}.json"
        with open(task_path, "w") as f:
            json.dump(task_dict, f, indent=2)

        # 12. Log swap
        self._log_swap(swap_result, task.task_id, intent)

        result = {
            "task_id": task.task_id,
            "intent": intent,
            "model": model,
            "swapped": swap_result.get("swapped", False),
            "swap_time_s": swap_result.get("swap_time_s", 0),
            "output": gen_result["content"],
            "specialist_adapter_used": False,
            "eval_count": eval_count,
            "tok_s": round(tok_s, 2),
            "wall_time_s": wall_time,
            "tool_calls": len(tool_calls),
            "receipt": str(task_path),
            "memory_turn": self.memory.state.get("turn_count", 0),
            "context_mode": context_mode,
            "prompt_tokens": gen_result.get("prompt_tokens", 0),
            "first_prompt_tokens": gen_result.get("first_prompt_tokens", 0),
            "first_prompt_evaluated": gen_result.get("first_prompt_evaluated", 0),
            "max_prompt_tokens": gen_result.get("max_prompt_tokens", 0),
            "prompt_ms": gen_result.get("prompt_ms", 0.0),
            "timings_measured": gen_result.get("timings_measured", False),
            "context_compiled": {
                "tokens_est": compiled["tokens_est"],
                "sources": compiled["sources"],
                "searches": compiled["searches"],
            },
            "context_ceiling_hit": gen_result.get("context_ceiling_hit", False),
            "stagnation": gen_result.get("stagnation", False),
            "force_commit": gen_result.get("force_commit"),
        }
        if escalations:
            result["escalations"] = escalations
        return result

    def status(self) -> dict:
        """Current state: loaded model, swap history, roster."""
        current = self.pool.which_loaded()
        backends = {k: v.get("backend", "ollama") for k, v in self.roster.items()}
        return {
            "loaded_model": current or "(none)",
            "swap_policy": self.swap_policy,
            "roster": list(self.roster.keys()),
            "backends": backends,
            "helix_alive": self.pool._helix_alive(),
            "llama_server_alive": self.pool._llama_server_alive(),
            "swap_history": self.pool.get_swap_history(),
            "swap_count": len(self.pool.get_swap_history()),
            "memory": self.memory.to_dict(),
            "specialists": [s.state_dict() for s in self.specialists],
        }

    def classify_only(self, user_input: str) -> dict:
        """Classify intent and show routing decision without generating."""
        intent = classify(user_input)
        model = route_model(intent)
        current = self.pool.which_loaded()
        need_swap = self._should_swap(current, model, intent)
        return {
            "intent": intent,
            "routed_model": model,
            "current_model": current or "(none)",
            "would_swap": need_swap,
            "swap_policy": self.swap_policy,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="cell",
        description="Cell orchestrator — 3-model local assistant",
    )
    parser.add_argument("text", nargs="*", help="Input text")
    parser.add_argument("--classify-only", "-c", action="store_true",
                        help="Classify only, don't generate")
    parser.add_argument("--status", "-s", action="store_true",
                        help="Show current state")
    parser.add_argument("--force-model", "-f", type=str, default=None,
                        help="Override routing, use this model")
    parser.add_argument("--json", "-j", action="store_true",
                        help="JSON output")
    parser.add_argument("--tools", "-t", action="store_true",
                        help="Enable tool use (model can call tools)")
    parser.add_argument("--list-tools", action="store_true",
                        help="List available tools")
    parser.add_argument("--download", "-d", type=str, default=None,
                        help="Download model(s) from HF (name or 'all')")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json")
    args = parser.parse_args()

    # Handle download before creating Orchestrator (may not have models yet)
    if args.download:
        from cell.download_models import download_model, MODELS
        targets = list(MODELS.keys()) if args.download == "all" else [args.download]
        models_dir = os.environ.get("CELL_MODELS_DIR", "/models")
        ok = sum(1 for t in targets if download_model(t, models_dir))
        print(f"\n{ok}/{len(targets)} models ready.")
        return

    orch = Orchestrator(config_path=args.config)

    if args.list_tools:
        for t in orch.tool_registry.list_tools():
            print(f"  {t['name']}: {t['description']}")
        return

    if args.status:
        result = orch.status()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"  Loaded: {result['loaded_model']}")
            print(f"  Policy: {result['swap_policy']}")
            print(f"  Helix:  {'UP' if result['helix_alive'] else 'DOWN'}")
            for name in result['roster']:
                be = result['backends'].get(name, 'ollama')
                print(f"  → {name} [{be}]")
            print(f"  Swaps this session: {result['swap_count']}")
        return

    # Gather input
    if args.text:
        user_input = " ".join(args.text)
    elif not sys.stdin.isatty():
        user_input = sys.stdin.read().strip()
    else:
        print("Usage: cell <text>", file=sys.stderr)
        print("       echo 'text' | cell", file=sys.stderr)
        sys.exit(1)

    if not user_input:
        print("Error: empty input", file=sys.stderr)
        sys.exit(1)

    if args.classify_only:
        result = orch.classify_only(user_input)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"  Intent: {result['intent']}")
            print(f"  Route:  {result['routed_model']}")
            print(f"  Current: {result['current_model']}")
            print(f"  Would swap: {result['would_swap']}")
        return

    # Full pipeline
    result = orch.process(user_input, force_model=args.force_model,
                          use_tools=args.tools)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        swap_info = ""
        if result["swapped"]:
            swap_info = f" [swapped in {result['swap_time_s']:.1f}s]"
        print(f"  [{result['intent']}] → {result['model']}{swap_info}")
        print(f"  {result['tok_s']} tok/s, {result['wall_time_s']:.1f}s total")
        print(f"  Receipt: {result['receipt']}")
        print()
        print(result["output"])

        # Display escalation requests
        if result.get("escalations"):
            print()
            print("=" * 60)
            print(f"  ESCALATION REQUESTS ({len(result['escalations'])})")
            print("=" * 60)
            for i, esc in enumerate(result["escalations"], 1):
                print(f"\n  [{i}] {esc.get('request_type', 'general').upper()}: {esc.get('goal', '(no goal)')}")
                if esc.get("files"):
                    print(f"      Files: {', '.join(esc['files'])}")
                if esc.get("proposed_change"):
                    print(f"      Proposed: {esc['proposed_change'][:200]}")
                if esc.get("reason"):
                    print(f"      Reason: {esc['reason']}")
                print(f"      Priority: {esc.get('priority', 'normal')}")


if __name__ == "__main__":
    main()
