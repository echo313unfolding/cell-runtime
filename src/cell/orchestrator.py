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

CONFIG_PATH = Path(__file__).parent / "config.json"
MAX_TOOL_TURNS = 5


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

        self.tool_registry = create_default_registry()

        # Memory lane
        mem_cfg = self.config.get("memory_lane", {})
        self.memory = MemoryLane(
            llm_enabled=mem_cfg.get("llm_enhanced", False),
        )
        # Wire LLM callback — uses whatever model is currently loaded
        self.memory._llm_call = self._memory_llm_call

        # Resolve logging paths
        log_cfg = self.config.get("logging", {})
        self.swap_log_path = Path(os.path.expanduser(
            log_cfg.get("swap_log", "~/receipts/cell/swap_log.jsonl")))
        self.task_dir = Path(os.path.expanduser(
            log_cfg.get("task_dir", "~/receipts/cell/tasks/")))
        self.swap_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(parents=True, exist_ok=True)

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
        return {
            "content": content,
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
        }

    def _generate(self, model: str, user_input: str, use_tools: bool = False) -> dict:
        """Generate via the appropriate backend. If use_tools, run a multi-turn tool loop."""
        backend = self._backend_for(model)
        system_prompt = self.roster.get(model, {}).get(
            "system_prompt", "You are a helpful assistant.")
        if use_tools:
            system_prompt += self.tool_registry.format_for_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        opts = {
            "temperature": self.gen_opts.get("temperature", 0.0),
            "num_ctx": self.gen_opts.get("num_ctx", 4096),
            "num_predict": self.gen_opts.get("num_predict", 1024),
        }

        total_eval_count = 0
        total_eval_duration = 0
        tool_calls_log = []
        data = None

        for turn in range(MAX_TOOL_TURNS if use_tools else 1):
            if backend in ("helix", "llama-server"):
                base_url = (self.config.get("llama_server_url", "http://localhost:8080")
                            if backend == "llama-server"
                            else self.config.get("helix_url", "http://localhost:8001"))
                hx = self._openai_chat(base_url,
                                       messages, max_tokens=opts["num_predict"],
                                       temperature=opts["temperature"])
                content = hx["content"]
                total_eval_count += hx["eval_count"]
                rated_toks = self.roster.get(model, {}).get("tok_s", 10)
                est_dur_ns = int(hx["eval_count"] / rated_toks * 1e9) if rated_toks else 0
                total_eval_duration += est_dur_ns
            else:
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": opts,
                }
                data = self.pool._api("api/chat", payload, timeout=300)
                msg = data.get("message", {})
                content = msg.get("content", "")
                total_eval_count += data.get("eval_count", 0)
                total_eval_duration += data.get("eval_duration", 0)

            if not use_tools:
                return {
                    "content": content,
                    "eval_count": total_eval_count,
                    "eval_duration": total_eval_duration,
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0) if backend == "ollama" and data else 0,
                    "total_duration": data.get("total_duration", 0) if backend == "ollama" and data else 0,
                    "tool_calls": [],
                }

            # Check for tool calls
            calls = parse_tool_calls(content)
            if not calls:
                # No tool calls — model is done
                return {
                    "content": content,
                    "eval_count": total_eval_count,
                    "eval_duration": total_eval_duration,
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0) if backend == "ollama" and data else 0,
                    "total_duration": data.get("total_duration", 0) if backend == "ollama" and data else 0,
                    "tool_calls": tool_calls_log,
                }

            # Execute tool calls and feed results back
            messages.append({"role": "assistant", "content": content})
            escalations = []
            for tc in calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_result = self.tool_registry.execute(tool_name, tool_args)
                tool_calls_log.append({
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": tool_result[:2000],
                })

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

                messages.append({
                    "role": "user",
                    "content": f"Tool result ({tool_name}):\n{tool_result[:3000]}",
                })

            # If there were escalations, include them in the result
            if escalations:
                return {
                    "content": content,
                    "eval_count": total_eval_count,
                    "eval_duration": total_eval_duration,
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0) if backend == "ollama" and data else 0,
                    "total_duration": data.get("total_duration", 0) if backend == "ollama" and data else 0,
                    "tool_calls": tool_calls_log,
                    "escalations": escalations,
                }

        # Max turns reached
        return {
            "content": content,
            "eval_count": total_eval_count,
            "eval_duration": total_eval_duration,
            "prompt_eval_duration": 0,
            "total_duration": 0,
            "tool_calls": tool_calls_log,
        }

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
                use_tools: bool = False) -> dict:
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

        # 6. Inject memory capsule into input if available
        capsule_ctx = self.memory.get_capsule_prompt()
        augmented_input = user_input
        if capsule_ctx:
            augmented_input = user_input + capsule_ctx

        # 7. Generate (with optional tool loop)
        gen_result = self._generate(model, augmented_input, use_tools=use_tools)

        # 8. Feed result to memory lane
        self.memory.ingest({
            "intent": intent,
            "model": model,
            "user_input": user_input,
            "output": gen_result.get("content", ""),
            "tool_calls": gen_result.get("tool_calls", []),
        })

        # 9. Record result
        eval_ns = gen_result.get("eval_duration", 0)
        eval_count = gen_result.get("eval_count", 0)
        tok_s = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0

        task.set_result(
            model=model,
            output_text=gen_result["content"],
            reasoning=f"Generated {eval_count} tokens at {tok_s:.1f} tok/s",
        )

        # 10. Save task with cost block
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
        }
        if tool_calls:
            task_dict["tool_calls"] = tool_calls
        if escalations:
            task_dict["escalations"] = escalations

        task_path = self.task_dir / f"task_{task.task_id}.json"
        with open(task_path, "w") as f:
            json.dump(task_dict, f, indent=2)

        # 11. Log swap
        self._log_swap(swap_result, task.task_id, intent)

        result = {
            "task_id": task.task_id,
            "intent": intent,
            "model": model,
            "swapped": swap_result.get("swapped", False),
            "swap_time_s": swap_result.get("swap_time_s", 0),
            "output": gen_result["content"],
            "eval_count": eval_count,
            "tok_s": round(tok_s, 2),
            "wall_time_s": wall_time,
            "tool_calls": len(tool_calls),
            "receipt": str(task_path),
            "memory_turn": self.memory.state.get("turn_count", 0),
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
