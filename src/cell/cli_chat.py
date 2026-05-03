#!/usr/bin/env python3
"""Interactive multi-turn chat REPL for the cell orchestrator.

Usage:
    python3 -m cell.cli_chat
    ech0 chat

Keeps session message history, feeds MemoryLane each turn,
writes per-turn receipts. Ctrl-C or /exit to quit.
"""

import json
import os
import readline  # enables arrow keys, history in input()
import signal
import sys
import time
from pathlib import Path

from cell.orchestrator import Orchestrator

# Max turns to keep in context (sliding window)
MAX_HISTORY_TURNS = 20


def _slash_help():
    return """
  /help      Show this help
  /status    Show loaded model, roster, backends
  /memory    Show MemoryLane state
  /clear     Clear session history (start fresh)
  /model X   Force model for remaining turns
  /tools     Toggle tool use on/off
  /history   Show conversation so far
  /exit      Exit chat
  /quit      Exit chat
""".strip()


CAPSULE_CONFIG = os.path.expanduser("~/tools/capsule/config.json")


def run_chat(config_path: str = None, force_model: str = None, use_tools: bool = False):
    """Main REPL loop."""
    if not config_path and os.path.exists(CAPSULE_CONFIG):
        config_path = CAPSULE_CONFIG
    orch = Orchestrator(config_path=config_path)

    # Session state
    history = []  # list of {"role": "user"|"assistant", "content": ...}
    session_model = force_model
    session_tools = use_tools
    turn_count = 0

    # Show banner
    current = orch.pool.which_loaded()
    print("ech0 chat — interactive mode")
    print(f"  Loaded: {current or '(none)'}")
    print(f"  Tools:  {'ON' if session_tools else 'OFF'}")
    if session_model:
        print(f"  Forced: {session_model}")
    print(f"  Type /help for commands, /exit to quit")
    print()

    # Graceful Ctrl-C
    def _sigint(sig, frame):
        print("\n\nExiting chat.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    while True:
        # Prompt
        try:
            user_input = input("ech0> ").strip()
        except EOFError:
            print("\nExiting chat.")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            args = user_input[len(cmd):].strip()

            if cmd in ("/exit", "/quit"):
                print("Exiting chat.")
                break

            elif cmd == "/help":
                print(_slash_help())
                continue

            elif cmd == "/status":
                status = orch.status()
                print(f"  Loaded:  {status['loaded_model']}")
                print(f"  Policy:  {status['swap_policy']}")
                for name in status["roster"]:
                    be = status["backends"].get(name, "ollama")
                    print(f"  → {name} [{be}]")
                print(f"  Swaps:   {status['swap_count']}")
                print(f"  Turns:   {turn_count}")
                print(f"  History: {len(history)} messages")
                print(f"  Tools:   {'ON' if session_tools else 'OFF'}")
                continue

            elif cmd == "/memory":
                mem = orch.memory.to_dict()
                print(f"  Turn count: {mem.get('turn_count', 0)}")
                print(f"  Entities:   {len(mem.get('entities', {}))}")
                recent = mem.get("recent_intents", [])
                if recent:
                    print(f"  Recent:     {', '.join(recent[-5:])}")
                continue

            elif cmd == "/clear":
                history.clear()
                turn_count = 0
                print("  Session history cleared.")
                continue

            elif cmd == "/model":
                if args:
                    session_model = args
                    print(f"  Forced model: {session_model}")
                else:
                    session_model = None
                    print("  Model forcing cleared (auto-route).")
                continue

            elif cmd == "/tools":
                session_tools = not session_tools
                print(f"  Tools: {'ON' if session_tools else 'OFF'}")
                continue

            elif cmd == "/history":
                if not history:
                    print("  (empty)")
                else:
                    for msg in history:
                        role = msg["role"]
                        text = msg["content"][:120]
                        prefix = "  you:" if role == "user" else "  bot:"
                        print(f"{prefix} {text}")
                continue

            else:
                print(f"  Unknown command: {cmd}. Type /help.")
                continue

        # --- Normal turn ---
        turn_count += 1
        t0 = time.time()

        # Trim history to sliding window
        trimmed = history[-(MAX_HISTORY_TURNS * 2):]

        try:
            result = orch.process(
                user_input,
                force_model=session_model,
                use_tools=session_tools,
                history=trimmed if trimmed else None,
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        output = result.get("output", "")
        elapsed = time.time() - t0

        # Update session history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": output})

        # Print response
        model_used = result.get("model", "?")
        tok_s = result.get("tok_s", 0)
        swap_info = ""
        if result.get("swapped"):
            swap_info = f" [swapped in {result.get('swap_time_s', 0):.1f}s]"

        print()
        print(output)
        print()
        print(f"  [{model_used}] {tok_s} tok/s, {elapsed:.1f}s{swap_info} (turn {turn_count})")

        # Show escalations if any
        if result.get("escalations"):
            print(f"  ESCALATION REQUESTS: {len(result['escalations'])}")
            for i, esc in enumerate(result["escalations"], 1):
                print(f"    [{i}] {esc.get('request_type', '').upper()}: {esc.get('goal', '')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(prog="ech0 chat", description="Interactive chat with local models")
    parser.add_argument("--force-model", "-f", type=str, default=None)
    parser.add_argument("--tools", "-t", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    run_chat(config_path=args.config, force_model=args.force_model, use_tools=args.tools)


if __name__ == "__main__":
    main()
