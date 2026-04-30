#!/usr/bin/env python3
"""OpenAI-compatible API gateway for the Cell 3-lane runtime.

Wraps the cell orchestrator behind standard OpenAI endpoints so any
client that speaks the OpenAI API can use cell as a drop-in backend.

Endpoints:
    POST /v1/chat/completions  — auto-routes or forces a lane
    GET  /v1/models            — lists cell-auto + individual lane models
    GET  /health               — liveness probe
    GET  /status               — full cell state (loaded model, swaps, roster)
    GET  /swap-history         — swap event log

Usage:
    python3 gateway.py                          # default :8800
    python3 gateway.py --port 9000              # custom port
    python3 gateway.py --config config.native.json --port 8800
    curl http://localhost:8800/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"cell-auto","messages":[{"role":"user","content":"write a sort function"}]}'
"""
import argparse
import json
import os
import platform
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

# Add tools/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cell.orchestrator import Orchestrator

# Model aliases: these names are exposed in the API
CAPSULE_AUTO = "cell-auto"
LANE_MODELS = {
    "cell-reasoning": "smollm3",
    "cell-coder": "qwen2.5-coder",
    "cell-sentinel": "qwen2.5-sentinel",
    "cell-memory": None,  # Memory lane (not a routable model)
}
# Reverse: roster name → API name
ROSTER_TO_API = {v: k for k, v in LANE_MODELS.items()}

# Created once per server lifetime
_orchestrator: Orchestrator = None


def _model_objects() -> list:
    """Build the /v1/models response list."""
    ts = int(time.time())
    models = [
        {
            "id": CAPSULE_AUTO,
            "object": "model",
            "created": ts,
            "owned_by": "echo-cell",
            "description": "Auto-routes to the best lane (reasoning, coding, security) based on intent.",
        },
    ]
    for api_name, roster_name in LANE_MODELS.items():
        info = _orchestrator.roster.get(roster_name, {})
        models.append({
            "id": api_name,
            "object": "model",
            "created": ts,
            "owned_by": "echo-cell",
            "description": f"Direct access to {roster_name} ({', '.join(info.get('intents', []))})",
        })
    # Also expose raw roster names
    for roster_name, info in _orchestrator.roster.items():
        models.append({
            "id": roster_name,
            "object": "model",
            "created": ts,
            "owned_by": "echo-cell",
        })
    return models


def _resolve_model(requested: str) -> str | None:
    """Map API model name to force_model arg (None = auto-route)."""
    if requested == CAPSULE_AUTO or not requested:
        return None
    if requested in LANE_MODELS:
        return LANE_MODELS[requested]
    if requested in _orchestrator.roster:
        return requested
    return None


def _handle_chat_completions(body: dict) -> tuple[dict, int]:
    """Process a /v1/chat/completions request, return (response_dict, status_code)."""
    messages = body.get("messages", [])
    if not messages:
        return {"error": {"message": "messages is required", "type": "invalid_request_error"}}, 400

    # Extract user content for classification
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return {"error": {"message": "at least one user message is required", "type": "invalid_request_error"}}, 400
    user_input = user_msgs[-1].get("content", "")

    # Resolve model
    requested_model = body.get("model", CAPSULE_AUTO)
    force_model = _resolve_model(requested_model)

    # Check if model name is valid
    if requested_model not in (CAPSULE_AUTO, "") and requested_model not in LANE_MODELS and requested_model not in _orchestrator.roster:
        return {"error": {"message": f"Unknown model: {requested_model}. Use /v1/models to list available models.", "type": "invalid_request_error"}}, 400

    # Check for tool use flag
    use_tools = body.get("cell_tools", False)

    # Run through orchestrator
    result = _orchestrator.process(user_input, force_model=force_model,
                                   use_tools=use_tools)

    if "error" in result:
        return {"error": {"message": result["error"], "type": "server_error"}}, 500

    # Build OpenAI-format response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    content = result.get("output", "")

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result.get("model", requested_model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": result.get("eval_count", 0),
            "total_tokens": result.get("eval_count", 0),
        },
        # Capsule-specific metadata (non-standard, safe to include)
        "cell": {
            "task_id": result.get("task_id"),
            "intent": result.get("intent"),
            "routed_model": result.get("model"),
            "swapped": result.get("swapped", False),
            "swap_time_s": result.get("swap_time_s", 0),
            "tok_s": result.get("tok_s", 0),
            "wall_time_s": result.get("wall_time_s", 0),
            "receipt": result.get("receipt"),
            "tool_calls": result.get("tool_calls", 0),
        },
    }
    if result.get("escalations"):
        response["cell"]["escalations"] = result["escalations"]

    return response, 200


class GatewayHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the cell gateway."""

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = self.path.rstrip("/")

        if path == "/v1/models":
            self._send_json({"object": "list", "data": _model_objects()})

        elif path == "/health":
            alive = _orchestrator.pool._llama_server_alive()
            loaded = _orchestrator.pool.which_loaded()
            self._send_json({
                "status": "ok" if alive or loaded else "degraded",
                "loaded_model": loaded or "(none)",
                "llama_server_alive": alive,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

        elif path == "/status":
            self._send_json(_orchestrator.status())

        elif path == "/swap-history":
            self._send_json({
                "swaps": _orchestrator.pool.get_swap_history(),
                "count": len(_orchestrator.pool.get_swap_history()),
            })

        elif path == "/v1/memory/capsule":
            self._send_json({
                "cell": _orchestrator.memory.get_capsule(),
                "turn_count": _orchestrator.memory.state.get("turn_count", 0),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

        else:
            self._send_json(
                {"error": {"message": f"Unknown endpoint: {self.path}", "type": "invalid_request_error"}},
                404)

    def do_POST(self):
        path = self.path.rstrip("/")

        if path == "/v1/chat/completions":
            raw = self._read_body()
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json(
                    {"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
                    400)
                return

            resp, status = _handle_chat_completions(body)
            self._send_json(resp, status)

        elif path == "/v1/memory/ingest":
            raw = self._read_body()
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json(
                    {"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
                    400)
                return
            _orchestrator.memory.ingest(body)
            self._send_json({
                "status": "ingested",
                "turn_count": _orchestrator.memory.state.get("turn_count", 0),
            })

        elif path == "/v1/memory/reset":
            _orchestrator.memory.reset()
            self._send_json({"status": "reset", "turn_count": 0})

        else:
            self._send_json(
                {"error": {"message": f"Unknown endpoint: {self.path}", "type": "invalid_request_error"}},
                404)

    def log_message(self, format, *args):
        """Override to include timestamp and model info."""
        sys.stderr.write(f"[gateway {time.strftime('%H:%M:%S')}] {format % args}\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread."""
    daemon_threads = True


def main():
    global _orchestrator

    parser = argparse.ArgumentParser(
        prog="cell-gateway",
        description="OpenAI-compatible API gateway for Cell 3-lane runtime",
    )
    parser.add_argument("--port", "-p", type=int, default=8800,
                        help="Port to listen on (default: 8800)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to cell config.json")
    args = parser.parse_args()

    _orchestrator = Orchestrator(config_path=args.config)

    server = ThreadedHTTPServer((args.host, args.port), GatewayHandler)
    loaded = _orchestrator.pool.which_loaded()

    print(f"Cell Gateway")
    print(f"  Listening: http://{args.host}:{args.port}")
    print(f"  Roster:    {', '.join(_orchestrator.roster.keys())}")
    print(f"  Loaded:    {loaded or '(cold start — first request will load)'}")
    print(f"  Policy:    {_orchestrator.swap_policy}")
    print()
    print(f"  Memory:    {'LLM-enhanced' if _orchestrator.memory.llm_enabled else 'programmatic'}")
    print()
    print(f"Endpoints:")
    print(f"  POST /v1/chat/completions  — chat (auto-routes or force model)")
    print(f"  GET  /v1/models            — list models")
    print(f"  GET  /health               — liveness")
    print(f"  GET  /status               — full state")
    print(f"  GET  /swap-history         — swap events")
    print(f"  GET  /v1/memory/capsule    — current memory state")
    print(f"  POST /v1/memory/ingest     — feed event to memory")
    print(f"  POST /v1/memory/reset      — clear memory")
    print()
    print(f"Models: {CAPSULE_AUTO}, {', '.join(LANE_MODELS.keys())}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        _orchestrator.pool.cleanup()
        server.server_close()


if __name__ == "__main__":
    main()
