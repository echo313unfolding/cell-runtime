#!/usr/bin/env python3
"""Model pool — manages Ollama, Helix, and llama-server backends for the cell runtime.

Three backends, one GPU:
  - ollama: SmolLM3 (swap via api/chat with num_predict=1)
  - helix:  HXQ-compressed models via affine_server.py (OpenAI-compatible)
  - llama-server: Native GGUF via llama-server (OpenAI-compatible, HXQ fork)

Only one model can hold the GPU at a time. Swapping between backends
requires unloading one and starting the other.
"""
import json
import os
import signal
import subprocess
import time
import urllib.request


class ModelPool:
    def __init__(self, config: dict):
        self.ollama_url = config.get("ollama_url", "http://localhost:11434")
        self.helix_url = config.get("helix_url", "http://localhost:8001")
        self.llama_server_url = config.get("llama_server_url", "http://localhost:8080")
        self.roster = config.get("roster", {})
        self._swap_history = []
        self._helix_proc = None
        self._llama_proc = None
        self._llama_current_model = None

    def _backend_for(self, model: str) -> str:
        """Return 'ollama' or 'helix' for a roster model."""
        return self.roster.get(model, {}).get("backend", "ollama")

    def _normalize_name(self, name: str) -> str:
        """Strip :latest tag for comparison."""
        if name.endswith(":latest"):
            name = name[:-7]
        return name

    # ── Generic HTTP helper ──

    def _http(self, url: str, data: dict = None, timeout: int = 180) -> dict:
        """POST/GET JSON to a URL."""
        if data is not None:
            payload = json.dumps(data).encode()
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
        else:
            req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    # ── Ollama backend ──

    def _api(self, endpoint: str, data: dict = None, timeout: int = 180) -> dict:
        """Call Ollama API."""
        return self._http(f"{self.ollama_url}/{endpoint}", data, timeout)

    def _ollama_loaded(self) -> str | None:
        """Return the roster model currently loaded in Ollama, or None."""
        try:
            data = self._api("api/ps", timeout=5)
        except Exception:
            return None
        for m in data.get("models", []):
            name = self._normalize_name(m.get("name", ""))
            if name in self.roster:
                return name
            raw_name = m.get("name", "")
            if raw_name in self.roster:
                return raw_name
        return None

    def _ollama_unload(self, model: str):
        """Unload an Ollama model to free VRAM."""
        try:
            self._api("api/generate", {
                "model": model,
                "prompt": "",
                "keep_alive": 0,
            }, timeout=30)
        except Exception:
            pass

    # ── Helix backend ──

    def _helix_alive(self) -> bool:
        """Check if helix inference server is responding."""
        try:
            data = self._http(f"{self.helix_url}/health", timeout=5)
            return data.get("status") == "ok"
        except Exception:
            return False

    def _helix_model_name(self) -> str | None:
        """Return the model name the helix server is serving, or None."""
        try:
            data = self._http(f"{self.helix_url}/health", timeout=5)
            return data.get("model")
        except Exception:
            return None

    def _helix_start(self, model: str) -> dict:
        """Start the helix inference server for a model. Returns status dict."""
        model_path = os.path.expanduser(
            self.roster.get(model, {}).get("model_path", ""))
        if not model_path or not os.path.exists(model_path):
            return {"error": f"model_path not found for {model}: {model_path}"}

        server_script = os.path.expanduser(
            "~/cell-runtime/cell/affine_server.py")
        if not os.path.exists(server_script):
            return {"error": f"affine_server.py not found at {server_script}"}

        # Parse port from helix_url
        port = self.helix_url.rsplit(":", 1)[-1].rstrip("/")

        log_path = os.path.expanduser("~/receipts/cell/helix_server.log")
        t0 = time.time()

        # Start server in background
        env = os.environ.copy()
        self._helix_proc = subprocess.Popen(
            ["python3", server_script,
             "--model", model_path,
             "--port", port],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        # Wait for server to be ready (model load takes 15-30s)
        for i in range(60):
            time.sleep(1)
            if self._helix_alive():
                return {
                    "started": True,
                    "pid": self._helix_proc.pid,
                    "load_time_s": round(time.time() - t0, 1),
                    "log": log_path,
                }
            # Check if process died
            if self._helix_proc.poll() is not None:
                return {"error": f"helix server exited with code {self._helix_proc.returncode}. Check {log_path}"}

        # Timeout
        self._helix_stop()
        return {"error": f"helix server did not respond within 60s. Check {log_path}"}

    def _helix_stop(self):
        """Stop the helix inference server."""
        if self._helix_proc and self._helix_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._helix_proc.pid), signal.SIGTERM)
                self._helix_proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._helix_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._helix_proc = None
        # Also kill any orphaned helix server
        try:
            result = subprocess.run(
                ["pgrep", "-f", "affine_server"],
                capture_output=True, text=True, timeout=5)
            for pid in result.stdout.strip().split("\n"):
                if pid.strip():
                    os.kill(int(pid.strip()), signal.SIGTERM)
        except Exception:
            pass

    # ── llama-server backend ──

    def _llama_server_alive(self) -> bool:
        """Check if llama-server is responding."""
        try:
            data = self._http(f"{self.llama_server_url}/health", timeout=5)
            return data.get("status") == "ok"
        except Exception:
            return False

    def _llama_server_start(self, model: str) -> dict:
        """Start llama-server for a GGUF model. Returns status dict."""
        gguf_path = os.path.expanduser(
            self.roster.get(model, {}).get("gguf", ""))
        if not gguf_path or not os.path.exists(gguf_path):
            return {"error": f"GGUF not found for {model}: {gguf_path}"}

        # Find llama-server binary
        llama_server = None
        for candidate in ["/usr/local/bin/llama-server",
                          os.path.expanduser("~/llama.cpp/build/bin/llama-server")]:
            if os.path.exists(candidate):
                llama_server = candidate
                break
        if not llama_server:
            return {"error": "llama-server binary not found"}

        port = self.llama_server_url.rsplit(":", 1)[-1].rstrip("/")
        gen_cfg = {"n_ctx": 4096, "n_gpu_layers": 99}

        log_path = os.path.expanduser(
            os.environ.get("CELL_RECEIPTS_DIR", "~/receipts/cell")
        ) + "/llama_server.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        t0 = time.time()

        env = os.environ.copy()
        # Ensure shared libs are found for locally-built llama-server
        lib_dir = os.path.dirname(llama_server)
        if lib_dir:
            env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        self._llama_proc = subprocess.Popen(
            [llama_server,
             "--model", gguf_path,
             "--port", port,
             "--ctx-size", str(gen_cfg["n_ctx"]),
             "--n-gpu-layers", str(gen_cfg["n_gpu_layers"]),
             "--host", "0.0.0.0"],
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

        # Wait for server to be ready
        for i in range(120):
            time.sleep(1)
            if self._llama_server_alive():
                self._llama_current_model = model
                return {
                    "started": True,
                    "pid": self._llama_proc.pid,
                    "load_time_s": round(time.time() - t0, 1),
                    "log": log_path,
                }
            if self._llama_proc.poll() is not None:
                self._llama_current_model = None
                return {"error": f"llama-server exited with code {self._llama_proc.returncode}. Check {log_path}"}

        self._llama_server_stop()
        return {"error": f"llama-server did not respond within 120s. Check {log_path}"}

    def _llama_server_stop(self):
        """Stop the llama-server process."""
        if self._llama_proc and self._llama_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._llama_proc.pid), signal.SIGTERM)
                self._llama_proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._llama_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self._llama_proc = None
        self._llama_current_model = None
        # Kill orphaned llama-server processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server"],
                capture_output=True, text=True, timeout=5)
            for pid in result.stdout.strip().split("\n"):
                if pid.strip():
                    os.kill(int(pid.strip()), signal.SIGTERM)
        except Exception:
            pass

    # ── Unified interface ──

    def which_loaded(self) -> str | None:
        """Return whichever roster model is currently loaded (any backend)."""
        # Check llama-server first (native GGUF path)
        if self._llama_current_model and self._llama_server_alive():
            return self._llama_current_model

        # Check helix
        if self._helix_alive():
            hname = self._helix_model_name()
            for rname, rcfg in self.roster.items():
                if rcfg.get("backend") == "helix":
                    mpath = os.path.expanduser(rcfg.get("model_path", ""))
                    if hname and (hname in mpath or mpath.endswith(hname)):
                        return rname
            if hname:
                return f"helix:{hname}"

        # Check Ollama
        return self._ollama_loaded()

    def swap_to(self, target: str) -> dict:
        """Ensure target model is loaded. Handles cross-backend swaps."""
        current = self.which_loaded()
        target_backend = self._backend_for(target)

        # Already loaded?
        if current == target:
            return {"swapped": False, "already_loaded": target, "swap_time_s": 0}

        t0 = time.time()
        current_backend = self._backend_for(current) if current and current in self.roster else None

        # ── Free VRAM from current backend ──
        # Always check for rogue llama-server processes holding the GPU
        self._llama_server_stop()
        if current_backend == "helix" or self._helix_alive():
            self._helix_stop()
            time.sleep(2)
        if current_backend == "ollama" and current:
            self._ollama_unload(current)
            time.sleep(1)

        # ── Load target ──
        if target_backend == "llama-server":
            start_result = self._llama_server_start(target)
            if start_result.get("error"):
                return {"swapped": False, "error": start_result["error"],
                        "swap_time_s": round(time.time() - t0, 2)}
            swap_time = round(time.time() - t0, 2)
            event = {
                "swapped": True,
                "from": current or "(none)",
                "to": target,
                "swap_time_s": swap_time,
                "backend": "llama-server",
                "pid": start_result.get("pid"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self._swap_history.append(event)
            return event

        elif target_backend == "helix":
            start_result = self._helix_start(target)
            if start_result.get("error"):
                return {"swapped": False, "error": start_result["error"],
                        "swap_time_s": round(time.time() - t0, 2)}
            swap_time = round(time.time() - t0, 2)
            event = {
                "swapped": True,
                "from": current or "(none)",
                "to": target,
                "swap_time_s": swap_time,
                "backend": "helix",
                "helix_pid": start_result.get("pid"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self._swap_history.append(event)
            return event

        else:
            # Ollama backend
            try:
                self._api("api/chat", {
                    "model": target,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "options": {"num_predict": 1},
                }, timeout=180)
            except Exception as e:
                return {"swapped": False, "error": str(e),
                        "swap_time_s": round(time.time() - t0, 2)}

            swap_time = round(time.time() - t0, 2)
            event = {
                "swapped": True,
                "from": current or "(none)",
                "to": target,
                "swap_time_s": swap_time,
                "backend": "ollama",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self._swap_history.append(event)
            return event

    def get_swap_history(self) -> list:
        return list(self._swap_history)

    def cleanup(self):
        """Stop any managed processes."""
        self._helix_stop()
        self._llama_server_stop()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    config_path = __import__("pathlib").Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    pool = ModelPool(config)
    loaded = pool.which_loaded()
    print(f"Currently loaded: {loaded or '(none)'}")
    print(f"Helix alive: {pool._helix_alive()}")
    print(f"Ollama loaded: {pool._ollama_loaded()}")
    print(f"Roster: {list(config['roster'].keys())}")
