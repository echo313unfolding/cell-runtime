#!/usr/bin/env python3
"""Pod script: Quantize Qwen2.5-Coder-7B-Instruct to native HXQ_AFFINE_6 GGUF.

Run on RunPod (3090/4090/A6000) — fast download + quantize, then scp result back.

Steps:
  1. Clone llama.cpp hxq-affine-type branch
  2. Apply 3 bug fixes (get_rows, quantizer, ftype)
  3. Build llama-quantize with CUDA
  4. Download Qwen2.5-Coder-7B-Instruct F16 GGUF (~14 GB)
  5. Quantize F16 → HXQ_AFFINE_6 (~3-4 GB expected)
  6. Quick sanity check (llama-bench or llama-server)
  7. Output: /root/Qwen2.5-Coder-7B-Instruct-HXQ-AFFINE-6.gguf

Usage on pod:
    nohup python3 hxq_native_7b_quantize.py > /root/hxq_quantize.log 2>&1 &

Transfer back to T2000:
    scp -P <port> -i ~/.ssh/id_ed25519_runpod \
        root@<ip>:/root/Qwen2.5-Coder-7B-Instruct-HXQ-AFFINE-6.gguf \
        ~/models/

Then locally:
    python3 tools/sweep_gpu_layers.py \
        ~/models/Qwen2.5-Coder-7B-Instruct-HXQ-AFFINE-6.gguf \
        --layers 0,4,8,12,16,20,21,22,23,24,25,26,28,99
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

LLAMA_REPO = "https://github.com/echo313unfolding/llama.cpp.git"
LLAMA_BRANCH = "hxq-affine-type"
WORK_DIR = "/root"
F16_GGUF = "Qwen2.5-Coder-7B-Instruct-f16.gguf"
HXQ_GGUF = "Qwen2.5-Coder-7B-Instruct-HXQ-AFFINE-6.gguf"


def run(cmd, cwd=None, timeout=600):
    """Run command, print output, raise on failure."""
    print(f"\n>>> {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd}")
    return result.stdout


def main():
    t_start = time.time()
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 60)
    print("HXQ Native GGUF Quantization — Pod Script")
    print("=" * 60)
    print(f"Host: {platform.node()}")
    print(f"Start: {start_iso}")
    print()

    # ── Step 1: Clone llama.cpp ──
    llama_dir = os.path.join(WORK_DIR, "llama.cpp")
    if not os.path.isdir(llama_dir):
        print("Step 1: Cloning llama.cpp...")
        run(f"git clone --branch {LLAMA_BRANCH} --depth 1 {LLAMA_REPO} {llama_dir}")
    else:
        print("Step 1: llama.cpp already cloned, pulling latest...")
        run(f"cd {llama_dir} && git pull origin {LLAMA_BRANCH}")

    # Verify branch
    branch = run("git branch --show-current", cwd=llama_dir).strip()
    print(f"  Branch: {branch}")
    assert branch == LLAMA_BRANCH, f"Wrong branch: {branch}"

    # ── Step 2: Apply bug fixes ──
    print("\nStep 2: Applying HXQ bug fixes...")

    # Fix 1: get_rows support for HXQ types in ops.cpp
    ops_path = os.path.join(llama_dir, "ggml/src/ggml-cpu/ops.cpp")
    with open(ops_path) as f:
        ops_src = f.read()
    if "GGML_TYPE_HXQ_AFFINE_6" not in ops_src:
        old = """        case GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } break;"""
        new = """        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_HXQ_AFFINE_G128:
        case GGML_TYPE_HXQ_AFFINE_6:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } break;"""
        if old in ops_src:
            ops_src = ops_src.replace(old, new)
            with open(ops_path, "w") as f:
                f.write(ops_src)
            print("  Fixed: get_rows HXQ support in ops.cpp")
        else:
            print("  WARN: Could not find get_rows patch point in ops.cpp")
    else:
        print("  get_rows fix already applied")

    # Fix 2: quantize_chunk for HXQ_AFFINE_6 in ggml.c
    ggml_path = os.path.join(llama_dir, "ggml/src/ggml.c")
    with open(ggml_path) as f:
        ggml_src = f.read()
    if "case GGML_TYPE_HXQ_AFFINE_6:" not in ggml_src:
        old2 = "case GGML_TYPE_HXQ_AFFINE_G128: result = quantize_hxq_affine_g128(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n        case GGML_TYPE_MXFP4:"
        new2 = "case GGML_TYPE_HXQ_AFFINE_G128: result = quantize_hxq_affine_g128(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n        case GGML_TYPE_HXQ_AFFINE_6:    result = quantize_hxq_affine_6(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n        case GGML_TYPE_MXFP4:"
        if old2 in ggml_src:
            ggml_src = ggml_src.replace(old2, new2)
            with open(ggml_path, "w") as f:
                f.write(ggml_src)
            print("  Fixed: quantize_chunk HXQ_AFFINE_6 in ggml.c")
        else:
            print("  WARN: Could not find quantize_chunk patch point in ggml.c")
    else:
        print("  quantize_chunk fix already applied")

    # Fix 3: ftype switch in llama-model-loader.cpp
    loader_path = os.path.join(llama_dir, "src/llama-model-loader.cpp")
    with open(loader_path) as f:
        loader_src = f.read()
    if "GGML_TYPE_HXQ_AFFINE_G128" not in loader_src.split("switch (type_max)")[-1].split("default:")[0]:
        old3 = """            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            default:"""
        new3 = """            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            case GGML_TYPE_HXQ_AFFINE_G128: ftype = LLAMA_FTYPE_MOSTLY_HXQ_AFFINE_G128; break;
            case GGML_TYPE_HXQ_AFFINE_6:    ftype = LLAMA_FTYPE_MOSTLY_HXQ_AFFINE_6;    break;
            default:"""
        if old3 in loader_src:
            loader_src = loader_src.replace(old3, new3)
            with open(loader_path, "w") as f:
                f.write(loader_src)
            print("  Fixed: ftype switch in llama-model-loader.cpp")
        else:
            print("  WARN: Could not find ftype patch point in llama-model-loader.cpp")
    else:
        print("  ftype fix already applied")

    # ── Step 3: Build ──
    print("\nStep 3: Building llama.cpp with CUDA...")
    build_dir = os.path.join(llama_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    run("cmake .. -DGGML_CUDA=ON", cwd=build_dir, timeout=120)
    run(f"make -j$(nproc) llama-quantize llama-server", cwd=build_dir, timeout=600)

    quantize_bin = os.path.join(build_dir, "bin/llama-quantize")
    assert os.path.isfile(quantize_bin), f"llama-quantize not found at {quantize_bin}"
    print(f"  Built: {quantize_bin}")

    # ── Step 4: Download F16 GGUF ──
    f16_path = os.path.join(WORK_DIR, F16_GGUF)
    if not os.path.isfile(f16_path):
        print(f"\nStep 4: Downloading {F16_GGUF}...")
        run(f"pip install -q huggingface_hub", timeout=120)
        run(
            f"huggingface-cli download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF "
            f"{F16_GGUF} --local-dir {WORK_DIR} --local-dir-use-symlinks False",
            timeout=600)
    else:
        print(f"\nStep 4: {F16_GGUF} already exists")

    f16_size = os.path.getsize(f16_path)
    print(f"  F16 GGUF: {f16_size / 1e9:.2f} GB")

    # ── Step 5: Quantize to HXQ_AFFINE_6 ──
    hxq_path = os.path.join(WORK_DIR, HXQ_GGUF)
    print(f"\nStep 5: Quantizing to HXQ_AFFINE_6...")
    t_quant = time.time()

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.join(build_dir, "bin") + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [quantize_bin, f16_path, hxq_path, "HXQ_AFFINE_6"],
        capture_output=True, text=True, env=env, timeout=600)

    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        # Write failure receipt
        receipt = {
            "status": "FAIL",
            "error": result.stderr[-500:],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(WORK_DIR, "hxq_quantize_FAIL.json"), "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"\nFAILED. See hxq_quantize_FAIL.json")
        sys.exit(1)

    quant_time = round(time.time() - t_quant, 1)
    hxq_size = os.path.getsize(hxq_path)
    print(f"  HXQ GGUF: {hxq_size / 1e9:.2f} GB")
    print(f"  Quantize time: {quant_time}s")
    print(f"  Compression: {f16_size / hxq_size:.2f}x from F16")

    # ── Step 6: CUDA runtime validation ──
    # This is the critical gate: does HXQ actually use GPU, or CPU fallback?
    print("\nStep 6: CUDA runtime validation (ngl=0 vs ngl=max)...")
    import urllib.request
    server_bin = os.path.join(build_dir, "bin/llama-server")
    assert os.path.isfile(server_bin), "llama-server not built"

    TEST_PROMPT = json.dumps({
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function to check if a number is prime. Keep it short."},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }).encode()

    def get_vram_mb():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip())
        except Exception:
            return -1

    def run_config(ngl, port=8099):
        """Start server at given ngl, run prompt, return metrics, kill server."""
        # Kill any leftover
        subprocess.run(["pkill", "-f", "llama-server"], capture_output=True, timeout=5)
        time.sleep(2)
        vram_before = get_vram_mb()

        log_path = os.path.join(WORK_DIR, f"hxq_sweep_ngl{ngl}.log")
        proc = subprocess.Popen(
            [server_bin, "--model", hxq_path, "--port", str(port),
             "--ctx-size", "512", "--n-gpu-layers", str(ngl), "--host", "0.0.0.0"],
            stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)

        # Wait for ready
        ready = False
        for i in range(120):
            time.sleep(1)
            try:
                req = urllib.request.Request(f"http://localhost:{port}/health")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                    if data.get("status") == "ok":
                        ready = True
                        break
            except Exception:
                pass
            if proc.poll() is not None:
                break

        if not ready:
            proc.terminate()
            proc.wait(timeout=10)
            return {"ngl": ngl, "status": "FAIL", "reason": "server_timeout"}

        vram_after = get_vram_mb()

        # Run prompt
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=TEST_PROMPT,
            headers={"Content-Type": "application/json"})
        t_gen = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            gen_data = json.loads(resp.read())
        gen_time = round(time.time() - t_gen, 2)

        usage = gen_data.get("usage", {})
        content = gen_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = usage.get("completion_tokens", 0)
        tok_s = round(tokens / gen_time, 2) if gen_time > 0 else 0

        # Read log for CUDA offload evidence
        proc.terminate()
        proc.wait(timeout=10)
        with open(log_path) as f:
            log_text = f.read()
        cuda_offload = "offloaded" in log_text.lower() or "CUDA" in log_text

        return {
            "ngl": ngl,
            "status": "PASS",
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_delta_mb": vram_after - vram_before,
            "tok_s": tok_s,
            "tokens": tokens,
            "gen_time_s": gen_time,
            "content_preview": content[:200],
            "cuda_evidence_in_log": cuda_offload,
            "log_path": log_path,
        }

    # Run ngl=0 (CPU only) and ngl=99 (full GPU)
    result_cpu = run_config(0)
    result_gpu = run_config(99)

    print(f"\n  ngl=0:  VRAM={result_cpu.get('vram_delta_mb', '?')} MB, tok/s={result_cpu.get('tok_s', '?')}")
    print(f"  ngl=99: VRAM={result_gpu.get('vram_delta_mb', '?')} MB, tok/s={result_gpu.get('tok_s', '?')}")

    # ── CUDA validation gate ──
    cuda_valid = False
    cuda_verdict = "UNKNOWN"

    if result_cpu["status"] == "PASS" and result_gpu["status"] == "PASS":
        vram_diff = result_gpu["vram_delta_mb"] - result_cpu["vram_delta_mb"]
        speed_ratio = result_gpu["tok_s"] / max(result_cpu["tok_s"], 0.01)

        print(f"  VRAM difference (ngl=99 - ngl=0): {vram_diff} MB")
        print(f"  Speed ratio (ngl=99 / ngl=0): {speed_ratio:.2f}x")

        if vram_diff > 500 and speed_ratio > 1.5:
            cuda_valid = True
            cuda_verdict = "PASS: real CUDA offload confirmed"
        elif vram_diff > 200:
            cuda_verdict = "PARTIAL: some CUDA offload, speed gain weak"
        else:
            cuda_verdict = "FAIL: CPU fallback — VRAM does not change with ngl, no speed gain"
    else:
        cuda_verdict = f"FAIL: cpu={result_cpu['status']}, gpu={result_gpu['status']}"

    print(f"\n  CUDA VALIDATION: {cuda_verdict}")

    sanity = {
        "ngl_0": result_cpu,
        "ngl_99": result_gpu,
        "cuda_valid": cuda_valid,
        "cuda_verdict": cuda_verdict,
    }

    # ── Write receipt ──
    wall_time = round(time.time() - t_start, 1)
    overall_status = "PASS" if cuda_valid else "PARTIAL" if "PARTIAL" in cuda_verdict else "FAIL_CUDA"
    receipt = {
        "receipt_id": f"hxq_native_gguf_7b_quantize_{time.strftime('%Y%m%dT%H%M%SZ')}",
        "title": "HXQ Native GGUF Quantization + CUDA Validation: Qwen2.5-Coder-7B-Instruct",
        "status": overall_status,
        "source": {
            "model": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            "file": F16_GGUF,
            "size_bytes": f16_size,
        },
        "output": {
            "file": HXQ_GGUF,
            "path": hxq_path,
            "size_bytes": hxq_size,
            "compression_from_f16": round(f16_size / hxq_size, 2),
        },
        "quantize_time_s": quant_time,
        "sanity_check": sanity,
        "llama_cpp": {
            "branch": LLAMA_BRANCH,
            "fixes_applied": [
                "get_rows HXQ support in ops.cpp",
                "quantize_chunk HXQ_AFFINE_6 in ggml.c",
                "ftype switch HXQ in llama-model-loader.cpp",
            ],
        },
        "cost": {
            "wall_time_s": wall_time,
            "cpu_time_s": round(time.process_time(), 3),
            "peak_memory_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "next": {
            "transfer": f"scp -P <port> -i ~/.ssh/id_ed25519_runpod root@<ip>:{hxq_path} ~/models/",
            "local_sweep": f"python3 tools/sweep_gpu_layers.py ~/models/{HXQ_GGUF} --layers 0,4,8,12,16,20,21,22,23,24,25,26,28,99",
        },
    }

    receipt_path = os.path.join(WORK_DIR, "hxq_native_gguf_7b_quantize.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE — {overall_status}")
    print("=" * 60)
    print(f"  HXQ GGUF: {hxq_path} ({hxq_size / 1e9:.2f} GB)")
    print(f"  CUDA validation: {cuda_verdict}")
    print(f"  Receipt: {receipt_path}")
    print(f"  Wall time: {wall_time}s")

    if cuda_valid:
        print(f"\nCUDA OFFLOAD CONFIRMED. Transfer to T2000:")
        print(f"  scp -P <port> -i ~/.ssh/id_ed25519_runpod root@<ip>:{hxq_path} ~/models/")
    else:
        print(f"\nCUDA OFFLOAD NOT CONFIRMED. Do NOT transfer yet.")
        print(f"  The HXQ GGUF was created but GPU inference path is not working.")
        print(f"  Check logs at /root/hxq_sweep_ngl*.log for details.")
        print(f"  Likely needs: CUDA mmvq kernel for GGML_TYPE_HXQ_AFFINE_6")


if __name__ == "__main__":
    main()
