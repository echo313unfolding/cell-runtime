#!/usr/bin/env bash
# Memory Lane A/B Test — Zamba2-1.2B vs Qwen2.5-1.5B
# Runs both models on the same 20-event Docker build saga
# and compares state-tracking accuracy.
#
# Usage: bash tools/capsule/run_memory_lane_ab.sh

set -euo pipefail

LLAMA_SERVER="${HOME}/llama.cpp/build/bin/llama-server"
TEST_SCRIPT="${HOME}/tools/capsule/memory_lane_test.py"
PORT=8090

ZAMBA_GGUF="${HOME}/cloud-work/ggufs/zamba2-1.2b-instruct-v2-q8_0.gguf"
QWEN_GGUF="${HOME}/models/qwen2.5-coder-1.5b-instruct-q8_0.gguf"

echo "============================================"
echo "  Memory Lane A/B Test"
echo "  Zamba2-1.2B (hybrid SSM) vs Qwen2.5-1.5B (decoder-only)"
echo "============================================"
echo ""

# Kill any existing llama-server on our port
pkill -f "llama-server.*--port ${PORT}" 2>/dev/null || true
sleep 1

# --- Model A: Zamba2-1.2B ---
echo "[1/2] Starting Zamba2-1.2B..."
"${LLAMA_SERVER}" \
    -m "${ZAMBA_GGUF}" \
    --port ${PORT} \
    -ngl 99 \
    --ctx-size 4096 \
    --log-disable \
    &
ZAMBA_PID=$!

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" | grep -q "ok"; then
        echo "  Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  ERROR: Zamba2 server failed to start"
        kill $ZAMBA_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo "  Running test..."
python3 "${TEST_SCRIPT}" --model-name zamba2-1.2b --port ${PORT} || true

kill $ZAMBA_PID 2>/dev/null
wait $ZAMBA_PID 2>/dev/null || true
sleep 2

# --- Model B: Qwen2.5-Coder-1.5B ---
echo "[2/2] Starting Qwen2.5-Coder-1.5B..."
"${LLAMA_SERVER}" \
    -m "${QWEN_GGUF}" \
    --port ${PORT} \
    -ngl 99 \
    --ctx-size 4096 \
    --log-disable \
    &
QWEN_PID=$!

echo "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" | grep -q "ok"; then
        echo "  Server ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "  ERROR: Qwen server failed to start"
        kill $QWEN_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo "  Running test..."
python3 "${TEST_SCRIPT}" --model-name qwen2.5-1.5b --port ${PORT} || true

kill $QWEN_PID 2>/dev/null
wait $QWEN_PID 2>/dev/null || true

# --- Compare ---
echo ""
echo "============================================"
echo "  Results saved to ~/receipts/"
echo "  Compare with:"
echo "    python3 -c \""
echo "import json, glob"
echo "files = sorted(glob.glob(os.path.expanduser('~/receipts/memory_lane_test_*.json')))"
echo "for f in files[-2:]:"
echo "    d = json.load(open(f))"
echo "    print(f'{d[\"model\"]:20s} score={d[\"mean_overall_score\"]:.4f} fails={d[\"parse_failures\"]} latency={d[\"median_latency_ms\"]}ms')"
echo "\""
echo "============================================"
