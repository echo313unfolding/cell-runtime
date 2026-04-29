# Echo Capsule Runtime — 3-model local AI assistant
# Builds llama.cpp (HXQ fork) + capsule orchestrator
#
# Usage:
#   docker build -t echo-cell .
#   docker run --gpus all -v /path/to/models:/models echo-cell "write a sort function"
#   docker run --gpus all -v /path/to/models:/models echo-cell status
#   docker run --gpus all -v /path/to/models:/models -it echo-cell --interactive
#
# To auto-download models from HF on first run:
#   docker run --gpus all -v capsule-models:/models echo-cell --download smollm3
#   docker run --gpus all -v capsule-models:/models echo-cell --download all

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Build llama.cpp HXQ fork
WORKDIR /build
COPY llama.cpp/ ./llama.cpp/
WORKDIR /build/llama.cpp
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86;89" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build -j$(nproc) --target llama-server llama-quantize llama-cli llama-perplexity

# --- Runtime stage ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl procps libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir huggingface_hub

# Copy llama.cpp binaries + shared libraries
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-quantize /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-cli /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/llama-perplexity /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/bin/*.so* /usr/local/lib/
RUN ldconfig

# Copy capsule runtime — preserve package structure so imports work:
#   /app/tools/router.py
#   /app/tools/task_record.py
#   /app/cell/__init__.py
#   /app/cell/orchestrator.py
#   ...
# orchestrator.py does sys.path.insert(0, parent.parent) → /app/tools/
# then: from router import classify (finds /app/tools/router.py)
# and:  from capsule.model_pool import ModelPool (finds /app/cell/)
WORKDIR /app/cell
COPY cell/orchestrator.py .
COPY cell/model_pool.py .
COPY cell/tool_registry.py .
COPY cell/config.native.json ./config.json
COPY cell/__init__.py .
COPY cell/mcp_server.py .
COPY cell/mcp_wrapper.sh .
COPY cell/download_models.py .
COPY cell/gateway.py .
COPY tools/router.py /app/tools/
COPY tools/task_record.py /app/tools/
COPY bin/ech0 /usr/local/bin/ech0

# Default model directory (mount your GGUFs here)
VOLUME /models
VOLUME /receipts

ENV CAPSULE_MODELS_DIR=/models
ENV CAPSULE_RECEIPTS_DIR=/receipts
ENV PYTHONUNBUFFERED=1

# llama-server port (internal) + gateway port (API surface)
EXPOSE 8080 8800

ENTRYPOINT ["python3", "/app/cell/orchestrator.py"]
CMD ["--help"]
