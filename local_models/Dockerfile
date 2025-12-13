FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir \
    huggingface_hub \
    "llama-cpp-python[server]" \
    sentence-transformers \
    fastapi \
    uvicorn

# Create app directory
WORKDIR /app

# Copy startup script
COPY start.sh /app/start.sh
COPY embedding_server.py /app/embedding_server.py
RUN chmod +x /app/start.sh

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8081
ENV MODEL_REPO="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF"
ENV MODEL_FILE="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"

# Entrypoint
CMD ["/app/start.sh"]
