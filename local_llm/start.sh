#!/bin/bash
set -e

CACHE_DIR="/root/.cache/huggingface"
MODEL_PATH="$CACHE_DIR/$MODEL_FILE"

echo "Checking for model $MODEL_FILE in $CACHE_DIR..."

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found. Downloading from $MODEL_REPO..."
    echo "Model not found. Downloading from $MODEL_REPO..."
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$MODEL_REPO', filename='$MODEL_FILE', local_dir='$CACHE_DIR', local_dir_use_symlinks=False)"
    echo "Download complete."
    echo "Download complete."
else
    echo "Model found at $MODEL_PATH."
fi

echo "Starting Local LLM Server..."
# Run llama-cpp-python server
# -n_ctx 2048: Context window (adjust as needed)
# -n_gpu_layers -1: Offload all to GPU if available (auto-detect)
python3 -m llama_cpp.server --model $MODEL_PATH --host $HOST --port $PORT --n_ctx 4096 --n_gpu_layers -1
