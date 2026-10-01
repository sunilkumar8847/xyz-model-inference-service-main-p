# scripts/start_triton.sh
#!/bin/bash

# Start Triton Inference Server locally
# Requires: Docker, NVIDIA Docker runtime, GPU

echo "Starting Triton Inference Server..."
echo "Model repository: /tmp/triton_models"

# Create model directory if it doesn't exist
mkdir -p /tmp/triton_models

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA GPU not found. Running Triton in CPU mode..."
    docker run --rm \
      --name triton-inference \
      -p 8000:8000 \
      -p 8001:8001 \
      -p 8002:8002 \
      -v /tmp/triton_models:/models:ro \
      nvcr.io/nvidia/tritonserver:23.10-py3 \
      tritonserver --model-repository=/models --log-verbose=true
else
    echo "✓ NVIDIA GPU detected. Running Triton with GPU support..."
    docker run --rm \
      --gpus all \
      --name triton-inference \
      -p 8000:8000 \
      -p 8001:8001 \
      -p 8002:8002 \
      -v /tmp/triton_models:/models:ro \
      nvcr.io/nvidia/tritonserver:23.10-py3 \
      tritonserver --model-repository=/models --log-verbose=true
fi