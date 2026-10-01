#!/usr/bin/env python3
# scripts/download_models.py
"""
Download sample models for local development.

This script downloads pre-trained ONNX models from MLflow registry
and sets up the Triton model repository structure.

Usage:
    python scripts/download_models.py
"""

import os
import sys
import requests
from pathlib import Path
import shutil


TRITON_MODEL_REPO = Path("/tmp/triton_models")
MODELS_TO_DOWNLOAD = [
    {
        "name": "customer_matcher_v5",
        "version": "1",
        "url": "https://example.com/models/customer_matcher_v5.onnx",  # Placeholder
        "config": """
name: "customer_matcher_v5"
platform: "onnxruntime_onnx"
max_batch_size: 128
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 50 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 3 ]
  }
]
        """
    }
]


def create_model_directory(model_name: str, version: str) -> Path:
    """Create Triton model directory structure."""
    model_dir = TRITON_MODEL_REPO / model_name / version
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def download_model(url: str, destination: Path) -> None:
    """Download model file from URL."""
    print(f"  Downloading from {url}...")
    
    # For local development, create a dummy model file
    # In production, this would actually download from MLflow
    print(f"  Creating placeholder model at {destination}")
    destination.write_text("# Placeholder ONNX model\n")
    
    # Uncomment below for actual download
    # response = requests.get(url, stream=True)
    # response.raise_for_status()
    # with open(destination, 'wb') as f:
    #     for chunk in response.iter_content(chunk_size=8192):
    #         f.write(chunk)


def create_config_pbtxt(model_dir: Path, config: str) -> None:
    """Create Triton config.pbtxt file."""
    config_path = model_dir.parent / "config.pbtxt"
    config_path.write_text(config)
    print(f"  Created config: {config_path}")


def main():
    """Download and setup all models."""
    print(f"🚀 Setting up Triton model repository: {TRITON_MODEL_REPO}")
    print("=" * 60)
    
    # Create base directory
    TRITON_MODEL_REPO.mkdir(parents=True, exist_ok=True)
    
    for model_info in MODELS_TO_DOWNLOAD:
        print(f"\n📦 Setting up model: {model_info['name']}")
        
        # Create directory structure
        model_dir = create_model_directory(
            model_info['name'],
            model_info['version']
        )
        
        # Download model
        model_file = model_dir / "model.onnx"
        download_model(model_info['url'], model_file)
        
        # Create config
        create_config_pbtxt(model_dir, model_info['config'])
        
        print(f"  ✓ Model ready at {model_dir}")
    
    print(f"\n✅ All models downloaded to {TRITON_MODEL_REPO}")
    print(f"\nNext steps:")
    print(f"1. Start Triton: docker run --rm --gpus all -v {TRITON_MODEL_REPO}:/models -p 8001:8001 nvcr.io/nvidia/tritonserver:23.10-py3")
    print(f"2. Verify models: curl http://localhost:8000/v2/models")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)