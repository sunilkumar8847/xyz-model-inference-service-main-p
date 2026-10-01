#!/usr/bin/env bash
# Regenerate gRPC Python stubs from inference.proto.
# Run after any proto change:  bash scripts/generate_proto.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

python -m grpc_tools.protoc \
  -I "$ROOT/src/grpc" \
  --python_out="$ROOT/src/grpc" \
  --grpc_python_out="$ROOT/src/grpc" \
  "$ROOT/src/grpc/inference.proto"

# Fix the generated import so it works as a Python package (not standalone)
sed -i 's/^import inference_pb2 as inference__pb2$/from src.grpc import inference_pb2 as inference__pb2/' \
  "$ROOT/src/grpc/inference_pb2_grpc.py"

echo "Proto stubs regenerated in src/grpc/"
