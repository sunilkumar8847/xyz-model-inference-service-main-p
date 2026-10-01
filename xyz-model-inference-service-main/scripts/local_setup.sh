# scripts/local_setup.sh
#!/bin/bash

set -e

echo "🚀 XYZ MDM 3.0 - Model Inference Service Local Setup"
echo "=================================================="

# Check prerequisites
echo "\n✓ Checking prerequisites..."
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11+ not found. Please install Python 3.11+"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker"
    exit 1
fi

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Installing poetry..."
    pip install poetry==1.7.0
fi

echo "✓ All prerequisites found"

# Install dependencies
echo "\n📦 Installing Python dependencies..."
poetry install

# Copy environment file
echo "\n⚙️  Setting up environment..."
if [ ! -f .env.local ]; then
    cp .env.example .env.local
    echo "✓ Created .env.local"
else
    echo "✓ .env.local already exists"
fi

# Start Docker services
echo "\n🐳 Starting Docker services..."
docker-compose up -d
echo "✓ Docker services started"

# Wait for services
echo "\n⏳ Waiting for services to be healthy (30 seconds)..."
sleep 30

# Verify services
echo "\n🔍 Verifying services..."
docker-compose ps

echo "\n✅ Setup complete!"
echo "\nNext steps:"
echo "1. Activate Poetry shell: poetry shell"
echo "2. Run service: uvicorn src.main:app --reload --port 8000"
echo "3. View API docs: http://localhost:8000/docs"
