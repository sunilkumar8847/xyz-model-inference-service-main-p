# Model Inference Service Documentation

Complete documentation for XYZ MDM 3.0 Model Inference Service.

## Quick Links

- [Getting Started](LOCAL_SETUP.md) - Local development setup
- [API Reference](API.md) - REST API endpoints
- [Architecture](ARCHITECTURE.md) - Design decisions and patterns

## Overview

The Model Inference Service provides real-time AI-powered entity matching with:

- **Sub-200ms Latency**: P95 inference latency < 200ms
- **98.5% Accuracy**: F1 score on production data
- **High Availability**: 99.999% uptime with automatic failover
- **Resilience**: Circuit breakers, timeouts, rule-based fallback
- **Observability**: Prometheus metrics, structured logging, distributed tracing

## Architecture
```
┌─────────────────────────────────────┐
│   FastAPI Application               │
├─────────────────────────────────────┤
│ Service Layer                       │
│ ├─ InferenceService                 │
│ ├─ ModelManager                     │
│ └─ FallbackService                  │
├─────────────────────────────────────┤
│ Adapter Layer                       │
│ ├─ TritonClient (gRPC)              │
│ ├─ FeatureStoreClient (REST)        │
│ ├─ MLflowClient (REST)              │
│ └─ RedisCache                       │
└─────────────────────────────────────┘
```

## Key Features

### AI Inference
- Ensemble models (Transformer + GNN + XGBoost)
- SHAP explanations for every prediction
- Confidence intervals
- A/B testing support

### Performance
- Multi-tier caching (Redis + in-process)
- Batch processing
- GPU acceleration via Triton
- Async/await for concurrency

### Resilience
- Circuit breakers on external calls
- Automatic timeouts (200ms default)
- Rule-based fallback when AI unavailable
- Graceful degradation

### Observability
- Prometheus metrics export
- Structured JSON logging
- OpenTelemetry distributed tracing
- Health check endpoints

## Development

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for detailed setup instructions.

Quick start:
```bash
poetry install
cp .env.example .env.local
docker-compose up -d
poetry shell
uvicorn src.main:app --reload
```

## Deployment

### Kubernetes
```bash
kubectl apply -f deployments/kubernetes/
```

### Docker
```bash
docker build -t xyzmdm/model-inference-service:3.0.0 .
docker run -p 8000:8000 xyzmdm/model-inference-service:3.0.0
```

## Monitoring

Access metrics at `/metrics` endpoint in Prometheus format.

Key metrics:
- `inference_latency_ms` - Inference duration histogram
- `inference_requests_total` - Total request count
- `inference_errors_total` - Error count by type
- `cache_hit_rate` - Cache hit rate percentage

## Support

For issues or questions, contact the XYZ MDM Engineering team.