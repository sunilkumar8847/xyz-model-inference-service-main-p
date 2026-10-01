# API Reference - Model Inference Service

Base URL: `http://localhost:8000` (local) or `https://api.xyzmdm.com` (production)

## Authentication

All endpoints except `/health` require JWT authentication:
```bash
curl -H "Authorization: Bearer <jwt_token>" \
  http://localhost:8000/v1/matching/predict
```

## Endpoints

### POST /v1/matching/predict

Predict if two entities are duplicates (matches).

**Request:**
```json
{
  "entity_id_1": "CUST_001",
  "entity_id_2": "CUST_002",
  "model_version": "v5",        // optional
  "include_shap": true,          // optional, default: true
  "timeout_ms": 200              // optional, default: 200
}
```

**Response (200 OK):**
```json
{
  "score": 0.96,
  "confidence_interval": [0.94, 0.98],
  "decision": "AUTO_MATCH",
  "model_version": "v5",
  "inference_ms": 145,
  "explanation": {
    "field_contributions": {
      "email": {"contribution": 0.35, "similarity": 1.0},
      "name": {"contribution": 0.28, "similarity": 0.94}
    },
    "top_k_fields": ["email", "name", "phone"],
    "human_readable": "High confidence match: email exact, name 94% similar"
  },
  "fallback_used": false
}
```

**Decision Values:**
- `AUTO_MATCH` (≥0.95): Auto-merge
- `LIKELY_MATCH` (0.85-0.95): Human review
- `UNCERTAIN` (0.50-0.85): Expert review
- `LIKELY_NON` (0.15-0.50): Verification
- `AUTO_NON_MATCH` (<0.15): Reject

---

### POST /v1/matching/batch

Batch predict matches for up to 1000 entity pairs.

**Request:**
```json
{
  "entity_pairs": [
    ["CUST_001", "CUST_002"],
    ["CUST_003", "CUST_004"]
  ],
  "timeout_ms": 2000
}
```

**Response (200 OK):**
```json
{
  "results": [
    {
      "score": 0.96,
      "decision": "AUTO_MATCH",
      ...
    }
  ],
  "total": 2
}
```

---

### GET /v1/models

List available models.

**Response (200 OK):**
```json
{
  "models": {
    "v5": {
      "name": "customer_matcher",
      "version": "v5",
      "framework": "ONNX",
      "f1_score": 0.947,
      "status": "ready",
      "loaded_at": "2025-01-18T10:30:00Z",
      "rollout_pct": 100
    }
  },
  "champion_version": "v5",
  "challenger_version": null,
  "ab_test_enabled": false
}
```

---

### GET /health

Health check endpoint.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "model-inference-service",
  "version": "3.0.0",
  "timestamp": "2025-01-18T10:30:00Z",
  "triton_status": "healthy",
  "feature_store_status": "healthy",
  "redis_status": "healthy"
}
```

---

### GET /metrics

Prometheus metrics endpoint.

**Response (200 OK - text/plain):**
```
# HELP inference_latency_ms Inference latency in milliseconds
# TYPE inference_latency_ms histogram
inference_latency_ms_bucket{le="50"} 1234
inference_latency_ms_sum 145678
inference_latency_ms_count 5000
...
```

## Error Responses

**Validation Error (422):**
```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [
    {
      "loc": ["body", "entity_id_1"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Timeout Error (504):**
```json
{
  "error_code": "INFERENCE_TIMEOUT",
  "message": "Inference exceeded timeout (200ms)"
}
```

**Service Unavailable (503):**
```json
{
  "error_code": "CIRCUIT_BREAKER_OPEN",
  "message": "Circuit breaker for 'triton' is open"
}
```

## Rate Limits

- Bronze: 100 req/sec
- Silver: 1,000 req/sec
- Gold: 10,000 req/sec

Rate limit headers included in response:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1705581600
```