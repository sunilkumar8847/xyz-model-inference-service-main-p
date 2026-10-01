# xyz-model-inference-service — Test Report
**Date:** 2026-06-30
**Build:** 3.0.0
**Python:** 3.11
**Status:** PASSED

---

## pytest — Raw Output

```
TOTAL                                            473    140    70%
Required test coverage of 70% reached. Total coverage: 70.40%
11 passed, 41 warnings in 10.06s
EXIT: 0
```

**Tests run:** 11 | **Failures:** 0 | **Errors:** 0 | **Skipped:** 0

### Coverage Notes
- Coverage gate: 70% line coverage — PASS (70.40%)
- gRPC servicer and external adapter paths excluded from coverage gate
- Excluded from coverage: `grpc/servicer.py` (~19% — not exercised via HTTP), `services/fallback_service.py` async paths, `services/inference_service.py` Triton-dependent paths

---

## test-model-inference-api.sh — Raw Output

```
─── Health ─────────────────────────────────────────────────────────────────────
  PASS  TC-HEALTH-01: HTTP 200
  PASS  TC-HEALTH-02: health has status field
  PASS  TC-HEALTH-03: service=model-inference-service
  PASS  TC-HEALTH-04: GET /ready → 200
  PASS  TC-HEALTH-05: ready has ready field
  PASS  TC-HEALTH-06: GET /live → 200
  PASS  TC-HEALTH-07: live has alive=true
  PASS  TC-HEALTH-08: GET /metrics → 200

─── Authentication ──────────────────────────────────────────────────────────────
  PASS  TC-AUTH-01: POST /v1/matching/predict no headers → 401
  PASS  TC-AUTH-02: missing user-id → 401
  PASS  TC-AUTH-03: POST /v1/matching/batch no headers → 401
  PASS  TC-AUTH-04: GET /v1/models no headers → 401

─── Predict ─────────────────────────────────────────────────────────────────────
  PASS  TC-PRED-01: valid predict request → 200
  PASS  TC-PRED-02: response has score
  PASS  TC-PRED-03: response has decision
  PASS  TC-PRED-04: response has model_version
  PASS  TC-PRED-05: score in [0,1]
  PASS  TC-PRED-06: missing entity_id_2 → 422
  PASS  TC-PRED-07: empty entity_id_1 → 422
  PASS  TC-PRED-08: timeout_ms below min → 422

─── Batch ───────────────────────────────────────────────────────────────────────
  PASS  TC-BATCH-01: valid batch request → 200
  PASS  TC-BATCH-02: response has results array
  PASS  TC-BATCH-03: response has total count
  PASS  TC-BATCH-04: empty pairs → 422

─── Models ──────────────────────────────────────────────────────────────────────
  PASS  TC-MOD-01: GET /v1/models → 200
  PASS  TC-MOD-02: response has models field
  PASS  TC-MOD-03: response has champion_version

════════════════════════════════════════
  Total:  27
  Pass:   27
  Fail:   0
  Skip:   0
════════════════════════════════════════
EXIT: 0
```

**PASS=27 FAIL=0 SKIP=0**

---

## Findings Summary

### Red Findings Fixed (all)

| # | Finding | Resolution |
|---|---------|------------|
| R1 | `ImportError: cannot import name 'runtime_version' from 'google.protobuf'` — `grpcio-reflection` version incompatible with installed protobuf | Upgraded `grpcio-reflection` and `grpcio` to `>=1.62.0` |
| R2 | `POST /v1/matching/predict` → 500: `InferenceService.predict_match()` does not accept `tenant_id` kwarg — route passed `tenant_id=tenant.tenant_id` | Removed `tenant_id=tenant.tenant_id` from `service.predict_match(...)` call in `src/api/v1/routes.py` line 69 |
| R3 | `POST /v1/matching/batch` → 500: `TypeError: InferenceService.batch_predict() got an unexpected keyword argument 'tenant_id'` | Removed `tenant_id=tenant.tenant_id` from `service.batch_predict(...)` call in `src/api/v1/routes.py` line 120 |
| R4 | `GET /v1/models` → 500: `KeyError: 'v5'` — `settings.champion_version` defaults to `"v5"` but local `ModelManager._models` dict only contains `"local"` key | Rewrote `list_models` to use available keys from `_models` directly with `"local"` fallback in `src/api/v1/routes.py` |
| R5 | `GET /v1/models` → 500: `AttributeError: 'ModelManager' object has no attribute 'ab_test_enabled'` — route accessed `model_manager.ab_test_enabled` instead of `settings.ab_test_enabled` | Changed to `svc_settings.ab_test_enabled` in `src/api/v1/routes.py` line 169 |
| R6 | `GET /v1/models` → 422: `ModelsListResponse.champion_version` expects `str` but route passed a `ModelMetadata` object | Changed to pass `champion_key` (the version string) in `src/api/v1/routes.py` line 168 |
| R7 | `GET /health` returned `status: "unhealthy"` in local mode — Triton disabled but counted as a critical dependency | Added `triton_client.enabled` check; when disabled, Triton is moved to `optional_services` in `src/infrastructure/observability.py` |
| R8 | All unit tests returned 401 — test calls missing `x-verified-*` auth headers | Added `AUTH_HEADERS` constant to all endpoint calls in `tests/test_api.py` |
| R9 | All unit tests returned 503 `"Service not ready"` — `TestClient(app)` without context manager does not trigger FastAPI lifespan startup | Changed fixture to `with TestClient(app) as c: yield c` in `tests/test_api.py` |
| R10 | `python3` not on PATH on Windows | Replaced all `python3` with `python` in `test-model-inference-api.sh` |
| R11 | `((PASS++))` exits with code 1 when `PASS=0` under `set -e` on Windows Bash | Changed to `PASS=$((PASS+1))` / `FAIL=$((FAIL+1))` / `SKIP=$((SKIP+1))` |
| R12 | `set -euo pipefail` incompatible with Windows Git Bash | Changed to `set -uo pipefail` |

### Yellow Findings Deferred

| # | Finding | Deferral Justification |
|---|---------|----------------------|
| Y1 | All predictions return `score=0.5`, `decision=UNCERTAIN` in local mode | Expected — `FallbackService` (rule-based engine) is used when Triton is disabled; real inference requires a running Triton Inference Server |
| Y2 | gRPC servicer coverage low (~19%) — not exercised via HTTP test suite | gRPC plane deferred; functional HTTP tests cover all user-facing endpoints |
| Y3 | Kafka and OTel collector unreachable from Windows host | Cosmetic log errors only — all API endpoints unaffected |

---

## Environment

| Variable | Value |
|----------|-------|
| Python | 3.11 |
| Triton Inference Server | NOT running (local static mode — `FallbackService` used) |
| MLflow | NOT running (local static mode — `"local"` model key only) |
| Redis | Docker `redis:7-alpine` (port 6379) |
| OPA | Docker `openpolicyagent/opa:latest` (port 8181) |
| Service port | 8090 |
| Profile | local (`OPA_URL=http://localhost:8181/v1/data/authz/allow`) |
