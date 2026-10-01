#!/usr/bin/env bash
# Functional smoke tests for xyz-model-inference-service (port 8090)
# Usage: ./test-model-inference-service-api.sh [BASE_URL]
# Exit code: 0 = all passed, 1 = one or more failures

BASE_URL="${1:-http://localhost:8090}"
PASS=0; FAIL=0; SKIP=0
TENANT_ID="00000000-0000-0000-0000-000000000001"
USER_ID="00000000-0000-0000-0000-000000000002"

AUTH_HEADERS=(
  -H "x-verified-tenant-id: ${TENANT_ID}"
  -H "x-verified-user-id: ${USER_ID}"
  -H "x-verified-roles: PLATFORM_ADMIN"
  -H "x-verified-platform: true"
)

log_pass() { PASS=$((PASS+1)); echo "  ✅ $1" >&2; }
log_fail() { FAIL=$((FAIL+1)); echo "  ❌ $1" >&2; }
log_skip() { SKIP=$((SKIP+1)); echo "  ⏭  $1" >&2; }
log_section() { echo "" >&2; echo "=== $1 ===" >&2; }

# `call` writes the response body to $BODY_FILE instead of stdout. Capturing it with
# BODY=$(call ...) ran call in a subshell, so its PASS/FAIL increments were lost.
BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT

call() {
  local desc="$1" expected_status="$2"; shift 2
  local response; response=$(curl -s -w "\n%{http_code}" "$@")
  local body; body=$(echo "$response" | head -n -1)
  local status_code; status_code=$(echo "$response" | tail -1)
  if [ "$status_code" = "$expected_status" ]; then
    log_pass "[$status_code] $desc"
  else
    log_fail "[$status_code ≠ $expected_status] $desc — body: $(echo "$body" | head -c 200)"
  fi
  printf '%s' "$body" > "$BODY_FILE"
}

log_section "TC-HEALTH — Health & Readiness"
call "TC-HEALTH-01: GET /health returns 200" 200 "${BASE_URL}/health"
call "TC-HEALTH-02: GET /health has status field" 200 "${BASE_URL}/health"
HEALTH_BODY=$(cat "$BODY_FILE")
if echo "${HEALTH_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'status' in d" 2>/dev/null; then
  log_pass "TC-HEALTH-02b: health response has status field"
else
  log_fail "TC-HEALTH-02b: health response missing status field"
fi
call "TC-HEALTH-03: GET /ready returns 200" 200 "${BASE_URL}/ready"
call "TC-HEALTH-04: GET /live returns 200" 200 "${BASE_URL}/live"
call "TC-HEALTH-05: GET /metrics returns 200" 200 "${BASE_URL}/metrics"

log_section "TC-AUTH — Authentication enforcement"
call "TC-AUTH-01: POST /v1/matching/predict no headers → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d '{"entity_id_1":"CUST_001","entity_id_2":"CUST_002"}' \
  "${BASE_URL}/v1/matching/predict"
call "TC-AUTH-02: POST /v1/matching/predict missing user-id → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -H "x-verified-tenant-id: ${TENANT_ID}" \
  -H "x-verified-roles: PLATFORM_ADMIN" \
  -d '{"entity_id_1":"CUST_001","entity_id_2":"CUST_002"}' \
  "${BASE_URL}/v1/matching/predict"
call "TC-AUTH-03: POST /v1/matching/batch no headers → 401" 401 \
  -X POST -H "Content-Type: application/json" \
  -d '{"entity_pairs":[["CUST_001","CUST_002"]]}' \
  "${BASE_URL}/v1/matching/batch"
call "TC-AUTH-04: GET /v1/models no headers → 401" 401 \
  "${BASE_URL}/v1/models"

log_section "TC-PREDICT — Match prediction"
call "TC-PRED-01: POST /v1/matching/predict valid → 200" 200 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_id_1":"CUST_001","entity_id_2":"CUST_002","include_shap":true}' \
  "${BASE_URL}/v1/matching/predict"
PRED_BODY=$(cat "$BODY_FILE")
if echo "${PRED_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 0<=d['score']<=1 and 'decision' in d and 'model_version' in d and 'confidence_interval' in d" 2>/dev/null; then
  log_pass "TC-PRED-02: response has score∈[0,1], decision, model_version, confidence_interval"
else
  log_fail "TC-PRED-02: response missing required fields"
fi
call "TC-PRED-03: POST /v1/matching/predict missing entity_id_2 → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_id_1":"CUST_001"}' \
  "${BASE_URL}/v1/matching/predict"
call "TC-PRED-04: POST /v1/matching/predict empty entity_id_1 → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_id_1":"","entity_id_2":"CUST_002"}' \
  "${BASE_URL}/v1/matching/predict"
call "TC-PRED-05: POST /v1/matching/predict timeout_ms too low → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_id_1":"CUST_001","entity_id_2":"CUST_002","timeout_ms":10}' \
  "${BASE_URL}/v1/matching/predict"

log_section "TC-BATCH — Batch prediction"
call "TC-BATCH-01: POST /v1/matching/batch valid → 200" 200 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_pairs":[["CUST_001","CUST_002"],["CUST_003","CUST_004"]]}' \
  "${BASE_URL}/v1/matching/batch"
BATCH_BODY=$(cat "$BODY_FILE")
if echo "${BATCH_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['results'])==2 and d['total']==2" 2>/dev/null; then
  log_pass "TC-BATCH-02: batch response has results[2] and total=2"
else
  log_fail "TC-BATCH-02: batch response incorrect"
fi
call "TC-BATCH-03: POST /v1/matching/batch empty pairs → 422" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_pairs":[]}' \
  "${BASE_URL}/v1/matching/batch"

log_section "TC-MODELS — Model list"
call "TC-MOD-01: GET /v1/models → 200" 200 \
  "${AUTH_HEADERS[@]}" "${BASE_URL}/v1/models"
MODELS_BODY=$(cat "$BODY_FILE")
if echo "${MODELS_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'models' in d and 'champion_version' in d and 'ab_test_enabled' in d" 2>/dev/null; then
  log_pass "TC-MOD-02: models response has models, champion_version, ab_test_enabled"
else
  log_fail "TC-MOD-02: models response missing required fields"
fi

log_section "TC-ERROR — Error response shape"
call "TC-ERR-01: validation error has trace_id and timestamp" 422 \
  -X POST "${AUTH_HEADERS[@]}" -H "Content-Type: application/json" \
  -d '{"entity_id_1":""}' \
  "${BASE_URL}/v1/matching/predict"
ERR_BODY=$(cat "$BODY_FILE")
if echo "${ERR_BODY}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'trace_id' in d and 'timestamp' in d and 'error_code' in d" 2>/dev/null; then
  log_pass "TC-ERR-02: error response has trace_id, timestamp, error_code"
else
  log_fail "TC-ERR-02: error response missing trace_id/timestamp/error_code"
fi

echo "" >&2
echo "================================" >&2
echo "PASS:${PASS}  FAIL:${FAIL}  SKIP:${SKIP}" >&2
echo "================================" >&2
[ "$FAIL" -eq 0 ]
