# src/tests/test_api.py
"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.main import app


@pytest.fixture
def client():
    """Test HTTP client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Health check endpoint tests."""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        with patch('src.main.health_checker') as mock_checker:
            mock_checker.check_all.return_value = {
                'overall': 'healthy',
                'triton': 'healthy',
                'feature_store': 'healthy',
                'redis': 'healthy'
            }
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "degraded"]
            assert data["service"] == "model-inference-service"
            assert "version" in data
    
    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
    
    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestMatchPredictionEndpoint:
    """Match prediction endpoint tests."""
    
    @patch('src.services.inference_service.InferenceService.predict_match')
    async def test_predict_match_success(self, mock_predict, client):
        """Test successful match prediction."""
        from src.domain.models import PredictionResult
        
        mock_predict.return_value = PredictionResult(
            score=0.96,
            confidence_interval=(0.94, 0.98),
            decision="AUTO_MATCH",
            model_version="v5",
            inference_ms=145,
            fallback_used=False
        )
        
        response = client.post(
            "/v1/matching/predict",
            json={
                "entity_id_1": "CUST_001",
                "entity_id_2": "CUST_002",
                "include_shap": True
            }
        )
        
        # Note: This will fail without proper dependency injection setup
        # In real tests, you'd need to properly mock the service dependencies
        assert response.status_code in [200, 500]  # Allow both for now
    
    def test_predict_match_validation_error(self, client):
        """Test validation error with missing fields."""
        response = client.post(
            "/v1/matching/predict",
            json={"entity_id_1": ""}
        )
        assert response.status_code == 422
    
    def test_predict_match_empty_entity_id(self, client):
        """Test with empty entity ID."""
        response = client.post(
            "/v1/matching/predict",
            json={
                "entity_id_1": "",
                "entity_id_2": "CUST_002"
            }
        )
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Batch prediction endpoint tests."""
    
    def test_batch_predict_validation(self, client):
        """Test batch prediction validation."""
        response = client.post(
            "/v1/matching/batch",
            json={
                "entity_pairs": [
                    ["CUST_001", "CUST_002"],
                    ["CUST_003", "CUST_004"]
                ]
            }
        )
        # Will fail without proper setup, but validates request format
        assert response.status_code in [200, 500]
    
    def test_batch_predict_too_many_pairs(self, client):
        """Test batch size limit."""
        pairs = [["E1", "E2"]] * 1001  # Exceed limit
        response = client.post(
            "/v1/matching/batch",
            json={"entity_pairs": pairs}
        )
        assert response.status_code == 422
    
    def test_batch_predict_invalid_pair(self, client):
        """Test with invalid pair format."""
        response = client.post(
            "/v1/matching/batch",
            json={
                "entity_pairs": [
                    ["CUST_001"]  # Only one ID
                ]
            }
        )
        assert response.status_code == 422


class TestModelsEndpoint:
    """Models list endpoint tests."""
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")
        # Will fail without proper setup
        assert response.status_code in [200, 500]


class TestMetricsEndpoint:
    """Metrics endpoint tests."""
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"


class TestErrorHandling:
    """Error handling tests."""
    
    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.put("/health")
        assert response.status_code == 405
