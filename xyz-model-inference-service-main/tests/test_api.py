# src/tests/test_api.py
"""API endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    """Test HTTP client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Health check endpoint tests."""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["service"] == "model-inference-service"
        assert "version" in data


class TestMatchPredictionEndpoint:
    """Match prediction endpoint tests."""
    
    def test_predict_match_success(self, client):
        """Test successful match prediction."""
        response = client.post(
            "/v1/matching/predict",
            json={
                "entity_id_1": "CUST_001",
                "entity_id_2": "CUST_002",
                "include_shap": True
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["score"] <= 1
        assert "decision" in data
        assert "confidence_interval" in data
    
    def test_predict_match_invalid_input(self, client):
        """Test with invalid input."""
        response = client.post(
            "/v1/matching/predict",
            json={"entity_id_1": ""}
        )
        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Batch prediction endpoint tests."""
    
    def test_batch_predict_success(self, client):
        """Test batch prediction."""
        response = client.post(
            "/v1/matching/batch",
            json={
                "entity_pairs": [
                    ["CUST_001", "CUST_002"],
                    ["CUST_003", "CUST_004"]
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["total"] == 2


class TestModelsEndpoint:
    """Models list endpoint tests."""
    
    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "champion_version" in data
        assert "ab_test_enabled" in data