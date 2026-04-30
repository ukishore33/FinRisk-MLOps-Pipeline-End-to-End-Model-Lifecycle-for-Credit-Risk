"""FastAPI endpoint tests."""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schema import LoanApplicationRequest


client = TestClient(app)


class TestPredictEndpoint:
    """Test /predict endpoint."""
    
    @pytest.fixture
    def valid_application(self):
        """Valid loan application."""
        return {
            "age": 35,
            "income_monthly": 50000,
            "loan_amount": 500000,
            "loan_tenure_m": 60,
            "interest_rate": 10.5,
            "existing_loans": 1,
            "credit_score": 750,
            "employment_type": "Salaried",
            "education": "Graduate",
            "residence_type": "Own",
            "delinquency_30d": 0,
            "utilisation_pct": 30,
            "months_employed": 24,
            "loan_purpose": "Home Improvement"
        }
    
    def test_predict_valid_input(self, valid_application):
        """Test prediction with valid input."""
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 200
        data = response.json()
        
        assert "default_probability" in data
        assert 0 <= data["default_probability"] <= 1
        assert data["predicted_class"] in [0, 1]
        assert 0 <= data["risk_score"] <= 100
    
    def test_predict_invalid_age(self, valid_application):
        """Test prediction with invalid age."""
        valid_application["age"] = 150
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 422
    
    def test_predict_negative_income(self, valid_application):
        """Test prediction with negative income."""
        valid_application["income_monthly"] = -1000
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 422
    
    def test_predict_invalid_employment_type(self, valid_application):
        """Test prediction with invalid employment type."""
        valid_application["employment_type"] = "InvalidType"
        response = client.post("/predict", json=valid_application)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Test /batch-predict endpoint."""
    
    def test_batch_predict(self):
        """Test batch prediction."""
        batch_request = {
            "request_id": "batch-001",
            "loans": [
                {
                    "age": 35, "income_monthly": 50000, "loan_amount": 500000,
                    "loan_tenure_m": 60, "interest_rate": 10.5, "existing_loans": 1,
                    "credit_score": 750, "employment_type": "Salaried",
                    "education": "Graduate", "residence_type": "Own",
                    "delinquency_30d": 0, "utilisation_pct": 30,
                    "months_employed": 24, "loan_purpose": "Home Improvement"
                },
                {
                    "age": 28, "income_monthly": 35000, "loan_amount": 300000,
                    "loan_tenure_m": 48, "interest_rate": 12.0, "existing_loans": 0,
                    "credit_score": 680, "employment_type": "Self-Employed",
                    "education": "Professional", "residence_type": "Rented",
                    "delinquency_30d": 1, "utilisation_pct": 45,
                    "months_employed": 12, "loan_purpose": "Vehicle"
                }
            ]
        }
        
        response = client.post("/batch-predict", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_records"] == 2
        assert data["successful_predictions"] == 2
        assert len(data["predictions"]) == 2


class TestHealthCheckEndpoint:
    """Test /health endpoint."""
    
    def test_health_check(self):
        """Test health check."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ["healthy", "degraded"]


class TestMetricsEndpoint:
    """Test /metrics endpoint."""
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus format
        assert b"HELP" in response.content or b"TYPE" in response.content