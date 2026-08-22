# placeholder
import pytest
from fastapi.testclient import TestClient
from src.inference.serve import app

client = TestClient(app)

def test_health_endpoint():
    """Test if the /health endpoint returns HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint_valid_input():
    """Test if /predict endpoint accepts valid telemetry data and returns predictions."""
    payload = {
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data