from fastapi.testclient import TestClient
from src.app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_and_latency():
    payload = {
        "transaction_amount": 520.0,
        "transaction_hour": 3,
        "distance_from_home": 110.0,
        "ratio_to_median_purchase_price": 4.8,
        "used_chip": 0,
        "used_pin_number": 0,
        "online_order": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["latency_ms"] < 500.0


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_requests" in response.json()
