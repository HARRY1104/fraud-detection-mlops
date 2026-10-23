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


def test_persons_endpoints():
    profile = {
        "person_id": "TEST-100",
        "name": "Jane Doe",
        "median_purchase_price": 75.0,
        "home_address": "Seattle, WA",
        "email": "jane@example.com",
        "phone": "+1-555-9999",
        "risk_status": "ACTIVE",
    }
    # Create/store
    res = client.post("/persons", json=profile)
    assert res.status_code == 200
    assert res.json()["person_id"] == "TEST-100"

    # List
    list_res = client.get("/persons")
    assert list_res.status_code == 200
    ids = [p["person_id"] for p in list_res.json()]
    assert "TEST-100" in ids

    # Get single
    get_res = client.get("/persons/TEST-100")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Jane Doe"

    # Predict & store transaction for person
    tx_payload = {
        "transaction_amount": 120.0,
        "transaction_hour": 15,
        "distance_from_home": 5.0,
        "ratio_to_median_purchase_price": 1.6,
        "used_chip": 1,
        "used_pin_number": 1,
        "online_order": 0,
    }
    pred_res = client.post("/persons/TEST-100/predict", json=tx_payload)
    assert pred_res.status_code == 200
    assert "stored_transaction" in pred_res.json()
    assert pred_res.json()["stored_transaction"]["person_id"] == "TEST-100"
