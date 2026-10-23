import pandas as pd
from src.models.predict import FraudPredictor


def test_fraud_predictor_single():
    predictor = FraudPredictor()
    payload = {
        "transaction_amount": 500.0,
        "transaction_hour": 3,
        "distance_from_home": 100.0,
        "ratio_to_median_purchase_price": 5.0,
        "used_chip": 0,
        "used_pin_number": 0,
        "online_order": 1,
    }
    result = predictor.predict_single(payload)
    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert "risk_tier" in result
    assert "latency_ms" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_fraud_predictor_batch():
    predictor = FraudPredictor()
    df = pd.DataFrame(
        [
            {
                "transaction_amount": 10.0,
                "transaction_hour": 12,
                "distance_from_home": 2.0,
                "ratio_to_median_purchase_price": 0.8,
                "used_chip": 1,
                "used_pin_number": 1,
                "online_order": 0,
            },
            {
                "transaction_amount": 1500.0,
                "transaction_hour": 2,
                "distance_from_home": 250.0,
                "ratio_to_median_purchase_price": 10.0,
                "used_chip": 0,
                "used_pin_number": 0,
                "online_order": 1,
            },
        ]
    )
    scored_df, latency_ms = predictor.predict_batch(df)
    assert len(scored_df) == 2
    assert "fraud_probability" in scored_df.columns
    assert "is_fraud" in scored_df.columns
    assert "risk_tier" in scored_df.columns
    assert latency_ms > 0.0


def test_fraud_predictor_explain():
    predictor = FraudPredictor()
    suspicious_payload = {
        "transaction_amount": 1200.0,
        "transaction_hour": 3,
        "distance_from_home": 120.0,
        "ratio_to_median_purchase_price": 6.5,
        "used_chip": 0,
        "used_pin_number": 0,
        "online_order": 1,
    }
    factors = predictor.explain_transaction(suspicious_payload)
    assert len(factors) > 0
    factor_names = [f["factor"] for f in factors]
    assert "Extreme Spend Ratio" in factor_names
    assert "Unusual Transaction Hour" in factor_names
