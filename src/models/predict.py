import time
from pathlib import Path
import joblib
import pandas as pd
from src.models.train import MODEL_PATH, train_model


class FraudPredictor:
    """Production inference wrapper for the serialized fraud detection pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            train_model()
        self.pipeline = joblib.load(self.model_path)

    def predict_single(self, payload: dict) -> dict:
        start = time.perf_counter()
        df = pd.DataFrame([payload])
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        if prob >= 0.70:
            risk_tier = "HIGH RISK"
        elif prob >= 0.35:
            risk_tier = "MEDIUM RISK"
        else:
            risk_tier = "LOW RISK"

        return {
            "is_fraud": prob >= 0.50,
            "fraud_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "latency_ms": latency_ms,
        }
