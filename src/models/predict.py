import time
from pathlib import Path
import joblib
import pandas as pd
from src.models.train import MODEL_PATH, train_model
import shap
import numpy as np

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

    def predict_batch(self, df: pd.DataFrame, threshold: float = 0.50) -> tuple[pd.DataFrame, float]:
        start = time.perf_counter()
        probs = self.pipeline.predict_proba(df)[:, 1]
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        scored_df = df.copy()
        scored_df["fraud_probability"] = [round(float(p), 4) for p in probs]
        scored_df["is_fraud"] = [bool(p >= threshold) for p in probs]
        scored_df["risk_tier"] = [
            "HIGH RISK" if p >= 0.70 else "MEDIUM RISK" if p >= 0.35 else "LOW RISK"
            for p in probs
        ]
        return scored_df, latency_ms

    def explain_transaction(self, payload: dict) -> list[dict]:
        factors = []
        amt = float(payload.get("transaction_amount", 0.0))
        hour = int(payload.get("transaction_hour", 12))
        dist = float(payload.get("distance_from_home", 0.0))
        ratio = float(payload.get("ratio_to_median_purchase_price", 1.0))
        used_chip = int(payload.get("used_chip", 0))
        used_pin = int(payload.get("used_pin_number", 0))
        online = int(payload.get("online_order", 0))

        if ratio >= 4.0:
            factors.append({
                "factor": "Extreme Spend Ratio",
                "severity": "High Risk",
                "desc": f"Transaction is {ratio:.1f}x normal median spending"
            })
        elif ratio >= 2.0:
            factors.append({
                "factor": "Elevated Spend Ratio",
                "severity": "Moderate Risk",
                "desc": f"Transaction is {ratio:.1f}x normal median spending"
            })

        if hour <= 5 or hour >= 23:
            factors.append({
                "factor": "Unusual Transaction Hour",
                "severity": "Moderate Risk",
                "desc": f"Occurred at {hour:02d}:00 (high-risk night window 23:00 - 05:00)"
            })

        if dist >= 100:
            factors.append({
                "factor": "Far from Home",
                "severity": "High Risk",
                "desc": f"{dist:.1f} km away from cardholder home location"
            })
        elif dist >= 40:
            factors.append({
                "factor": "Moderate Distance",
                "severity": "Moderate Risk",
                "desc": f"{dist:.1f} km away from home location"
            })

        if online and not used_pin and not used_chip:
            factors.append({
                "factor": "Unauthenticated Online Transaction",
                "severity": "High Risk",
                "desc": "Online purchase without chip or PIN verification"
            })

        if amt >= 1000:
            factors.append({
                "factor": "High Transaction Amount",
                "severity": "Moderate Risk",
                "desc": f"Total amount ${amt:,.2f} exceeds standard retail threshold"
            })

        if used_pin:
            factors.append({
                "factor": "PIN Authenticated",
                "severity": "Protective",
                "desc": "Cardholder entered secure PIN at terminal"
            })

        if used_chip:
            factors.append({
                "factor": "EMV Chip Verified",
                "severity": "Protective",
                "desc": "Physical cryptographic chip authenticated by terminal"
            })

        return factors

class ExplainerService:
    def __init__(self, model_pipeline):
        # Extract the fitted tree classifier from pipeline
        self.classifier = model_pipeline.named_steps["classifier"]
        self.preprocessor = model_pipeline.named_steps["features"]
        self.explainer = shap.TreeExplainer(self.classifier)

    def explain_payload(self, df_raw):
        transformed = self.preprocessor.transform(df_raw)
        shap_values = self.explainer.shap_values(transformed)
        
        # For binary classification, inspect the positive (fraud) class
        fraud_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        feature_names = self.preprocessor.get_feature_names_out()
        
        ranked_indices = np.argsort(np.abs(fraud_shap))[::-1]
        top_reasons = []
        for idx in ranked_indices[:3]:
            top_reasons.append({
                "feature": str(feature_names[idx]),
                "impact": round(float(fraud_shap[idx]), 4),
                "direction": "Risk Escalator" if fraud_shap[idx] > 0 else "Protective Factor",
            })
        return top_reasons