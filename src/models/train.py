from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.make_dataset import load_or_create_dataset
from src.features.build_features import build_feature_pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"

RAW_FEATURE_COLUMNS = [
    "transaction_amount",
    "transaction_hour",
    "distance_from_home",
    "ratio_to_median_purchase_price",
    "used_chip",
    "used_pin_number",
    "online_order",
]


def train_model() -> dict:
    """Trains the fraud detection pipeline, logs to MLflow if available, and exports model artifact."""
    df = load_or_create_dataset()
    X = df[RAW_FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    }

    pipeline = Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_probs)), 4),
        "precision": round(float(precision_score(y_test, y_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_preds, zero_division=0)), 4),
    }

    try:
        import mlflow
        mlflow.set_tracking_uri(f"file://{ROOT_DIR / 'mlruns'}")
        mlflow.set_experiment("fraud-detection-mlops")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
    except Exception:
        pass

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} | Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    train_model()
