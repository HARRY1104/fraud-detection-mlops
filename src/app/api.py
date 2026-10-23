from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.responses import HTMLResponse
from pydantic import BaseModel

from src.data.validate import TransactionSchema
from src.models.predict import FraudPredictor

service_state = {
    "predictor": None,
    "total_requests": 0,
    "flagged_fraud_count": 0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_state["predictor"] = FraudPredictor()
    yield
    service_state["predictor"] = None


app = FastAPI(
    title="Fraud Detection MLOps Service",
    description="Real-time fraud scoring API with schema validation, metrics, and endpoints",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionOut(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_tier: str
    latency_ms: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": service_state["predictor"] is not None,
    }


@app.get("/metrics")
def get_metrics():
    total = service_state["total_requests"]
    avg_lat = round(service_state["total_latency_ms"] / total, 2) if total > 0 else 0.0
    return {
        "total_requests": total,
        "flagged_fraud_count": service_state["flagged_fraud_count"],
        "avg_latency_ms": avg_lat,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(tx: TransactionSchema):
    if service_state["predictor"] is None:
        service_state["predictor"] = FraudPredictor()

    result = service_state["predictor"].predict_single(tx.model_dump())
    service_state["total_requests"] += 1
    service_state["total_latency_ms"] += result["latency_ms"]
    if result["is_fraud"]:
        service_state["flagged_fraud_count"] += 1
    return result


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><title>Fraud Detection MLOps</title></head>
    <body style="font-family: sans-serif; padding: 2rem; background: #0f172a; color: #f8fafc;">
        <h2>Fraud Detection MLOps API</h2>
        <p>API is running. Access <a href="/docs" style="color: #38bdf8;">/docs</a> for Swagger UI or <a href="/metrics" style="color: #38bdf8;">/metrics</a> for metrics.</p>
        <p>To run the interactive Streamlit dashboard, run: <code>streamlit run src/app/main.py</code></p>
    </body>
    </html>
    """
