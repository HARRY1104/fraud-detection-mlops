from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from src.data.person_store import (
    get_person_data,
    list_all_persons,
    seed_sample_persons,
    store_person_data,
    store_person_transaction,
)
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
    seed_sample_persons()
    yield
    service_state["predictor"] = None


app = FastAPI(
    title="Fraud Detection MLOps Service",
    description="Real-time fraud scoring API with schema validation, metrics, cardholder profiles, and endpoints",
    version="1.1.0",
    lifespan=lifespan,
)


class PredictionOut(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_tier: str
    latency_ms: float


class PersonProfileIn(BaseModel):
    person_id: str = Field(..., description="Unique cardholder/person identifier")
    name: str = Field(..., description="Full name")
    median_purchase_price: float = Field(50.0, gt=0, description="Baseline median spend in USD")
    home_address: Optional[str] = Field("Home", description="Home city or location")
    email: Optional[str] = Field("", description="Email address")
    phone: Optional[str] = Field("", description="Phone number")
    risk_status: Optional[str] = Field("ACTIVE", description="Account status: ACTIVE, UNDER_REVIEW, BLOCKED")


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


# =============================================================================
# Individual Person / Cardholder Profile Endpoints
# =============================================================================
@app.get("/persons")
def get_persons():
    """Lists all stored individual persons/cardholders with spending & fraud stats."""
    return list_all_persons()


@app.post("/persons")
def create_or_update_person(profile: PersonProfileIn):
    """Stores or updates an individual person's profile in the database."""
    stored = store_person_data(
        person_id=profile.person_id,
        name=profile.name,
        median_purchase_price=profile.median_purchase_price,
        home_address=profile.home_address or "Home",
        email=profile.email or "",
        phone=profile.phone or "",
        risk_status=profile.risk_status or "ACTIVE",
    )
    return stored


@app.get("/persons/{person_id}")
def get_person(person_id: str):
    """Retrieves an individual person's profile and full transaction ledger."""
    person = get_person_data(person_id, include_transactions=True)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")
    return person


@app.post("/persons/{person_id}/predict")
def predict_and_store_for_person(person_id: str, tx: TransactionSchema):
    """Scores a transaction and stores the transaction under the individual person's record."""
    if service_state["predictor"] is None:
        service_state["predictor"] = FraudPredictor()

    person = get_person_data(person_id, include_transactions=False)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found.")

    tx_dict = tx.model_dump()
    result = service_state["predictor"].predict_single(tx_dict)

    combined = {**tx_dict, **result}
    saved_tx = store_person_transaction(person_id, combined)

    service_state["total_requests"] += 1
    service_state["total_latency_ms"] += result["latency_ms"]
    if result["is_fraud"]:
        service_state["flagged_fraud_count"] += 1

    return {"prediction": result, "stored_transaction": saved_tx}


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><title>Fraud Detection MLOps</title></head>
    <body style="font-family: sans-serif; padding: 2rem; background: #0f172a; color: #f8fafc;">
        <h2>Fraud Detection MLOps API</h2>
        <p>API is running. Access <a href="/docs" style="color: #38bdf8;">/docs</a> for Swagger UI or <a href="/metrics" style="color: #38bdf8;">/metrics</a> for metrics.</p>
        <p>To run the interactive Streamlit dashboard, run: <code>streamlit run app.py</code></p>
    </body>
    </html>
    """
