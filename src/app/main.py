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
    description="Real-time fraud scoring API with schema validation, metrics, and interactive UI",
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
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Fraud Detection MLOps</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen p-8">
        <div class="max-w-4xl mx-auto">
            <div class="flex items-center justify-between border-b border-slate-800 pb-5 mb-8">
                <div>
                    <h1 class="text-2xl font-bold text-white">Real-Time Fraud Detection MLOps</h1>
                    <p class="text-slate-400 text-sm">FastAPI Service &bull; <code class="text-blue-400">src/app/main.py</code></p>
                </div>
                <div class="space-x-2">
                    <a href="/metrics" class="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-2 rounded border border-slate-700">/metrics</a>
                    <a href="/docs" class="text-xs bg-blue-600 hover:bg-blue-500 px-3 py-2 rounded text-white">/docs</a>
                </div>
            </div>
            <div class="grid md:grid-cols-2 gap-8">
                <form id="fraudForm" class="bg-slate-900 p-6 rounded-xl border border-slate-800 space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label class="text-xs text-slate-400">Amount ($)</label>
                            <input type="number" step="0.01" id="transaction_amount" value="480.00" class="w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400">Hour (0-23)</label>
                            <input type="number" min="0" max="23" id="transaction_hour" value="2" class="w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400">Distance From Home (km)</label>
                            <input type="number" step="0.1" id="distance_from_home" value="95.0" class="w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400">Ratio to Median Spend</label>
                            <input type="number" step="0.1" id="ratio_to_median_purchase_price" value="4.5" class="w-full mt-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
                        </div>
                    </div>
                    <div class="grid grid-cols-3 gap-3 pt-2">
                        <label class="flex items-center space-x-2 text-sm bg-slate-800 p-2.5 rounded border border-slate-700">
                            <input type="checkbox" id="used_chip"> <span>Chip</span>
                        </label>
                        <label class="flex items-center space-x-2 text-sm bg-slate-800 p-2.5 rounded border border-slate-700">
                            <input type="checkbox" id="used_pin_number"> <span>PIN</span>
                        </label>
                        <label class="flex items-center space-x-2 text-sm bg-slate-800 p-2.5 rounded border border-slate-700">
                            <input type="checkbox" id="online_order" checked> <span>Online</span>
                        </label>
                    </div>
                    <button type="submit" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg transition">
                        Score Transaction
                    </button>
                </form>
                <div class="bg-slate-900 p-6 rounded-xl border border-slate-800 flex flex-col justify-between">
                    <div>
                        <span class="text-xs uppercase tracking-wider text-slate-400">Prediction Verdict</span>
                        <div id="badge" class="mt-4 inline-block px-3 py-1 rounded-full text-xs font-bold bg-slate-800 text-slate-300">
                            Ready for inference
                        </div>
                        <div class="mt-6">
                            <div class="text-4xl font-extrabold" id="probText">--%</div>
                            <div class="text-xs text-slate-400 mt-1">Fraud Probability</div>
                        </div>
                    </div>
                    <div class="border-t border-slate-800 pt-4 text-xs text-slate-400 flex justify-between">
                        <span>Threshold: 50.0%</span>
                        <span id="latencyText">Latency: -- ms</span>
                    </div>
                </div>
            </div>
        </div>
        <script>
            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const payload = {
                    transaction_amount: parseFloat(document.getElementById('transaction_amount').value),
                    transaction_hour: parseInt(document.getElementById('transaction_hour').value),
                    distance_from_home: parseFloat(document.getElementById('distance_from_home').value),
                    ratio_to_median_purchase_price: parseFloat(document.getElementById('ratio_to_median_purchase_price').value),
                    used_chip: document.getElementById('used_chip').checked ? 1 : 0,
                    used_pin_number: document.getElementById('used_pin_number').checked ? 1 : 0,
                    online_order: document.getElementById('online_order').checked ? 1 : 0,
                };
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                document.getElementById('probText').innerText = (data.fraud_probability * 100).toFixed(1) + '%';
                document.getElementById('latencyText').innerText = `Latency: ${data.latency_ms} ms`;
                const badge = document.getElementById('badge');
                badge.innerText = `${data.risk_tier} (${data.is_fraud ? 'BLOCK' : 'APPROVE'})`;
                badge.className = data.is_fraud
                    ? 'mt-4 inline-block px-3 py-1 rounded-full text-xs font-bold bg-red-900/60 text-red-300 border border-red-700'
                    : 'mt-4 inline-block px-3 py-1 rounded-full text-xs font-bold bg-emerald-900/60 text-emerald-300 border border-emerald-700';
            });
        </script>
    </body>
    </html>
    """
