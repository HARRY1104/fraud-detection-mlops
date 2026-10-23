"""Root entrypoint for Streamlit dashboard.

Allows running either:
    streamlit run app.py
or
    streamlit run src/app/main.py
"""

from pathlib import Path
import runpy

app_path = Path(__file__).resolve().parent / "src" / "app" / "main.py"
runpy.run_path(str(app_path), run_name="__main__")

import time
import random
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide",
)

# Load model pipeline
@st.cache_resource
def load_model():
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        from src.models.train import train_model
        train_model()
    return joblib.load(model_path)

pipeline = load_model()

# Sidebar navigation / mode selection
st.sidebar.title("🛡️ MLOps Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Real-Time Scoring", "Live Traffic Simulator"])

# ==========================================
# MODE 1: REAL-TIME SCORING (Single Transaction)
# ==========================================
if app_mode == "Real-Time Scoring":
    st.title("🛡️ Real-Time Transaction Risk Scoring")
    st.markdown("Evaluate individual transaction payloads against the trained ensemble model.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Transaction Parameters")
        
        # Quick Presets
        preset = st.selectbox(
            "⚡ Quick Presets",
            ["Custom", "Normal Coffee ($4.50)", "Daytime Grocery ($85.00)", "Online Tech ($480.00)", "Midnight Spree ($2,250)"]
        )
        
        if preset == "Normal Coffee ($4.50)":
            amt, hr, dist, ratio, chip, pin, online = 4.50, 10, 2.1, 0.9, 1, 1, 0
        elif preset == "Daytime Grocery ($85.00)":
            amt, hr, dist, ratio, chip, pin, online = 85.00, 14, 4.5, 1.1, 1, 1, 0
        elif preset == "Online Tech ($480.00)":
            amt, hr, dist, ratio, chip, pin, online = 480.00, 21, 95.0, 4.5, 0, 0, 1
        elif preset == "Midnight Spree ($2,250)":
            amt, hr, dist, ratio, chip, pin, online = 2250.00, 2, 140.0, 8.2, 0, 0, 1
        else:
            amt, hr, dist, ratio, chip, pin, online = 480.00, 2, 95.0, 4.5, 0, 0, 1

        transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=amt, step=10.0)
        transaction_hour = st.slider("Hour of Day (0-23)", 0, 23, hr)
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, value=dist, step=5.0)
        ratio_to_median = st.number_input("Ratio to Median Spend", min_value=0.1, value=ratio, step=0.1)

        st.markdown("### Authentication & Channel Flags")
        c1, c2, c3 = st.columns(3)
        used_chip = c1.checkbox("EMV Chip Used", value=bool(chip))
        used_pin_number = c2.checkbox("PIN Entered", value=bool(pin))
        online_order = c3.checkbox("Online Order", value=bool(online))

    with col2:
        st.subheader("Real-Time Decision Verdict")
        
        payload = {
            "transaction_amount": transaction_amount,
            "transaction_hour": transaction_hour,
            "distance_from_home": distance_from_home,
            "ratio_to_median_purchase_price": ratio_to_median,
            "used_chip": int(used_chip),
            "used_pin_number": int(used_pin_number),
            "online_order": int(online_order),
        }
        
        start_time = time.perf_counter()
        input_df = pd.DataFrame([payload])
        prob = float(pipeline.predict_proba(input_df)[0, 1])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        is_fraud = prob >= 0.50

        if prob >= 0.70:
            risk_tier, box_color = "HIGH RISK", "error"
        elif prob >= 0.35:
            risk_tier, box_color = "MEDIUM RISK", "warning"
        else:
            risk_tier, box_color = "LOW RISK", "success"

        if box_color == "error":
            st.error(f"🚨 **ELEVATED: BLOCK TRANSACTION ({risk_tier})**")
        elif box_color == "warning":
            st.warning(f"⚠️ **ELEVATED: MANUAL REVIEW RECOMMENDED ({risk_tier})**")
        else:
            st.success(f"✅ **APPROVED: LOW RISK TRANSACTION ({risk_tier})**")

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud Probability", f"{prob * 100:.1f}%")
        m2.metric("Inference Latency", f"{latency_ms} ms")
        m3.metric("Decision Cutoff", "50.0%")

        st.markdown("### 🔍 Key Risk Drivers & Signals")
        if ratio_to_median > 3.0:
            st.markdown(f"- 🔴 **Extreme Spend Ratio (High Risk):** Transaction is {ratio_to_median}x normal median spending.")
        if transaction_hour <= 5 or transaction_hour >= 23:
            st.markdown(f"- 🔴 **Suspicious Hour (High Risk):** Transaction occurred during late-night hours ({transaction_hour}:00).")
        if online_order and not used_chip:
            st.markdown(f"- ⚠️ **Card-Not-Present Risk:** Online order executed without EMV chip or PIN verification.")
        if not online_order and used_chip and used_pin_number:
            st.markdown(f"- ✅ **Secure Physical Swipe:** Chip and PIN verified successfully at point of sale.")

# ==========================================
# MODE 2: LIVE TRAFFIC SIMULATOR (Embedded Stream)
# ==========================================
elif app_mode == "Live Traffic Simulator":
    st.title("⚡ Live Transaction Traffic Simulator")
    st.markdown("Simulate automated production transaction streams flowing into the model in real time.")

    col_btn1, col_btn2 = st.columns([1, 4])
    start_sim = col_btn1.button("▶️ Start Simulation", type="primary")
    stop_sim = col_btn2.button("⏹️ Stop / Reset")

    # Session state for live metrics
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False
    if "total_req" not in st.session_state:
        st.session_state.total_req = 0
    if "total_fraud" not in st.session_state:
        st.session_state.total_fraud = 0

    if start_sim:
        st.session_state.sim_running = True
    if stop_sim:
        st.session_state.sim_running = False
        st.session_state.total_req = 0
        st.session_state.total_fraud = 0

    # Telemetry placeholders
    metric_ph1, metric_ph2, metric_ph3 = st.columns(3)
    table_placeholder = st.empty()

    simulation_log = []

    if st.session_state.sim_running:
        for _ in range(50):  # Run 50 simulated batches per click
            if not st.session_state.sim_running:
                break
                
            # Generate random transaction (20% fraud intent)
            is_fraud_intent = random.random() < 0.20
            if is_fraud_intent:
                tx = {
                    "transaction_amount": round(random.uniform(850.0, 3500.0), 2),
                    "transaction_hour": random.choice([1, 2, 3, 23, 0]),
                    "distance_from_home": round(random.uniform(60.0, 250.0), 2),
                    "ratio_to_median_purchase_price": round(random.uniform(4.0, 9.5), 2),
                    "used_chip": 0, "used_pin_number": 0, "online_order": 1
                }
            else:
                tx = {
                    "transaction_amount": round(random.uniform(4.50, 120.0), 2),
                    "transaction_hour": random.randint(8, 21),
                    "distance_from_home": round(random.uniform(0.5, 15.0), 2),
                    "ratio_to_median_purchase_price": round(random.uniform(0.5, 1.8), 2),
                    "used_chip": 1, "used_pin_number": 1, "online_order": random.choice([0, 1])
                }

            start_t = time.perf_counter()
            df_tx = pd.DataFrame([tx])
            prob = float(pipeline.predict_proba(df_tx)[0, 1])
            lat = round((time.perf_counter() - start_t) * 1000, 2)
            is_fraud = prob >= 0.50

            st.session_state.total_req += 1
            if is_fraud:
                st.session_state.total_fraud += 1

            status = "🚨 BLOCK / FRAUD" if is_fraud else "✅ APPROVE"
            
            simulation_log.insert(0, {
                "Timestamp": time.strftime("%H:%M:%S"),
                "Verdict": status,
                "Amount ($)": tx["transaction_amount"],
                "Hour": tx["transaction_hour"],
                "Distance (km)": tx["distance_from_home"],
                "Fraud Prob (%)": round(prob * 100, 1),
                "Latency (ms)": lat
            })

            # Keep only last 10 rows
            if len(simulation_log) > 10:
                simulation_log.pop()

            # Update metrics live
            metric_ph1.metric("Total Transactions Scored", st.session_state.total_req)
            metric_ph2.metric("Flagged & Blocked", st.session_state.total_fraud)
            fraud_rate = (st.session_state.total_fraud / st.session_state.total_req) * 100 if st.session_state.total_req > 0 else 0
            metric_ph3.metric("Live Fraud Rate", f"{fraud_rate:.1f}%")

            # Update table live
            table_placeholder.dataframe(pd.DataFrame(simulation_log), use_container_width=True)
            
            time.sleep(1.0)  # 1 second interval between transactions