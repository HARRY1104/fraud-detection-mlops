# Real-Time Fraud Detection MLOps Pipeline

An end-to-end production Machine Learning system for financial fraud detection featuring schema validation, leak-free feature engineering, MLflow experiment tracking, interactive Streamlit operational dashboard, FastAPI low-latency serving, statistical drift monitoring, and automated CI/CD.

## System Architecture

```text
[Raw Data / DVC] ---> [src/data/validate.py] (Pydantic & Schema Checks)
                             |
                             v
                   [src/features/build_features.py] (Domain & Sklearn Pipeline)
                             |
                             v
                      [src/models/train.py] ---> [MLflow Tracking & Model Artifact]
                             |
            +----------------+----------------+
            |                                 |
            v                                 v
   [src/app/main.py]               [src/monitoring/drift_check.py]
 (Streamlit Dashboard & API)        (KS-Test Statistical Drift Engine)
```

## Quickstart Commands

```bash
# 1. Install dependencies
make setup

# 2. Generate & validate dataset, then train the pipeline
make train

# 3. Run unit & inference latency tests
make test

# 4. Generate production data drift report
make drift

# 5. Launch interactive Streamlit Dashboard
make run-streamlit
# or: streamlit run app.py
# or: streamlit run src/app/main.py

# 6. (Optional) Launch FastAPI REST endpoint
make run-api
```

## Features in Streamlit Dashboard
- **🎯 Real-Time Scoring**: Score individual transactions with pre-configured scenarios (routine coffee, daytime grocery, midnight spree), cardholder profile linking, and instant explainability factor breakdown.
- **👤 Cardholder Profiles**: Persistent SQLite storage for individual person profiles, spending baselines, and individual transaction ledgers.
- **⚡ Live Traffic Simulator**: Stream synthetic live transaction feeds directly into the inference engine with live telemetry and animated logs.
- **📂 Batch Detection**: Upload CSV files or generate synthetic batches, inspect risk distributions with Plotly, and export scored transactions.
- **📈 Drift Monitoring**: Kolmogorov-Smirnov (KS-test) hypothesis testing and side-by-side baseline vs live distribution shift visualizations.
- **🧠 Model Architecture & Retraining**: Real-time evaluation metrics (ROC-AUC, PR-AUC, F1) with one-click in-app model retraining.
- **📜 Session Audit Log**: Full transaction history with telemetry metrics (fraud rate, latency, request volume).

