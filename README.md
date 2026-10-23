# Real-Time Fraud Detection MLOps Pipeline

An end-to-end production Machine Learning system for financial fraud detection featuring schema validation, leak-free feature engineering, MLflow experiment tracking, FastAPI low-latency serving, statistical drift monitoring, and automated CI/CD.

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
 (FastAPI /predict, /metrics)       (KS-Test HTML Drift Report)
```

## Quickstart Commands

```bash
# 1. Install dependencies
make setup

# 2. Generate & validate dataset, then train the pipeline
make train

# 3. Run unit & API latency tests
make test

# 4. Generate production data drift report
make drift

# 5. Launch FastAPI server & interactive web UI
make run-api
```
