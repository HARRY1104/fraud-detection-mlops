from pathlib import Path
import numpy as np
import pandas as pd
from src.data.validate import validate_dataframe

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "transactions.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "train_processed.csv"


def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generates realistic imbalanced financial transaction data."""
    rng = np.random.default_rng(seed)

    transaction_amount = np.round(rng.lognormal(mean=3.5, sigma=1.2, size=n_samples), 2)
    transaction_hour = rng.integers(0, 24, size=n_samples)
    distance_from_home = np.round(rng.exponential(scale=25.0, size=n_samples), 2)
    ratio_to_median = np.round(rng.lognormal(mean=0.0, sigma=0.8, size=n_samples), 2)
    used_chip = rng.choice([0, 1], size=n_samples, p=[0.35, 0.65])
    used_pin_number = rng.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    online_order = rng.choice([0, 1], size=n_samples, p=[0.40, 0.60])

    risk_logit = (
        -3.8
        + 0.004 * transaction_amount
        + 0.45 * ratio_to_median
        + 0.012 * distance_from_home
        + 1.1 * online_order
        - 1.5 * used_pin_number
        - 0.6 * used_chip
        + 0.7 * ((transaction_hour <= 5) | (transaction_hour >= 23)).astype(int)
        + rng.normal(0, 0.5, size=n_samples)
    )
    prob = 1 / (1 + np.exp(-risk_logit))
    is_fraud = (prob > 0.65).astype(int)

    return pd.DataFrame(
        {
            "transaction_amount": transaction_amount,
            "transaction_hour": transaction_hour,
            "distance_from_home": distance_from_home,
            "ratio_to_median_purchase_price": ratio_to_median,
            "used_chip": used_chip,
            "used_pin_number": used_pin_number,
            "online_order": online_order,
            "is_fraud": is_fraud,
        }
    )


def load_or_create_dataset(filepath: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Ingests existing dataset or creates and validates a new one."""
    if filepath.exists() and filepath.stat().st_size > 0:
        df = pd.read_csv(filepath)
    else:
        df = generate_synthetic_data()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

    validate_dataframe(df)
    return df


if __name__ == "__main__":
    df = load_or_create_dataset()
    print(f"Dataset ready at {PROCESSED_DATA_PATH} | Shape: {df.shape} | Fraud Rate: {df['is_fraud'].mean():.2%}")
