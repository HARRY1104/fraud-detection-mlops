import pandas as pd
from pydantic import BaseModel, Field

REQUIRED_COLUMNS = {
    "transaction_amount",
    "transaction_hour",
    "distance_from_home",
    "ratio_to_median_purchase_price",
    "used_chip",
    "used_pin_number",
    "online_order",
    "is_fraud",
}


class TransactionSchema(BaseModel):
    transaction_amount: float = Field(..., gt=0, description="Amount in USD")
    transaction_hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    distance_from_home: float = Field(..., ge=0, description="Distance in km")
    ratio_to_median_purchase_price: float = Field(..., gt=0, description="Ratio to median spend")
    used_chip: int = Field(..., ge=0, le=1)
    used_pin_number: int = Field(..., ge=0, le=1)
    online_order: int = Field(..., ge=0, le=1)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validates schema, null constraints, and value bounds on the dataset."""
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Dataset contains null values in required columns.")

    if (df["transaction_amount"] <= 0).any():
        raise ValueError("transaction_amount must be strictly positive.")

    if not df["transaction_hour"].between(0, 23).all():
        raise ValueError("transaction_hour must be between 0 and 23.")

    if (df["distance_from_home"] < 0).any():
        raise ValueError("distance_from_home cannot be negative.")

    return True
