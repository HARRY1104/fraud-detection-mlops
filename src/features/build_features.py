import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "distance_from_home",
    "ratio_to_median_purchase_price",
    "amount_log",
    "is_night_tx",
]

BINARY_FEATURES = [
    "used_chip",
    "used_pin_number",
    "online_order",
]


class DomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Adds domain-specific fraud features inside the scikit-learn pipeline."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["amount_log"] = np.log1p(np.maximum(df["transaction_amount"], 0))
        df["is_night_tx"] = ((df["transaction_hour"] <= 5) | (df["transaction_hour"] >= 23)).astype(int)
        return df


def build_feature_pipeline() -> Pipeline:
    """Creates an end-to-end preprocessing pipeline preventing training-serving skew."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("domain_features", DomainFeatureGenerator()),
            ("column_preprocessor", column_preprocessor),
        ]
    )
