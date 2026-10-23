import pytest
from src.data.make_dataset import generate_synthetic_data
from src.data.validate import validate_dataframe


def test_synthetic_data_passes_schema_validation():
    df = generate_synthetic_data(n_samples=200)
    assert validate_dataframe(df) is True


def test_negative_amount_raises_validation_error():
    df = generate_synthetic_data(n_samples=50)
    df.loc[0, "transaction_amount"] = -10.0
    with pytest.raises(ValueError, match="transaction_amount must be strictly positive"):
        validate_dataframe(df)


def test_null_value_raises_validation_error():
    df = generate_synthetic_data(n_samples=50)
    df.loc[0, "distance_from_home"] = None
    with pytest.raises(ValueError, match="null values"):
        validate_dataframe(df)
