from pathlib import Path
import pytest
from src.data.person_store import (
    delete_person,
    get_person_data,
    list_all_persons,
    seed_sample_persons,
    store_person_data,
    store_person_transaction,
)


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_persons.db"
    return db_file


def test_store_and_get_person_data(temp_db):
    profile = store_person_data(
        person_id="P-999",
        name="Alice Wonderland",
        median_purchase_price=65.50,
        home_address="Oxford, UK",
        email="alice@example.com",
        phone="+44-12345",
        risk_status="ACTIVE",
        db_path=temp_db,
    )
    assert profile is not None
    assert profile["person_id"] == "P-999"
    assert profile["name"] == "Alice Wonderland"
    assert profile["median_purchase_price"] == 65.50
    assert profile["stats"]["total_transactions"] == 0

    # Retrieve again
    fetched = get_person_data("P-999", db_path=temp_db)
    assert fetched is not None
    assert fetched["home_address"] == "Oxford, UK"


def test_store_person_transaction(temp_db):
    store_person_data(
        person_id="P-888",
        name="Bob Builder",
        median_purchase_price=50.0,
        db_path=temp_db,
    )

    tx_payload = {
        "transaction_amount": 150.0,
        "transaction_hour": 14,
        "distance_from_home": 12.0,
        "ratio_to_median_purchase_price": 3.0,
        "used_chip": 1,
        "used_pin_number": 0,
        "online_order": 0,
        "fraud_probability": 0.12,
        "is_fraud": False,
        "risk_tier": "LOW RISK",
        "latency_ms": 1.45,
    }

    saved = store_person_transaction("P-888", tx_payload, db_path=temp_db)
    assert saved["transaction_id"] is not None
    assert saved["person_id"] == "P-888"

    # Verify person stats updated
    person = get_person_data("P-888", db_path=temp_db)
    assert person["stats"]["total_transactions"] == 1
    assert person["stats"]["total_volume"] == 150.0
    assert len(person["transactions"]) == 1


def test_list_and_delete_person(temp_db):
    seed_sample_persons(temp_db)
    persons = list_all_persons(temp_db)
    assert len(persons) >= 4

    # Delete one
    deleted = delete_person("P-1001", db_path=temp_db)
    assert deleted is True
    assert get_person_data("P-1001", db_path=temp_db) is None
