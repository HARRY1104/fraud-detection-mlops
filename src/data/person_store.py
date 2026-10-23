"""Storage and management module for individual person/cardholder data.

Provides SQLite persistence for:
1. Individual person profiles (ID, name, email, home address, median purchase price, account status)
2. Individual transaction ledgers linked to each person
3. Historical analytics and spending profile metrics for fraud evaluation
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT_DIR / "data" / "persons.db"


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a database connection with dictionary-like row access."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_person_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initializes tables for individual person profiles and their transactions."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                home_address TEXT,
                median_purchase_price REAL DEFAULT 50.0,
                risk_status TEXT DEFAULT 'ACTIVE',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS person_transactions (
                transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                transaction_amount REAL NOT NULL,
                transaction_hour INTEGER NOT NULL,
                distance_from_home REAL NOT NULL,
                ratio_to_median_purchase_price REAL NOT NULL,
                used_chip INTEGER NOT NULL,
                used_pin_number INTEGER NOT NULL,
                online_order INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                risk_tier TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                notes TEXT,
                FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_person_id 
            ON person_transactions (person_id)
            """
        )
        conn.commit()


def store_person_data(
    person_id: str,
    name: str,
    median_purchase_price: float = 50.0,
    home_address: str = "Home",
    email: str = "",
    phone: str = "",
    risk_status: str = "ACTIVE",
    db_path: Path = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Stores or updates an individual person's profile in the database (UPSERT).

    Args:
        person_id: Unique identifier for the individual (e.g. 'CUST-1001')
        name: Full name of the person
        median_purchase_price: Baseline median transaction amount in USD
        home_address: Home location / city for geodistance baseline
        email: Contact email address
        phone: Contact phone number
        risk_status: Current account status ('ACTIVE', 'UNDER_REVIEW', 'BLOCKED')
        db_path: Path to SQLite database file

    Returns:
        Dictionary representation of the stored person profile.
    """
    init_person_db(db_path)
    now = datetime.now().isoformat()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO persons (
                person_id, name, email, phone, home_address,
                median_purchase_price, risk_status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                name = excluded.name,
                email = excluded.email,
                phone = excluded.phone,
                home_address = excluded.home_address,
                median_purchase_price = excluded.median_purchase_price,
                risk_status = excluded.risk_status,
                updated_at = excluded.updated_at
            """,
            (
                str(person_id).strip(),
                str(name).strip(),
                str(email).strip(),
                str(phone).strip(),
                str(home_address).strip(),
                float(median_purchase_price),
                str(risk_status).strip().upper(),
                now,
                now,
            ),
        )
        conn.commit()

    return get_person_data(person_id, db_path=db_path)  # type: ignore[return-value]


def get_person_data(
    person_id: str,
    include_transactions: bool = True,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Retrieves an individual person's profile, aggregate statistics, and transaction history.

    Args:
        person_id: Unique person identifier
        include_transactions: Whether to include list of individual transaction records
        db_path: Path to SQLite database file

    Returns:
        Dictionary with profile, statistics, and transactions, or None if not found.
    """
    init_person_db(db_path)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM persons WHERE person_id = ?",
            (str(person_id).strip(),),
        )
        row = cursor.fetchone()
        if not row:
            return None

        profile = dict(row)

        # Aggregate stats
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_transactions,
                SUM(is_fraud) as flagged_transactions,
                AVG(transaction_amount) as avg_spend,
                MAX(transaction_amount) as max_spend,
                SUM(transaction_amount) as total_volume,
                AVG(fraud_probability) as avg_fraud_prob
            FROM person_transactions
            WHERE person_id = ?
            """,
            (str(person_id).strip(),),
        )
        stats_row = cursor.fetchone()
        stats = {
            "total_transactions": stats_row["total_transactions"] or 0,
            "flagged_transactions": stats_row["flagged_transactions"] or 0,
            "avg_spend": round(stats_row["avg_spend"] or 0.0, 2),
            "max_spend": round(stats_row["max_spend"] or 0.0, 2),
            "total_volume": round(stats_row["total_volume"] or 0.0, 2),
            "avg_fraud_prob": round(stats_row["avg_fraud_prob"] or 0.0, 4),
        }
        profile["stats"] = stats

        if include_transactions:
            cursor.execute(
                """
                SELECT * FROM person_transactions 
                WHERE person_id = ? 
                ORDER BY timestamp DESC
                """,
                (str(person_id).strip(),),
            )
            profile["transactions"] = [dict(tx) for tx in cursor.fetchall()]
        else:
            profile["transactions"] = []

        return profile


def store_person_transaction(
    person_id: str,
    transaction: Dict[str, Any],
    db_path: Path = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Records a scored transaction for an individual person.

    If the person does not exist yet, creates a basic profile for them.
    Calculates ratio_to_median_purchase_price automatically if omitted.

    Args:
        person_id: Unique person identifier
        transaction: Dictionary containing transaction details and prediction output
        db_path: Path to SQLite database file

    Returns:
        The saved transaction record with its assigned ID and timestamp.
    """
    init_person_db(db_path)
    person = get_person_data(person_id, include_transactions=False, db_path=db_path)
    if not person:
        store_person_data(
            person_id=person_id,
            name=transaction.get("name", f"Cardholder {person_id}"),
            median_purchase_price=float(transaction.get("median_purchase_price", 50.0)),
            home_address=transaction.get("home_address", "Primary Residence"),
            db_path=db_path,
        )
        person = get_person_data(person_id, include_transactions=False, db_path=db_path)

    median_price = float(person["median_purchase_price"])  # type: ignore[index]
    amount = float(transaction.get("transaction_amount", 0.0))

    ratio = transaction.get("ratio_to_median_purchase_price")
    if ratio is None or float(ratio) <= 0:
        ratio = round(amount / median_price, 2) if median_price > 0 else 1.0

    timestamp = transaction.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO person_transactions (
                person_id, timestamp, transaction_amount, transaction_hour,
                distance_from_home, ratio_to_median_purchase_price, used_chip,
                used_pin_number, online_order, fraud_probability, is_fraud,
                risk_tier, latency_ms, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(person_id).strip(),
                str(timestamp),
                amount,
                int(transaction.get("transaction_hour", 12)),
                float(transaction.get("distance_from_home", 0.0)),
                float(ratio),
                int(transaction.get("used_chip", 0)),
                int(transaction.get("used_pin_number", 0)),
                int(transaction.get("online_order", 0)),
                float(transaction.get("fraud_probability", 0.0)),
                1 if transaction.get("is_fraud") else 0,
                str(transaction.get("risk_tier", "LOW RISK")),
                float(transaction.get("latency_ms", 0.0)),
                str(transaction.get("notes", "")),
            ),
        )
        tx_id = cursor.lastrowid
        conn.commit()

    saved_tx = dict(transaction)
    saved_tx["transaction_id"] = tx_id
    saved_tx["person_id"] = person_id
    saved_tx["timestamp"] = timestamp
    saved_tx["ratio_to_median_purchase_price"] = ratio
    return saved_tx


def list_all_persons(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Returns all stored persons along with high-level transaction volume and risk stats."""
    init_person_db(db_path)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                p.person_id,
                p.name,
                p.email,
                p.home_address,
                p.median_purchase_price,
                p.risk_status,
                p.created_at,
                COUNT(t.transaction_id) as total_transactions,
                SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as flagged_count,
                COALESCE(SUM(t.transaction_amount), 0.0) as total_spent
            FROM persons p
            LEFT JOIN person_transactions t ON p.person_id = t.person_id
            GROUP BY p.person_id
            ORDER BY p.name ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_person(person_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Deletes an individual person and cascades deletion to their transaction records."""
    init_person_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM persons WHERE person_id = ?", (str(person_id).strip(),))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted


def seed_sample_persons(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Seeds the database with representative cardholder personas if empty."""
    init_person_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM persons")
        if cursor.fetchone()["count"] > 0:
            return  # Already populated

    sample_profiles = [
        {
            "person_id": "P-1001",
            "name": "Sarah Jenkins",
            "email": "sarah.jenkins@example.com",
            "phone": "+1-555-0101",
            "home_address": "Austin, TX",
            "median_purchase_price": 42.50,
            "risk_status": "ACTIVE",
        },
        {
            "person_id": "P-1002",
            "name": "David Chen",
            "email": "david.chen@example.com",
            "phone": "+1-555-0102",
            "home_address": "San Francisco, CA",
            "median_purchase_price": 125.00,
            "risk_status": "ACTIVE",
        },
        {
            "person_id": "P-1003",
            "name": "Elena Rostova",
            "email": "elena.rostova@example.com",
            "phone": "+1-555-0103",
            "home_address": "Seattle, WA",
            "median_purchase_price": 28.00,
            "risk_status": "ACTIVE",
        },
        {
            "person_id": "P-1004",
            "name": "Marcus Vance",
            "email": "marcus.vance@example.com",
            "phone": "+1-555-0104",
            "home_address": "Chicago, IL",
            "median_purchase_price": 310.00,
            "risk_status": "UNDER_REVIEW",
        },
    ]

    for p in sample_profiles:
        store_person_data(
            person_id=p["person_id"],
            name=p["name"],
            email=p["email"],
            phone=p["phone"],
            home_address=p["home_address"],
            median_purchase_price=p["median_purchase_price"],
            risk_status=p["risk_status"],
            db_path=db_path,
        )
