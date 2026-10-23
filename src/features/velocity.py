"""Real-time sliding window velocity calculator for transaction streams."""
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

class TransactionVelocityStore:
    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)
        self.history = defaultdict(list)  # card_id -> list of (timestamp, amount, lat, lon)

    def update_and_extract(self, card_id: str, amount: float, timestamp: datetime = None) -> dict:
        now = timestamp or datetime.utcnow()
        user_txs = self.history[card_id]
        
        # Purge transactions older than the rolling window
        cutoff = now - self.window
        user_txs = [tx for tx in user_txs if tx[0] >= cutoff]
        
        # Calculate velocity aggregations
        tx_count_last_1h = len(user_txs)
        total_spend_last_1h = sum(tx[1] for tx in user_txs)
        avg_spend_last_1h = total_spend_last_1h / tx_count_last_1h if tx_count_last_1h > 0 else amount
        
        # Record current transaction
        user_txs.append((now, amount))
        self.history[card_id] = user_txs
        
        return {
            "tx_velocity_1h": tx_count_last_1h,
            "spend_velocity_1h": round(total_spend_last_1h, 2),
            "ratio_to_recent_velocity": round(amount / (avg_spend_last_1h + 1e-5), 2),
        }