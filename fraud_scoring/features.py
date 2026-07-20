from collections import defaultdict, deque
from datetime import timedelta
import hashlib
import hmac
import json
from pathlib import Path
import pandas as pd

FEATURE_NAMES = [
    "transaction_amount",
    "account_txn_count_1h",
    "account_amount_sum_24h",
    "merchant_txn_count_24h",
    "merchant_prior_fraud_rate_30d",
]


def token(value: object, secret: str) -> str:
    """HMAC prevents reversible/raw identifier values from leaving ingestion."""
    return hmac.new(secret.encode(), str(value).encode(), hashlib.sha256).hexdigest()


def _trim(rows: deque, cutoff: pd.Timestamp) -> None:
    while rows and rows[0][0] < cutoff:
        rows.popleft()


def build_point_in_time_features(events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_time", "account_token", "merchant_token", "amount", "is_fraud"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    rows = events.copy()
    rows["event_time"] = pd.to_datetime(rows["event_time"], utc=True)
    rows = rows.sort_values("event_time", kind="stable").reset_index(drop=True)
    accounts: dict[str, deque] = defaultdict(deque)
    merchants: dict[str, deque] = defaultdict(deque)
    features: list[dict[str, float]] = []
    for event in rows.itertuples(index=False):
        account, merchant, now = accounts[event.account_token], merchants[event.merchant_token], event.event_time
        _trim(account, now - timedelta(hours=24))
        _trim(merchant, now - timedelta(days=30))
        account_1h = [row for row in account if row[0] >= now - timedelta(hours=1)]
        merchant_24h = [row for row in merchant if row[0] >= now - timedelta(hours=24)]
        features.append(
            {
                "transaction_amount": float(event.amount),
                "account_txn_count_1h": len(account_1h),
                "account_amount_sum_24h": sum(row[1] for row in account),
                "merchant_txn_count_24h": len(merchant_24h),
                "merchant_prior_fraud_rate_30d": (sum(row[1] for row in merchant) / len(merchant) if merchant else 0.0),
            }
        )
        account.append((now, float(event.amount)))
        merchant.append((now, int(event.is_fraud)))
    return pd.concat([rows, pd.DataFrame(features)], axis=1)


def materialize_snapshot(events: pd.DataFrame, path: str | Path) -> None:
    """Export the latest precomputed state; online scoring never recomputes aggregates."""
    rows = events.copy()
    rows["event_time"] = pd.to_datetime(rows["event_time"], utc=True)
    rows = rows.sort_values("event_time", kind="stable")
    accounts: dict[str, deque] = defaultdict(deque)
    merchants: dict[str, deque] = defaultdict(deque)
    snapshot = {"accounts": {}, "merchants": {}}
    for event in rows.itertuples(index=False):
        now = event.event_time
        account, merchant = accounts[event.account_token], merchants[event.merchant_token]
        _trim(account, now - timedelta(hours=24))
        _trim(merchant, now - timedelta(days=30))
        account.append((now, float(event.amount)))
        merchant.append((now, int(event.is_fraud)))
        snapshot["accounts"][event.account_token] = {
            "txn_count_1h": sum(row[0] >= now - timedelta(hours=1) for row in account),
            "amount_sum_24h": sum(row[1] for row in account),
            "updated_at": now.timestamp(),
        }
        snapshot["merchants"][event.merchant_token] = {
            "txn_count_24h": sum(row[0] >= now - timedelta(hours=24) for row in merchant),
            "prior_fraud_rate_30d": sum(row[1] for row in merchant) / len(merchant),
            "updated_at": now.timestamp(),
        }
    Path(path).write_text(json.dumps(snapshot))
