import json
from datetime import datetime, timezone
import redis


def _number(value: str | None) -> float:
    return float(value) if value is not None else 0.0


class RedisFeatureStore:
    def __init__(self, url: str):
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def get_features(self, account_token: str, merchant_token: str) -> tuple[dict[str, float], float | None]:
        account, merchant = self.client.pipeline().hgetall(f"account:{account_token}").hgetall(
            f"merchant:{merchant_token}"
        ).execute()
        now = datetime.now(timezone.utc).timestamp()
        updated = max(_number(account.get("updated_at")), _number(merchant.get("updated_at")))
        return {
            "account_txn_count_1h": _number(account.get("txn_count_1h")),
            "account_amount_sum_24h": _number(account.get("amount_sum_24h")),
            "merchant_txn_count_24h": _number(merchant.get("txn_count_24h")),
            "merchant_prior_fraud_rate_30d": _number(merchant.get("prior_fraud_rate_30d")),
        }, (max(0.0, now - updated) if updated else None)

    def apply_event(self, event: dict) -> None:
        """Single-consumer state update. ponytail: one Kafka consumer partition; use Lua/atomic state for parallel writers."""
        timestamp = datetime.fromisoformat(event["event_time"]).timestamp()
        if not self.client.set(f"seen:{event['transaction_id']}", "1", nx=True, ex=172800):
            return
        self._update_account(event["account_token"], timestamp, event["amount"])
        self._update_merchant(event["merchant_token"], timestamp, event.get("is_fraud"))

    def _load_state(self, key: str, horizon: float, now: float) -> list[list[float]]:
        rows = json.loads(self.client.get(key) or "[]")
        return [row for row in rows if row[0] >= now - horizon]

    def _update_account(self, token: str, now: float, amount: float) -> None:
        state_key = f"state:account:{token}"
        rows = self._load_state(state_key, 86400, now)
        rows.append([now, amount])
        hour_count = sum(row[0] >= now - 3600 for row in rows)
        amount_sum = sum(row[1] for row in rows)
        self.client.pipeline().set(state_key, json.dumps(rows)).hset(
            f"account:{token}", mapping={"txn_count_1h": hour_count, "amount_sum_24h": amount_sum, "updated_at": now}
        ).execute()
    def _update_merchant(self, token: str, now: float, is_fraud: int | None) -> None:
        state_key = f"state:merchant:{token}"
        rows = self._load_state(state_key, 30 * 86400, now)
        rows.append([now, int(is_fraud or 0)])
        recent = [row for row in rows if row[0] >= now - 86400]
        fraud_rate = sum(row[1] for row in rows) / len(rows)
        self.client.pipeline().set(state_key, json.dumps(rows)).hset(
            f"merchant:{token}", mapping={"txn_count_24h": len(recent), "prior_fraud_rate_30d": fraud_rate, "updated_at": now}
        ).execute()
