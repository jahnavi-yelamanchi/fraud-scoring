from pathlib import Path
import numpy as np
import pandas as pd
from fraud_scoring.features import token


def main() -> None:
    generator = np.random.default_rng(17)
    count = 6000
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    account_ids = [token(f"account-{index}", "demo-secret") for index in range(450)]
    merchant_ids = [token(f"merchant-{index}", "demo-secret") for index in range(60)]
    account = generator.choice(account_ids, count)
    merchant = generator.choice(merchant_ids, count)
    seconds = np.cumsum(generator.exponential(45, count)).astype(int)
    amount = np.round(generator.lognormal(3.5, 1.0, count), 2)
    suspicious = (amount > 180) | (generator.random(count) < 0.025)
    fraud_probability = np.where(suspicious, 0.28, 0.012)
    labels = (generator.random(count) < fraud_probability).astype(int)
    events = pd.DataFrame(
        {"transaction_id": [f"demo-{index}" for index in range(count)], "event_time": start + pd.to_timedelta(seconds, unit="s"),
         "account_token": account, "merchant_token": merchant, "amount": amount, "is_fraud": labels}
    )
    path = Path("data/processed/demo_events.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)
    Path("data/processed/benchmark_payload.json").write_text(
        pd.Series({"transaction_id": "benchmark-event", "account_token": account_ids[0], "merchant_token": merchant_ids[0], "amount": 42.0, "event_time": str(start + pd.Timedelta(days=7))}).to_json()
    )
    print(f"wrote {path} ({labels.mean():.2%} fraud)")


if __name__ == "__main__":
    main()
