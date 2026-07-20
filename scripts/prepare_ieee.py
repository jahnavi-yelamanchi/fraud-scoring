import argparse
import os
from pathlib import Path
import pandas as pd
from fraud_scoring.features import token


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert IEEE-CIS data to token-only service events.")
    parser.add_argument("--input", default="data/raw/train_transaction.csv")
    parser.add_argument("--output", default="data/processed/ieee_events.csv")
    arguments = parser.parse_args()
    secret = os.getenv("TOKENIZATION_SECRET")
    if not secret:
        raise SystemExit("Set TOKENIZATION_SECRET; raw identifiers must not be written to processed data.")
    raw = pd.read_csv(arguments.input, usecols=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1", "ProductCD", "P_emaildomain"])
    merchant_proxy = raw["ProductCD"].fillna("unknown") + "|" + raw["P_emaildomain"].fillna("unknown")
    events = pd.DataFrame({
        "transaction_id": raw["TransactionID"].astype(str),
        "event_time": pd.to_datetime(raw["TransactionDT"], unit="s", utc=True),
        "account_token": raw["card1"].fillna("unknown").map(lambda value: token(value, secret)),
        "merchant_token": merchant_proxy.map(lambda value: token(value, secret)),
        "amount": raw["TransactionAmt"],
        "is_fraud": raw["isFraud"],
    })
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)
    print(f"wrote {path}; raw fields were never persisted")


if __name__ == "__main__":
    main()
