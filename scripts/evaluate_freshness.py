import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd
from fraud_scoring.features import FEATURE_NAMES, build_point_in_time_features
from fraud_scoring.metrics import CostModel, evaluate


def static_rows(events: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        account = snapshot["accounts"].get(event.account_token, {})
        merchant = snapshot["merchants"].get(event.merchant_token, {})
        rows.append([event.amount, account.get("txn_count_1h", 0), account.get("amount_sum_24h", 0), merchant.get("txn_count_24h", 0), merchant.get("prior_fraud_rate_30d", 0)])
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--snapshot", default="artifacts/feature_snapshot.json")
    arguments = parser.parse_args()
    events = pd.read_csv(arguments.input)
    start = int(len(events) * 0.85)  # Matches scripts/train.py's chronological test boundary.
    test_events = events.iloc[start:].reset_index(drop=True)
    snapshot = json.loads(Path(arguments.snapshot).read_text())
    bundle = joblib.load(arguments.model)
    costs = CostModel(float(os.getenv("FRAUD_LOSS_USD", "500")), float(os.getenv("REVIEW_COST_USD", "5")))
    static = static_rows(test_events, snapshot)
    fresh = build_point_in_time_features(events).iloc[start:][FEATURE_NAMES].reset_index(drop=True)
    # Fraud labels arrive later; only velocity/spend features are refreshed by score events in this demo.
    fresh["merchant_prior_fraud_rate_30d"] = static["merchant_prior_fraud_rate_30d"]
    labels = test_events.is_fraud.to_numpy()
    report = {
        "static": evaluate(labels, bundle["model"].predict_proba(static)[:, 1], bundle["threshold"], costs),
        "fresh": evaluate(labels, bundle["model"].predict_proba(fresh)[:, 1], bundle["threshold"], costs),
        "assumption": "Score events refresh velocity/spend immediately; fraud labels are delayed and remain from the materialized snapshot.",
    }
    report["freshness_value"] = {
        "pr_auc_delta": round(report["fresh"]["pr_auc"] - report["static"]["pr_auc"], 6),
        "estimated_cost_delta_usd": round(report["static"]["operating_point"]["estimated_cost_used_usd"] - report["fresh"]["operating_point"]["estimated_cost_used_usd"], 2),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latest_freshness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
