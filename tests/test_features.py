import pandas as pd
from fraud_scoring.features import build_point_in_time_features
from fraud_scoring.metrics import CostModel, choose_threshold, expected_loss


def test_features_only_use_prior_events() -> None:
    events = pd.DataFrame({
        "event_time": ["2025-01-01T00:00:00Z", "2025-01-01T00:30:00Z", "2025-01-02T01:00:00Z"],
        "account_token": ["a", "a", "a"], "merchant_token": ["m", "m", "m"], "amount": [10, 20, 30], "is_fraud": [0, 1, 0],
    })
    output = build_point_in_time_features(events)
    assert output.loc[0, "account_txn_count_1h"] == 0
    assert output.loc[1, "account_txn_count_1h"] == 1
    assert output.loc[1, "merchant_prior_fraud_rate_30d"] == 0
    assert output.loc[2, "account_amount_sum_24h"] == 0


def test_threshold_reports_counterfactual_cost() -> None:
    labels = pd.Series([1, 0, 0]).to_numpy()
    scores = pd.Series([0.9, 0.8, 0.1]).to_numpy()
    choice = choose_threshold(labels, scores, CostModel(fraud_loss_usd=100, review_cost_usd=10))
    assert choice["estimated_cost_saved_vs_no_model_usd"] >= 0
    assert expected_loss(labels, scores, 0.5, CostModel())["tp"] == 1
