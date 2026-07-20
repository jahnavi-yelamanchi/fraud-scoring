from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class CostModel:
    fraud_loss_usd: float = 500.0
    review_cost_usd: float = 5.0


def expected_loss(y_true: np.ndarray, scores: np.ndarray, threshold: float, costs: CostModel) -> dict:
    flagged = scores >= threshold
    true_positive = int(np.sum(flagged & (y_true == 1)))
    false_positive = int(np.sum(flagged & (y_true == 0)))
    false_negative = int(np.sum(~flagged & (y_true == 1)))
    review_cost = int(np.sum(flagged)) * costs.review_cost_usd
    missed_fraud_cost = false_negative * costs.fraud_loss_usd
    total = review_cost + missed_fraud_cost
    baseline = int(np.sum(y_true == 1)) * costs.fraud_loss_usd
    return {
        "threshold": round(float(threshold), 4),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "review_cost_usd": round(review_cost, 2),
        "missed_fraud_cost_usd": round(missed_fraud_cost, 2),
        "estimated_cost_used_usd": round(total, 2),
        "estimated_cost_saved_vs_no_model_usd": round(baseline - total, 2),
    }


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, costs: CostModel) -> dict:
    candidates = [expected_loss(y_true, scores, threshold, costs) for threshold in np.linspace(0.01, 0.99, 99)]
    return min(candidates, key=lambda row: row["estimated_cost_used_usd"])


def bootstrap_pr_auc(y_true: np.ndarray, scores: np.ndarray, seed: int = 7, samples: int = 200) -> list[float]:
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = generator.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[index])) == 2:
            values.append(average_precision_score(y_true[index], scores[index]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float, costs: CostModel) -> dict:
    return {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 6),
        "pr_auc_95_ci": [round(value, 6) for value in bootstrap_pr_auc(y_true, scores)],
        "cost_model": asdict(costs),
        "operating_point": expected_loss(y_true, scores, threshold, costs),
    }
