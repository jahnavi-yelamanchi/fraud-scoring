import argparse
import hashlib
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from fraud_scoring.features import FEATURE_NAMES, build_point_in_time_features, materialize_snapshot
from fraud_scoring.metrics import CostModel, choose_threshold, evaluate


def model(weighted: bool, positive_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=7, scale_pos_weight=positive_weight if weighted else 1.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--feature-version", default="v1")
    arguments = parser.parse_args()
    costs = CostModel(float(os.getenv("FRAUD_LOSS_USD", "500")), float(os.getenv("REVIEW_COST_USD", "5")))
    events = pd.read_csv(arguments.input)
    dataset = build_point_in_time_features(events)
    train_end, validation_end = int(len(dataset) * 0.70), int(len(dataset) * 0.85)
    train, validation, test = dataset.iloc[:train_end], dataset.iloc[train_end:validation_end], dataset.iloc[validation_end:]
    x_train, y_train = train[FEATURE_NAMES], train.is_fraud.to_numpy()
    x_validation, y_validation = validation[FEATURE_NAMES], validation.is_fraud.to_numpy()
    positive_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    candidates = {}
    for name, x_fit, y_fit, weighted in (
        ("class_weight", x_train, y_train, True),
        ("smote", *SMOTE(sampling_strategy=0.20, random_state=7).fit_resample(x_train, y_train), False),
    ):
        fitted = model(weighted, positive_weight).fit(x_fit, y_fit)
        validation_scores = fitted.predict_proba(x_validation)[:, 1]
        decision = choose_threshold(y_validation, validation_scores, costs)
        candidates[name] = {"model": fitted, "threshold": decision["threshold"], "validation": decision}
    winner_name, winner = min(candidates.items(), key=lambda item: item[1]["validation"]["estimated_cost_used_usd"])
    test_scores = winner["model"].predict_proba(test[FEATURE_NAMES])[:, 1]
    report = {
        "data": {"input": arguments.input, "rows": len(dataset), "fraud_rate": round(float(dataset.is_fraud.mean()), 6), "split": "chronological 70/15/15"},
        "experiments": {name: value["validation"] for name, value in candidates.items()},
        "winner": winner_name,
        "test": evaluate(test.is_fraud.to_numpy(), test_scores, winner["threshold"], costs),
        "claim": "Offline counterfactual estimate, not realized financial savings.",
    }
    artifacts, reports = Path("artifacts"), Path("reports")
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    model_version = hashlib.sha256(json.dumps({"winner": winner_name, "features": FEATURE_NAMES, "rows": len(dataset)}).encode()).hexdigest()[:12]
    joblib.dump({"model": winner["model"], "threshold": winner["threshold"], "feature_names": FEATURE_NAMES, "model_version": model_version, "feature_version": arguments.feature_version}, artifacts / "model.joblib")
    materialize_snapshot(train, artifacts / "feature_snapshot.json")
    (reports / "latest_metrics.json").write_text(json.dumps(report, indent=2))
    (reports / "latest_metrics.md").write_text(
        f"# Evaluation\n\nWinner: `{winner_name}`. Test PR-AUC: `{report['test']['pr_auc']}`. "
        f"Estimated cost used: `${report['test']['operating_point']['estimated_cost_used_usd']}`; "
        f"saved vs no-model review: `${report['test']['operating_point']['estimated_cost_saved_vs_no_model_usd']}`.\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
