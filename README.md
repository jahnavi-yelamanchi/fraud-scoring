# Fraud scoring service

An event-driven, low-latency fraud scoring demo built around IEEE-CIS Fraud Detection. It trains an XGBoost model on point-in-time aggregate features, serves token-only requests through FastAPI, reads its online features from Redis, and publishes each scored transaction to Kafka so a consumer updates those aggregates for the next request.

## What makes it different

This is not a leaderboard notebook wrapped in an API. The project tests a production hypothesis:

> Fresh, point-in-time-correct entity history should improve the *economic operating point* of an imbalanced fraud model without breaking a latency budget.

That gives the project three measurable ablations:

1. **Imbalance treatment:** compare XGBoost class weighting with SMOTE, applied to the training period only.
2. **Feature freshness:** compare the same model using static offline features against velocity/spend features updated by the Kafka consumer.
3. **Decision rule:** choose the review threshold on the chronological validation period by minimum expected loss, then report it once on the final test period.

The evaluation follows the precision-recall recommendation for imbalanced problems in [Saito and Rehmsmeier (2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432). SMOTE is an experiment, not a default, following its original formulation in [Chawla et al. (2002)](https://jair.org/index.php/jair/article/view/10302). XGBoost is used as the deliberately compact tabular baseline described by [Chen and Guestrin (2016)](https://arxiv.org/abs/1603.02754). The point-in-time boundary mirrors the online/offline retrieval and leakage constraints documented by [Feast](https://docs.feast.dev/getting-started/concepts/feature-retrieval).

## Architecture

```text
token-only score request
          │
          ▼
 FastAPI ───────► Redis feature store ───────► XGBoost decision
    │                                                │
    └──────── sanitized event ─► Kafka ─► consumer ─┘
                                           updates next-request features
```

`account_token` and `merchant_token` are keyed HMAC values. `scripts/prepare_ieee.py` converts raw dataset columns in memory and writes only tokens to `data/processed/`; the serving API never accepts raw PII.

## Demo

Prerequisites: Python 3.12, Docker Desktop, and a virtual environment with `pip install -r requirements.txt && pip install -e .`.

```bash
cp .env.example .env
make demo-data train
make freshness
make up
```

In another terminal, load the initial online snapshot and score a transaction:

```bash
make load-features
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/v1/score \
  -H 'content-type: application/json' \
  --data @data/processed/benchmark_payload.json
make benchmark
```

The API returns a model risk score, review decision, feature freshness, and whether the sanitized event was accepted by Kafka. The consumer updates Redis so later requests see new transaction velocity. `make benchmark` writes p50/p95/p99 and throughput to `reports/latest_benchmark.json`.

`make freshness` writes the static-versus-fresh ablation to `reports/latest_freshness.json`. It intentionally holds the merchant fraud-rate feature static: a score event does not yet mean a confirmed fraud label. That separation avoids leaking future labels into the “real-time” claim.

The online model is single-threaded per request to avoid CPU oversubscription under concurrent traffic; scale serving workers only after measuring a larger load.

### Checked synthetic baseline

The committed code has been exercised on its generated demo data, not IEEE-CIS: fresh velocity/spend features increased PR-AUC by `0.044496` and reduced the configured held-out expected cost by `$1,920`. The local Docker benchmark completed 500 requests at concurrency 25 with `385.72 RPS` and `230.48 ms p99`. These figures validate the pipeline only; replace them with IEEE-CIS results before using them in a resume claim.

## Real-data path

Download `train_transaction.csv` from the IEEE-CIS Kaggle competition into `data/raw/` (ignored by Git), then:

```bash
export TOKENIZATION_SECRET='long-random-secret'
python scripts/prepare_ieee.py
python scripts/train.py --input data/processed/ieee_events.csv
```

The dataset has no literal merchant ID. `ProductCD|P_emaildomain` is therefore labeled **merchant proxy** everywhere in the report. That limitation is explicit rather than disguised as ground truth.

## Metrics and cost accounting

The generated report contains PR-AUC with a bootstrap 95% interval, ROC-AUC, TP/FP/FN, and the selected threshold. Cost is intentionally an **offline counterfactual**, not a claim of realized business savings:

```text
model cost used = flagged transactions × review cost + missed fraud × fraud loss
cost saved vs no-model = all fraud × fraud loss − model cost used
```

Default assumptions are $500 per missed fraud and $5 per review. Change `FRAUD_LOSS_USD` and `REVIEW_COST_USD` in `.env` or the shell before training, document the source of the values, and report sensitivity across a small range. Never present the estimate as actual dollars saved.

## Commit milestones

| Commit | Deliverable | Proof before push |
| --- | --- | --- |
| `chore: bootstrap fraud scoring service` | Repo hygiene, Docker Compose, FastAPI health check | `docker compose config` |
| `feat: add point-in-time fraud features` | Token-only IEEE adapter and leakage-safe aggregates | `pytest -q` |
| `feat: train cost-aware fraud model` | XGBoost, class-weight vs SMOTE comparison, test metrics | `make demo-data train` |
| `feat: stream feature updates through kafka` | Redis snapshot loader and Kafka consumer | score twice and show freshness/velocity change |
| `feat: add latency benchmark and project report` | concurrent benchmark, metric tables, architecture diagram | `make benchmark` |
| `docs: publish reproducible resume demo` | recorded run output, limitations, 60-second demo script | clean-clone walkthrough |

After each proof: `git add <files> && git commit -m "..." && git push -u origin main`. Do not commit `data/raw/`, HMAC secrets, generated model binaries, or tool state.

## Resume-safe claim template

“Built a token-only, event-driven fraud scoring service using XGBoost, FastAPI, Redis, and Kafka; compared class weighting with SMOTE using a chronological evaluation, achieved **[measured PR-AUC]**, and served scores at **[measured p99]** under **[measured load]**. Estimated review and missed-fraud cost on held-out data with an explicitly documented counterfactual.”

Replace brackets only with `reports/latest_metrics.json` and `reports/latest_benchmark.json` results from your machine.
