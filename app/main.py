import logging
from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from app.config import settings
from app.feature_store import RedisFeatureStore
from app.schemas import ScoreRequest, ScoreResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.bundle = joblib.load(settings.model_path)
    except FileNotFoundError as exc:
        raise RuntimeError("Train a model first: make demo-data train") from exc
    app.state.store = RedisFeatureStore(settings.redis_url)
    app.state.producer = None
    if settings.stream_enabled:
        try:
            app.state.producer = KafkaProducer(
                bootstrap_servers=settings.kafka_servers, value_serializer=lambda value: __import__("json").dumps(value).encode()
            )
        except Exception:  # API remains available if Kafka is temporarily unavailable.
            logger.exception("Kafka unavailable; scoring continues without stream publication")
    yield
    if app.state.producer:
        app.state.producer.close()


app = FastAPI(title="Fraud Scoring Service", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    bundle = app.state.bundle
    if bundle["feature_version"] != settings.feature_version:
        raise HTTPException(503, "model and feature versions differ")
    try:
        features, freshness = app.state.store.get_features(request.account_token, request.merchant_token)
    except Exception as exc:
        raise HTTPException(503, "feature store unavailable") from exc
    row = [[request.amount, *(features[name] for name in bundle["feature_names"][1:])]]
    risk_score = float(bundle["model"].predict_proba(row)[0, 1])
    enqueued = False
    if app.state.producer:
        try:
            app.state.producer.send(settings.fraud_topic, request.model_dump(mode="json"))
            enqueued = True
        except Exception:
            logger.exception("Could not publish transaction event")
    return ScoreResponse(
        risk_score=risk_score,
        decision="review" if risk_score >= bundle["threshold"] else "approve",
        threshold=bundle["threshold"],
        model_version=bundle["model_version"],
        feature_version=bundle["feature_version"],
        feature_freshness_seconds=freshness,
        stream_enqueued=enqueued,
    )
