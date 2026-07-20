from datetime import datetime
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=128)
    account_token: str = Field(min_length=16, max_length=128)
    merchant_token: str = Field(min_length=16, max_length=128)
    amount: float = Field(gt=0, le=1_000_000)
    event_time: datetime


class ScoreResponse(BaseModel):
    risk_score: float
    decision: str
    threshold: float
    model_version: str
    feature_version: str
    feature_freshness_seconds: float | None
    stream_enqueued: bool
