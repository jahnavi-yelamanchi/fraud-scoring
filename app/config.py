from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    kafka_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    fraud_topic: str = os.getenv("FRAUD_TOPIC", "transactions")
    stream_enabled: bool = os.getenv("STREAM_ENABLED", "false").lower() == "true"
    model_path: str = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    feature_version: str = os.getenv("FEATURE_VERSION", "v1")


settings = Settings()
