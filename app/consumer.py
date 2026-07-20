import json
from kafka import KafkaConsumer
from app.config import settings
from app.feature_store import RedisFeatureStore


def main() -> None:
    consumer = KafkaConsumer(
        settings.fraud_topic,
        bootstrap_servers=settings.kafka_servers,
        group_id="feature-updater-v1",
        auto_offset_reset="earliest",
        value_deserializer=lambda value: json.loads(value.decode()),
    )
    store = RedisFeatureStore(settings.redis_url)
    for message in consumer:
        store.apply_event(message.value)


if __name__ == "__main__":
    main()
