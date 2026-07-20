import json
from pathlib import Path
from app.config import settings
from app.feature_store import RedisFeatureStore


def main() -> None:
    snapshot = json.loads(Path("artifacts/feature_snapshot.json").read_text())
    store = RedisFeatureStore(settings.redis_url)
    with store.client.pipeline() as pipe:
        for token, fields in snapshot["accounts"].items():
            pipe.hset(f"account:{token}", mapping=fields)
        for token, fields in snapshot["merchants"].items():
            pipe.hset(f"merchant:{token}", mapping=fields)
        pipe.execute()
    print(f"loaded {len(snapshot['accounts'])} account and {len(snapshot['merchants'])} merchant feature rows")


if __name__ == "__main__":
    main()
