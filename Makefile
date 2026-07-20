.PHONY: demo-data train freshness load-features up down benchmark test

demo-data:
	python scripts/generate_demo_data.py

train:
	python scripts/train.py --input data/processed/demo_events.csv

freshness:
	python scripts/evaluate_freshness.py --input data/processed/demo_events.csv

load-features:
	python scripts/load_features.py

up:
	docker compose up --build

down:
	docker compose down

benchmark:
	python scripts/benchmark.py

test:
	pytest -q
