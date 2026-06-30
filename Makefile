.PHONY: install demo-data train test compile compose-up compose-down api producer consumer docker-build

install:
	python -m pip install -r requirements.txt

demo-data:
	python scripts/generate_demo_data.py --rows 5000 --fraud-rate 0.02

train:
	python model/train.py

test:
	pytest

compile:
	python -m compileall api model streaming scripts tests

compose-up:
	docker compose up -d

compose-down:
	docker compose down

api:
	uvicorn api.main:app --reload

producer:
	python streaming/producer.py --rate 50

consumer:
	python streaming/consumer.py

docker-build:
	docker build -t sentinelpay-fraud-api .
