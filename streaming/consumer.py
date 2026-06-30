from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from confluent_kafka import Consumer, KafkaException, Producer
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TRANSACTIONS_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "transactions")
RESULTS_TOPIC = os.getenv("RESULTS_TOPIC", "fraud-results")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_SCORE_PATH = os.getenv("API_SCORE_PATH", "/score")
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "5"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def score_with_retry(transaction: dict[str, Any], max_attempts: int = 3) -> dict[str, Any]:
    payload = {column: float(transaction[column]) for column in FEATURE_COLUMNS}
    headers = {"X-Transaction-ID": str(transaction.get("transaction_id", ""))}
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                f"{API_URL}{API_SCORE_PATH}",
                json=payload,
                headers=headers,
                timeout=API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            sleep_seconds = 2 ** (attempt - 1)
            logger.warning(
                "API score failed on attempt %s/%s: %s. Retrying in %.4f seconds",
                attempt,
                max_attempts,
                exc,
                float(sleep_seconds),
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"API score failed after {max_attempts} attempts: {last_error}")


def publish_result(
    producer: Producer,
    transaction: dict[str, Any],
    score_response: dict[str, Any],
) -> None:
    result = {
        "score": score_response,
        "transaction": transaction,
        "processed_at": round(time.time(), 4),
    }
    producer.produce(
        RESULTS_TOPIC,
        key=str(transaction.get("transaction_id", "")),
        value=json.dumps(result).encode("utf-8"),
    )
    producer.flush(5)


def main() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "group.id": os.getenv("KAFKA_GROUP_ID", "fraud-detection-consumer"),
            "auto.offset.reset": os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest"),
            "enable.auto.commit": False,
        }
    )
    producer = Producer({"bootstrap.servers": KAFKA_BROKER})
    consumer.subscribe([TRANSACTIONS_TOPIC])
    logger.info("Consuming %s and publishing %s via %s", TRANSACTIONS_TOPIC, RESULTS_TOPIC, KAFKA_BROKER)

    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                raise KafkaException(message.error())

            try:
                transaction = json.loads(message.value().decode("utf-8"))
                score_response = score_with_retry(transaction)
                publish_result(producer, transaction, score_response)

                if score_response.get("is_fraud"):
                    logger.warning(
                        "Flagged transaction %s with risk_score %.4f",
                        score_response.get("transaction_id"),
                        float(score_response.get("risk_score", 0.0)),
                    )
                consumer.commit(message)
            except Exception as exc:
                logger.exception("Failed to process message: %s", exc)
                consumer.commit(message)
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()
