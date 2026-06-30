from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Producer
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FEATURE_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TRANSACTIONS_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "transactions")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def delivery_report(error: Any, message: Any) -> None:
    if error is not None:
        logger.error("Kafka delivery failed: %s", error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay creditcard.csv rows into Kafka.")
    parser.add_argument("--rate", type=int, default=10, help="Transactions per second, max 1000.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to creditcard.csv.",
    )
    args = parser.parse_args()
    if args.rate < 1 or args.rate > 1000:
        parser.error("--rate must be between 1 and 1000 transactions per second.")
    return args


def transaction_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{column: float(row[column]) for column in FEATURE_COLUMNS},
    }


def main() -> None:
    args = parse_args()
    data_path = args.data_path if args.data_path.is_absolute() else PROJECT_ROOT / args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    producer = Producer({"bootstrap.servers": KAFKA_BROKER})
    interval_seconds = 1.0 / args.rate
    sent = 0
    started_at = time.perf_counter()

    logger.info("Publishing to %s via %s at %s tx/sec", TRANSACTIONS_TOPIC, KAFKA_BROKER, args.rate)
    for chunk in pd.read_csv(data_path, chunksize=1000):
        missing = [column for column in FEATURE_COLUMNS if column not in chunk.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

        for row in chunk.to_dict(orient="records"):
            tick = time.perf_counter()
            transaction = transaction_from_row(row)
            producer.produce(
                TRANSACTIONS_TOPIC,
                key=transaction["transaction_id"],
                value=json.dumps(transaction).encode("utf-8"),
                callback=delivery_report,
            )
            producer.poll(0)
            sent += 1

            if sent % 100 == 0:
                elapsed = time.perf_counter() - started_at
                throughput = sent / elapsed if elapsed else 0.0
                logger.info("Published %s messages at %.4f tx/sec", sent, throughput)

            elapsed_tick = time.perf_counter() - tick
            sleep_seconds = max(0.0, interval_seconds - elapsed_tick)
            if sleep_seconds:
                time.sleep(sleep_seconds)

    producer.flush()
    elapsed_total = time.perf_counter() - started_at
    logger.info("Finished publishing %s messages in %.4f seconds", sent, elapsed_total)


if __name__ == "__main__":
    main()
