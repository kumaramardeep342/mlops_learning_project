"""
kafka/producer.py

Reads the AI4I 2020 raw CSV row by row and publishes each row as a JSON
message to a Kafka topic, simulating a live sensor feed for the
inference pipeline.

Run:
    python kafka/producer.py
"""

import os
import sys
import json
import time

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from kafka import KafkaProducer

from src.utils.logger import get_logger

logger = get_logger(__name__)

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor_predictive_maintenance"
RAW_PATH = "data/raw/ai4i2020.csv"
DELAY_SECONDS = 0.2  # gap between messages — simulates a live sensor feed


def make_json_safe(value):
    """Converts numpy/pandas types to plain Python types so json.dumps works."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def row_to_message(row: pd.Series) -> dict:
    """Converts a pandas row into a JSON-serializable dict."""
    return {col: make_json_safe(val) for col, val in row.to_dict().items()}


def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
    )


def stream_csv():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded {df.shape[0]} rows from {RAW_PATH}")

    producer = get_producer()
    logger.info(
        f"Connected to Kafka at {BOOTSTRAP_SERVERS}, publishing to topic '{TOPIC}'"
    )

    sent = 0
    start_time = time.time()

    for _, row in df.iterrows():
        message = row_to_message(row)
        key = str(message.get("UDI", sent))  # UDI as partition key

        producer.send(TOPIC, key=key, value=message)
        sent += 1

        if sent % 500 == 0:
            logger.info(f"Published {sent} records so far...")

        time.sleep(DELAY_SECONDS)

    producer.flush()
    elapsed = time.time() - start_time
    throughput = sent / elapsed if elapsed > 0 else 0

    logger.info(f"Done. Total records published: {sent}")
    logger.info(
        f"Time taken: {elapsed:.2f}s | Throughput: {throughput:.2f} records/sec"
    )

    producer.close()


if __name__ == "__main__":
    stream_csv()
