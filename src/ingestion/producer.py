"""
src/preprocessing/producer.py

Reads the AI4I 2020 raw CSV as a Spark DataFrame (keeping the data path
consistent with the rest of the Spark-based pipeline), then replays it
row by row to a Kafka topic with a small delay, simulating a live
sensor feed.

Run:
    python src/preprocessing/producer.py
"""

import os
import sys
import json
import time

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from kafka import KafkaProducer

from src.utils.logger import get_logger

logger = get_logger(__name__)

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor_predictive_maintenance"
RAW_PATH = "data/raw/ai4i2020.csv"
DELAY_SECONDS = 0.2  # gap between messages — simulates a live sensor feed


def get_spark_session():
    return (
        SparkSession.builder.appName("PredictiveMaintenance-KafkaProducer")
        .master("local[*]")
        .getOrCreate()
    )


def get_kafka_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8") if k is not None else None,
    )


def stream_csv():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data file not found: {RAW_PATH}")

    spark = get_spark_session()

    df = spark.read.csv(RAW_PATH, header=True, inferSchema=True)
    row_count = df.count()
    logger.info(f"Loaded {row_count} rows from {RAW_PATH} as a Spark DataFrame")

    kafka_producer = get_kafka_producer()
    logger.info(
        f"Connected to Kafka at {BOOTSTRAP_SERVERS}, publishing to topic '{TOPIC}'"
    )

    sent = 0
    start_time = time.time()

    # toLocalIterator() streams rows from Spark to the driver one at a
    # time instead of collecting everything into memory at once —
    # keeps this scalable if the source data grows much larger
    for row in df.toLocalIterator():
        message = row.asDict()
        key = str(message.get("UDI", sent))

        kafka_producer.send(TOPIC, key=key, value=message)
        sent += 1

        if sent % 500 == 0:
            logger.info(f"Published {sent} records so far...")

        time.sleep(DELAY_SECONDS)

    kafka_producer.flush()
    elapsed = time.time() - start_time
    throughput = sent / elapsed if elapsed > 0 else 0

    logger.info(f"Done. Total records published: {sent}")
    logger.info(
        f"Time taken: {elapsed:.2f}s | Throughput: {throughput:.2f} records/sec"
    )

    kafka_producer.close()
    spark.stop()


if __name__ == "__main__":
    stream_csv()
