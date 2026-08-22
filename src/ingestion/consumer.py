"""
src/preprocessing/consumer.py

PySpark Structured Streaming consumer: reads sensor readings live from
the Kafka topic as a Spark DataFrame, assembles the feature vector, and
scores each micro-batch directly with the trained Spark ML
RandomForestClassificationModel (models/best_model) — no format
conversion needed since both sides speak Spark DataFrames natively.

Run:
    python src/preprocessing/consumer.py
"""

import os
import sys

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel

from src.utils.logger import get_logger

logger = get_logger(__name__)

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor_predictive_maintenance"
MODEL_PATH = "models/best_model"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# matches the raw CSV columns produced by producer.py
MESSAGE_SCHEMA = StructType(
    [
        StructField("UDI", IntegerType()),
        StructField("Product ID", StringType()),
        StructField("Type", StringType()),
        StructField("Air temperature [K]", DoubleType()),
        StructField("Process temperature [K]", DoubleType()),
        StructField("Rotational speed [rpm]", IntegerType()),
        StructField("Torque [Nm]", DoubleType()),
        StructField("Tool wear [min]", IntegerType()),
        StructField("Machine failure", IntegerType()),
        StructField("TWF", IntegerType()),
        StructField("HDF", IntegerType()),
        StructField("PWF", IntegerType()),
        StructField("OSF", IntegerType()),
        StructField("RNF", IntegerType()),
    ]
)


def get_spark_session():
    return (
        SparkSession.builder.appName("PredictiveMaintenance-StreamingInference")
        .master("local[*]")
        .config(
            "spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6"
        )
        .getOrCreate()
    )


def read_kafka_stream(spark: SparkSession) -> DataFrame:
    raw_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )
    parsed_df = (
        raw_df.selectExpr("CAST(value AS STRING) as json_str")
        .select(from_json(col("json_str"), MESSAGE_SCHEMA).alias("data"))
        .select("data.*")
    )
    return parsed_df


def make_batch_processor(model, assembler):
    def process_batch(batch_df: DataFrame, batch_id: int):
        count = batch_df.count()
        if count == 0:
            return

        featured_df = assembler.transform(batch_df)
        predictions = model.transform(featured_df)

        results = predictions.select("UDI", "prediction", "probability").collect()

        failures = [r for r in results if r["prediction"] == 1.0]
        logger.info(
            f"[batch {batch_id}] Processed {count} records | "
            f"{len(failures)} failures flagged"
        )

        for row in failures:
            failure_prob = row["probability"][1]
            logger.warning(
                f"⚠️ FAILURE PREDICTED | UDI={row['UDI']} | "
                f"probability={failure_prob:.2f}"
            )

    return process_batch


def run_streaming_job():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(f"Loading Spark ML model from {MODEL_PATH}")
    model = RandomForestClassificationModel.load(MODEL_PATH)

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

    stream_df = read_kafka_stream(spark)
    batch_processor = make_batch_processor(model, assembler)

    query = (
        stream_df.writeStream.foreachBatch(batch_processor)
        .outputMode("update")
        .trigger(processingTime="5 seconds")
        .start()
    )

    logger.info(f"Listening on topic '{TOPIC}'... (Ctrl+C to stop)")
    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_job()
