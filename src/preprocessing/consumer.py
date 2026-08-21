"""
PySpark Structured Streaming job: reads sensor readings live from the
Kafka topic, applies the same feature engineering used in training,
scores each micro-batch with the trained model, and logs predicted
failures in real time.


"""

import os
import sys
import json

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import joblib
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "sensor_predictive_maintenance"

MODEL_PATH = "models/production/current_model.pkl"
FEATURE_COLS_PATH = "models/production/feature_columns.json"

KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6"

# schema of the raw JSON message published by kafka/producer.py
MESSAGE_SCHEMA = StructType(
    [
        StructField("UDI", IntegerType()),
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
        .config("spark.jars.packages", KAFKA_PACKAGE)
        .getOrCreate()
    )


def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training (Step 5) first: "
            f"python src/training/train.py"
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    logger.info(f"Loaded model from {MODEL_PATH}, feature order: {feature_cols}")
    return model, feature_cols


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


def engineer_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("power", col("`Torque [Nm]`") * col("`Rotational speed [rpm]`"))
    df = df.withColumn(
        "temp_diff", col("`Process temperature [K]`") - col("`Air temperature [K]`")
    )
    df = df.withColumn(
        "overstrain_indicator", col("`Tool wear [min]`") * col("`Torque [Nm]`")
    )
    return df


def make_batch_processor(model, feature_cols):
    """Returns a foreachBatch function that scores each micro-batch with the model."""

    def process_batch(batch_df: DataFrame, batch_id: int):
        count = batch_df.count()
        if count == 0:
            return

        pdf = batch_df.select(*feature_cols, "UDI").toPandas()
        X = pdf[feature_cols]

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        pdf["prediction"] = predictions
        pdf["probability"] = probabilities

        failures = pdf[pdf["prediction"] == 1]
        logger.info(
            f"[batch {batch_id}] Processed {count} records | "
            f"{len(failures)} failures flagged"
        )

        for _, row in failures.iterrows():
            logger.warning(
                f"⚠️ FAILURE PREDICTED | UDI={row['UDI']} | "
                f"probability={row['probability']:.2f}"
            )

    return process_batch


def run_streaming_job():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    model, feature_cols = load_model_and_features()

    stream_df = read_kafka_stream(spark)
    features_df = engineer_features(stream_df)

    batch_processor = make_batch_processor(model, feature_cols)

    query = (
        features_df.writeStream.foreachBatch(batch_processor)
        .outputMode("append")
        .trigger(processingTime="5 seconds")
        .start()
    )

    logger.info("Streaming inference job started. Waiting for data... (Ctrl+C to stop)")
    query.awaitTermination()


if __name__ == "__main__":
    run_streaming_job()
