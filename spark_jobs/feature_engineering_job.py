"""
spark_jobs/feature_engineering_job.py

PySpark batch job: reads preprocessed train/test data, computes derived
features (power, temperature delta, overstrain indicator), writes
feature-engineered output as Parquet.

Run:
    python spark_jobs/feature_engineering_job.py
"""

import os
import sys

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from src.utils.logger import get_logger

logger = get_logger(__name__)

INPUT_FILES = {
    "train": "data/processed/train.csv",
    "test": "data/processed/test.csv",
}
OUTPUT_DIR = "data/features"


def get_spark_session():
    return (
        SparkSession.builder.appName("PredictiveMaintenance-FeatureEngineering")
        .master("local[*]")
        .getOrCreate()
    )


def engineer_features(df):
    df = df.withColumn("power", col("`Torque [Nm]`") * col("`Rotational speed [rpm]`"))
    df = df.withColumn(
        "temp_diff", col("`Process temperature [K]`") - col("`Air temperature [K]`")
    )
    df = df.withColumn(
        "overstrain_indicator", col("`Tool wear [min]`") * col("`Torque [Nm]`")
    )
    logger.info("Derived features added: power, temp_diff, overstrain_indicator")
    return df


def process_split(spark, split_name: str, input_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run preprocessing (Step 2) first."
        )

    logger.info(f"[{split_name}] Reading input from {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)
    logger.info(f"[{split_name}] Row count: {df.count()}")

    df = engineer_features(df)

    output_path = os.path.join(OUTPUT_DIR, f"{split_name}_features.parquet")
    df.write.mode("overwrite").parquet(output_path)
    logger.info(f"[{split_name}] Feature-engineered data written to {output_path}")


def run_job():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark_session()

    for split_name, input_path in INPUT_FILES.items():
        process_split(spark, split_name, input_path)

    spark.stop()
    logger.info("Spark feature engineering job completed successfully")


if __name__ == "__main__":
    run_job()
