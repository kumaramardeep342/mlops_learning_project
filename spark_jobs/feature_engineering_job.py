"""
spark_jobs/feature_engineering_job.py

PySpark batch job:
- Reads preprocessed/raw AI4I 2020 data
- Computes derived features:
    1. power
    2. temperature difference
    3. overstrain indicator
- Writes the feature-engineered data as a single CSV file.

Run:
    python spark_jobs/feature_engineering_job.py
"""

import os
import sys
import shutil

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from src.utils.logger import get_logger


logger = get_logger(__name__)

INPUT_FILE = "data/raw/ai4i2020.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data_features.csv")


def get_spark_session():
    return (
        SparkSession.builder
        .appName("PredictiveMaintenance-FeatureEngineering")
        .master("local[*]")
        .getOrCreate()
    )


def engineer_features(df):
    """
    Add derived features to the dataframe.
    """

    df = df.withColumn(
        "power",
        col("`Torque [Nm]`") * col("`Rotational speed [rpm]`")
    )

    df = df.withColumn(
        "temp_diff",
        col("`Process temperature [K]`")
        - col("`Air temperature [K]`")
    )

    df = df.withColumn(
        "overstrain_indicator",
        col("`Tool wear [min]`") * col("`Torque [Nm]`")
    )

    logger.info(
        "Derived features added: "
        "power, temp_diff, overstrain_indicator"
    )

    return df


def process_data(spark):
    """
    Read input data, perform feature engineering,
    and save the result as a single CSV file.
    """

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    logger.info(f"Reading input from {INPUT_FILE}")

    df = spark.read.csv(
        INPUT_FILE,
        header=True,
        inferSchema=True
    )

    logger.info(f"Input row count: {df.count()}")

    # Feature engineering
    df = engineer_features(df)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Temporary directory used by Spark
    temp_output_dir = os.path.join(
        OUTPUT_DIR,
        "temp_features"
    )

    # Remove old temporary output if it exists
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

    # Write exactly one CSV part file
    df.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", True) \
        .csv(temp_output_dir)

    # Find the generated CSV file
    csv_file = None

    for file_name in os.listdir(temp_output_dir):
        if file_name.endswith(".csv"):
            csv_file = os.path.join(
                temp_output_dir,
                file_name
            )
            break

    if csv_file is None:
        raise FileNotFoundError(
            "Spark did not generate the expected CSV file."
        )

    # Remove existing final file if present
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Rename Spark's part file
    shutil.move(
        csv_file,
        OUTPUT_FILE
    )

    # Remove temporary Spark directory
    shutil.rmtree(temp_output_dir)

    logger.info(
        f"Feature-engineered data written to {OUTPUT_FILE}"
    )


def run_job():

    spark = get_spark_session()

    try:
        process_data(spark)

    finally:
        spark.stop()

    logger.info(
        "Spark feature engineering job completed successfully"
    )


if __name__ == "__main__":
    run_job()