"""
spark_jobs/feature_engineering_job.py

Complete PySpark data processing pipeline.

Steps:
1. Read raw AI4I 2020 CSV
2. Validate input schema
3. Check missing/null values
4. Handle missing values
5. Remove duplicate records
6. Create engineered features:
      - power
      - temp_diff
      - overstrain_indicator
7. Validate engineered features
8. Save final dataset as:
      data/processed/data_features.csv

Run:
    python spark_jobs/feature_engineering_job.py
"""

import os
import sys
import shutil

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan
from pyspark.sql.types import NumericType

from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = "data/raw/ai4i2020.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "data_features.csv"
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure"
]


# ============================================================
# SPARK SESSION
# ============================================================

def get_spark_session():

    return (
        SparkSession.builder
        .appName(
            "PredictiveMaintenance-DataProcessing"
        )
        .master("local[*]")
        .getOrCreate()
    )


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(df):

    logger.info("Validating input schema...")

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    logger.info(
        "Schema validation PASSED."
    )


# ============================================================
# NULL VALUE CHECK
# ============================================================

def check_null_values(df):

    logger.info(
        "Checking for missing/null values..."
    )

    total_nulls = 0

    for field in df.schema.fields:

        column = field.name

        if isinstance(
            field.dataType,
            NumericType
        ):

            condition = (
                col(column).isNull()
                | isnan(col(column))
            )

        else:

            condition = col(column).isNull()

        null_count = (
            df.filter(condition)
            .count()
        )

        total_nulls += null_count

        logger.info(
            f"{column}: "
            f"{null_count} missing values"
        )

    logger.info(
        f"Total missing values: "
        f"{total_nulls}"
    )

    return total_nulls


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    logger.info(
        "Handling missing values..."
    )

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    numerical_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(
            field.dataType,
            NumericType
        )
    ]

    for column in numerical_columns:

        missing_count = (
            df.filter(
                col(column).isNull()
                | isnan(col(column))
            )
            .count()
        )

        if missing_count == 0:
            continue

        # Calculate median

        median_value = (
            df.approxQuantile(
                column,
                [0.5],
                0.01
            )[0]
        )

        logger.info(
            f"Imputing {missing_count} missing "
            f"values in '{column}' "
            f"using median={median_value}"
        )

        df = df.fillna(
            {column: median_value}
        )

    # --------------------------------------------------------
    # String columns
    # --------------------------------------------------------

    string_columns = [
        field.name
        for field in df.schema.fields
        if field.dataType.simpleString() == "string"
    ]

    for column in string_columns:

        missing_count = (
            df.filter(
                col(column).isNull()
            )
            .count()
        )

        if missing_count == 0:
            continue

        mode_row = (
            df.groupBy(column)
            .count()
            .orderBy(
                col("count").desc()
            )
            .first()
        )

        if mode_row:

            mode_value = mode_row[0]

            logger.info(
                f"Imputing {missing_count} missing "
                f"values in '{column}' "
                f"using mode='{mode_value}'"
            )

            df = df.fillna(
                {column: mode_value}
            )

    logger.info(
        "Missing value handling completed."
    )

    return df


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(df):

    logger.info(
        "Checking for duplicate records..."
    )

    before_count = df.count()

    df = df.dropDuplicates()

    after_count = df.count()

    duplicates_removed = (
        before_count - after_count
    )

    logger.info(
        f"Duplicate records removed: "
        f"{duplicates_removed}"
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    logger.info(
        "Starting feature engineering..."
    )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    df = df.withColumn(
        "power",
        col("`Torque [Nm]`")
        *
        col("`Rotational speed [rpm]`")
    )

    # --------------------------------------------------------
    # Temperature difference
    # --------------------------------------------------------

    df = df.withColumn(
        "temp_diff",
        col("`Process temperature [K]`")
        -
        col("`Air temperature [K]`")
    )

    # --------------------------------------------------------
    # Overstrain indicator
    # --------------------------------------------------------

    df = df.withColumn(
        "overstrain_indicator",
        col("`Tool wear [min]`")
        *
        col("`Torque [Nm]`")
    )

    logger.info(
        "Engineered features created:"
    )

    logger.info("  - power")
    logger.info("  - temp_diff")
    logger.info(
        "  - overstrain_indicator"
    )

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(df):

    logger.info(
        "Validating engineered features..."
    )

    required_features = [
        "power",
        "temp_diff",
        "overstrain_indicator"
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing engineered features: "
            f"{missing_features}"
        )

    for feature in required_features:

        null_count = (
            df.filter(
                col(feature).isNull()
                | isnan(col(feature))
            )
            .count()
        )

        if null_count > 0:

            raise ValueError(
                f"Feature '{feature}' contains "
                f"{null_count} missing values."
            )

    logger.info(
        "Feature validation PASSED."
    )


# ============================================================
# SAVE FINAL DATASET
# ============================================================

def save_dataset(df):

    logger.info(
        "Saving final feature-engineered dataset..."
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Temporary Spark output directory

    temp_output_dir = os.path.join(
        OUTPUT_DIR,
        "temp_features"
    )

    if os.path.exists(
        temp_output_dir
    ):

        shutil.rmtree(
            temp_output_dir
        )

    # --------------------------------------------------------
    # Write exactly one CSV part file
    # --------------------------------------------------------

    (
        df.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(temp_output_dir)
    )

    # --------------------------------------------------------
    # Find generated CSV
    # --------------------------------------------------------

    csv_file = None

    for file_name in os.listdir(
        temp_output_dir
    ):

        if file_name.endswith(".csv"):

            csv_file = os.path.join(
                temp_output_dir,
                file_name
            )

            break

    if csv_file is None:

        raise FileNotFoundError(
            "Spark did not generate "
            "the expected CSV file."
        )

    # --------------------------------------------------------
    # Remove old final file
    # --------------------------------------------------------

    if os.path.exists(
        OUTPUT_FILE
    ):

        os.remove(
            OUTPUT_FILE
        )

    # --------------------------------------------------------
    # Rename Spark output
    # --------------------------------------------------------

    shutil.move(
        csv_file,
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Remove temporary directory
    # --------------------------------------------------------

    shutil.rmtree(
        temp_output_dir
    )

    logger.info(
        f"Final dataset written to: "
        f"{OUTPUT_FILE}"
    )


# ============================================================
# PROCESS DATA
# ============================================================

def process_data(spark):

    logger.info(
        "Starting data processing pipeline..."
    )

    # --------------------------------------------------------
    # Input file check
    # --------------------------------------------------------

    if not os.path.exists(
        INPUT_FILE
    ):

        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    logger.info(
        f"Reading input from: "
        f"{INPUT_FILE}"
    )

    df = (
        spark.read
        .csv(
            INPUT_FILE,
            header=True,
            inferSchema=True
        )
    )

    initial_count = df.count()

    logger.info(
        f"Initial record count: "
        f"{initial_count}"
    )

    if initial_count == 0:

        raise ValueError(
            "Input dataset contains zero records."
        )

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    validate_schema(df)

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    check_null_values(df)

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    df = handle_missing_values(df)

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(df)

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = engineer_features(df)

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    validate_features(df)

    # --------------------------------------------------------
    # Final record count
    # --------------------------------------------------------

    final_count = df.count()

    logger.info(
        f"Final record count: "
        f"{final_count}"
    )

    if final_count == 0:

        raise ValueError(
            "Final dataset contains zero records."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(df)

    logger.info(
        "Data processing pipeline completed successfully."
    )


# ============================================================
# RUN JOB
# ============================================================

def run_job():

    spark = get_spark_session()

    try:

        process_data(spark)

    finally:

        spark.stop()

    logger.info(
        "Spark feature engineering job completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_job()