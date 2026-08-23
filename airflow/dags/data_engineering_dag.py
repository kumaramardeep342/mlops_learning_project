from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator

from datetime import datetime


with DAG(
    dag_id="mlops_data_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["MLOps", "PySpark", "Data"],
) as dag:

    # ========================================================
    # 1. DATA INGESTION CHECK
    # ========================================================

    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command="""
        echo "======================================"
        echo "DATA INGESTION CHECK"
        echo "======================================"

        cd /opt/airflow/project

        FILE="data/raw/ai4i2020.csv"

        if [ ! -f "$FILE" ]; then
            echo "ERROR: Raw dataset not found."
            exit 1
        fi

        if [ ! -s "$FILE" ]; then
            echo "ERROR: Raw dataset is empty."
            exit 1
        fi

        echo "Raw dataset found."
        echo "Data ingestion check PASSED."
        """
    )


    # ========================================================
    # 2. PYSPARK DATA PROCESSING
    # ========================================================

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command="""
        echo "======================================"
        echo "PYSPARK DATA PROCESSING"
        echo "======================================"

        cd /opt/airflow/project

        python spark_jobs/feature_engineering_job.py

        echo "PySpark processing completed."
        """
    )


    # ========================================================
    # 3. PARALLEL FINAL VALIDATION
    # ========================================================

    final_file_check = BashOperator(
        task_id="final_file_check",
        bash_command="""
        echo "======================================"
        echo "FINAL FILE CHECK"
        echo "======================================"

        cd /opt/airflow/project

        FILE="data/processed/data_features.csv"

        if [ ! -f "$FILE" ]; then
            echo "ERROR: data_features.csv does not exist."
            exit 1
        fi

        if [ ! -s "$FILE" ]; then
            echo "ERROR: data_features.csv is empty."
            exit 1
        fi

        ROW_COUNT=$(($(wc -l < "$FILE") - 1))

        echo "Final record count: $ROW_COUNT"

        if [ "$ROW_COUNT" -eq 0 ]; then
            echo "ERROR: Final dataset contains zero records."
            exit 1
        fi

        echo "Final file check PASSED."
        """
    )


    final_null_check = BashOperator(
        task_id="final_null_check",
        bash_command="""
        echo "======================================"
        echo "FINAL NULL CHECK"
        echo "======================================"

        cd /opt/airflow/project

        python - <<'PY'
import pandas as pd
import sys

file = "data/processed/data_features.csv"

df = pd.read_csv(file)

null_count = df.isnull().sum().sum()

print(f"Total NULL values: {null_count}")

if null_count > 0:
    print("ERROR: NULL values found.")
    sys.exit(1)

print("Final NULL check PASSED.")
PY
        """
    )


    final_feature_check = BashOperator(
        task_id="final_feature_check",
        bash_command="""
        echo "======================================"
        echo "FINAL FEATURE CHECK"
        echo "======================================"

        cd /opt/airflow/project

        python - <<'PY'
import pandas as pd
import sys

file = "data/processed/data_features.csv"

df = pd.read_csv(file)

required_features = [
    "power",
    "temp_diff",
    "overstrain_indicator"
]

missing = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing:
    print("ERROR: Missing engineered features:")
    print(missing)
    sys.exit(1)

print("Required engineered features:")
for feature in required_features:
    print(f"  - {feature}")

print("Final feature check PASSED.")
PY
        """
    )


    # ========================================================
    # 4. PARALLEL VALIDATION JOIN
    # ========================================================

    validation_complete = EmptyOperator(
        task_id="validation_complete"
    )


    # ========================================================
    # 5. PIPELINE COMPLETE
    # ========================================================

    pipeline_complete = BashOperator(
        task_id="pipeline_complete",
        bash_command="""
        echo "======================================"
        echo "MLOPS DATA PIPELINE COMPLETED"
        echo "======================================"

        cd /opt/airflow/project

        echo "Final output:"
        ls -lh data/processed/data_features.csv

        echo ""
        echo "Processed directory:"
        ls -lh data/processed/

        echo ""
        echo "Pipeline completed successfully."
        """
    )


    # ========================================================
    # DEPENDENCIES
    # ========================================================

    # Sequential:
    # ingestion -> Spark processing

    data_ingestion >> feature_engineering


    # Spark processing -> parallel validation

    feature_engineering >> [
        final_file_check,
        final_null_check,
        final_feature_check
    ]


    # All validation tasks must complete

    [
        final_file_check,
        final_null_check,
        final_feature_check
    ] >> validation_complete


    # Final pipeline completion

    validation_complete >> pipeline_complete