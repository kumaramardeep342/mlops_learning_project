"""
airflow/dags/data_engineering_dag.py

Orchestrates the data engineering + processing pipeline:
preprocessing -> Spark feature engineering -> DVC versioning
-> bounded Kafka streaming demo (replays data to prove streaming capability).

The Kafka producer task is safe to run inside Airflow because
src/preprocessing/producer.py terminates naturally once it has replayed
the CSV — it is not a long-running consumer. The consumer
(src/preprocessing/consumer.py) is intentionally NOT part of this DAG;
it runs as a standalone, always-on process outside Airflow's control.

Run manually:
    airflow dags trigger data_engineering_pipeline
"""

import os
import sys
from datetime import datetime, timedelta

# realpath() resolves symlinks (Airflow's DAGs folder often symlinks
# to the actual project), so this works regardless of who's machine
# it runs on or where their DAGs folder points to
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


def _ensure_spark_env():
    """Producer/Spark jobs need a clean Java/Spark environment — this
    avoids relying on whatever SPARK_HOME happens to be set on the
    machine running the Airflow worker."""
    os.environ.pop("SPARK_HOME", None)
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    os.environ["PATH"] = f"{os.environ['JAVA_HOME']}/bin:{os.environ['PATH']}"


def _run_preprocessing(**context):
    """Wrapped in a local import so Airflow's DAG parser doesn't need
    your full project environment just to load this file."""
    from src.preprocessing.preprocess import run_preprocessing
    run_preprocessing()


def _run_kafka_producer(**context):
    _ensure_spark_env()
    from src.preprocessing.producer import stream_csv
    stream_csv()


default_args = {
    "owner": "data-engineering-team",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="data_engineering_pipeline",
    default_args=default_args,
    description="Ingestion, cleaning, feature engineering, versioning, and a bounded Kafka demo",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data-engineering", "predictive-maintenance"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=_run_preprocessing,
    )

    spark_feature_engineering_task = BashOperator(
        task_id="spark_feature_engineering",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "unset SPARK_HOME && "
            "export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && "
            "export PATH=$JAVA_HOME/bin:$PATH && "
            "python spark_jobs/feature_engineering_job.py"
        ),
    )

    version_data_task = BashOperator(
        task_id="version_with_dvc",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "dvc add data/processed data/features && "
            "dvc commit -f && "
            "git add data/processed.dvc data/features.dvc && "
            "git commit -m 'chore: version processed data via Airflow pipeline' || true"
        ),
    )

    kafka_streaming_demo_task = PythonOperator(
        task_id="kafka_streaming_demo",
        python_callable=_run_kafka_producer,
    )

    # main batch flow
    preprocess_task >> spark_feature_engineering_task >> version_data_task

    # Kafka demo branches off after feature engineering, runs in parallel
    # with DVC versioning — it doesn't block or depend on it
    spark_feature_engineering_task >> kafka_streaming_demo_task