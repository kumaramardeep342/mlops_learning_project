from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
)

with DAG(
    dag_id="feature_engineering_dag",
    start_date=datetime(2026, 8, 23),
    schedule=None,
    catchup=False,
) as dag:

    feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command=(
                    "cd /opt/airflow/project && "
                    "python spark_jobs/feature_engineering_job.py"
        ),
    )