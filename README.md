# mlops_learning_project

This repository implements an end-to-end, modular MLOps platform for predictive maintenance, organized as a single monolithic codebase encompassing the complete machine learning lifecycle. The system leverages Apache Airflow for pipeline orchestration, Apache Spark for large-scale data processing, MLflow for experiment tracking and model registry management, DVC for dataset and model versioning, FastAPI for model serving, Docker for containerization, and Prometheus/Grafana for monitoring and observability.

Despite its monolithic structure, the platform is designed with clear modular boundaries between components — data ingestion, preprocessing, feature engineering, model training, deployment, and monitoring — enabling each stage to be developed, tested, and executed independently while remaining part of a unified, version-controlled repository.

# detail project strcuture

predictive-maintenance-mlops/
│
├── .github/            # inprogress
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── airflow/
│   ├── dags/
│   │   ├── data_ingestion_dag.py
│   │   ├── preprocessing_dag.py
│   │   ├── training_dag.py
│   │   └── deployment_dag.py
│   │
│   ├── plugins/
│   │
│   └── logs/
│
├── data/
│   ├── raw/
│   │   └── maintenance.csv
│   │
│   ├── interim/
│   │   └── cleaned_data.parquet
│   │
│   ├── processed/
│   │   └── feature_engineered.parquet
│   │
│   └── validation/
│       └── schema.json
│
├── spark_jobs/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── imbalance_handling.py
│   └── spark_utils.py
│
├── notebooks/                 # done
│   ├── EDA.ipynb
│   ├── Feature_Analysis.ipynb
│   └── Model_Experiments.ipynb
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── ingest.py
│   │   └── validate_schema.py
│   │
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── handle_missing.py
│   │   ├── scaling.py
│   │   └── encoding.py
│   │
│   ├── feature_engineering/
│   │   ├── feature_generator.py
│   │   └── feature_selector.py
│   │
│   ├── training/
│   │   ├── train_rf.py
│   │   ├── train_xgb.py
│   │   ├── evaluate.py
│   │   └── hyperparameter_tuning.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── model_loader.py
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py
│   │   ├── logging_service.py
│   │   └── metrics.py
│   │
│   └── utils/              # done
│       ├── logger.py
│       ├── config.py
│       └── helper.py
│
├── models/                 # done     
│   ├── random_forest/
│   │   └── model.pkl
│   │
│   ├── xgboost/
│   │   └── model.pkl
│   │
│   └── production/
│       └── current_model.pkl
│
├── mlflow/             # done
│   ├── artifacts/
│   └── mlruns/
│
├── dvc/
│   ├── data.dvc
│   ├── model.dvc
│   └── dvc.yaml
│
├── app/
│   ├── app.py
│   ├── schemas.py
│   ├── routes.py
│   ├── middleware.py
│   └── requirements.txt
│
├── monitoring/            # done
│   ├── prometheus/
│   │   └── prometheus.yml
│   │
│   ├── dashboards/
│   │   └── machine_failure_dashboard.json
│   │
│   └── logs/
│       └── application.log
│
├── tests/                  # done       
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_api.py
│   └── test_monitoring.py
│
├── docker/                # done
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── configs/               # done
│   ├── airflow_config.yaml
│   ├── spark_config.yaml
│   ├── model_config.yaml
│   └── deployment_config.yaml
│
├── project_documents/      # done
│   ├── architecture/
│   │   ├── HLD.png
│   │   ├── LLD.png
│   │   └── System_Architecture.png
│   │
│   ├── model_metrics/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── feature_importance.png
│   │
│   └── project_report.pdf
│
├── scripts/
│   ├── run_pipeline.sh
│   ├── train_model.sh
│   ├── deploy.sh
│   └── monitor.sh
│
├── .env                    # done
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── Makefile

# Project Architecture
<img src="project_documents/assets/AI4CPS MLOps Project.png" width="800">

# docker run
```
cd docker
docker compose down
docker compose up -d
docker logs airflow # check the logs of docker
```

| Service    | URL                                            | Login Detail                |
| ---------- | ---------------------------------------------- | --------------------------- |
| Airflow    | [http://localhost:8080](http://localhost:8080) | admin : BD3fE85nVqmH3X73    |
| MLflow     | [http://localhost:5000](http://localhost:5000) |                             |
| Prometheus | [http://localhost:9090](http://localhost:9090) |                             |
| Grafana    | [http://localhost:3000](http://localhost:3000) | admin : admin               |
