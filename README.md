# Predictive Maintenance

``` bash
# repo owner details
GitHub ID : sjagga001sj , NusrathGit , dhananjAI-coder , kavyasreek9 ,  kumaramardeep342
name :      Sagar Jagga , Nusrath Fatima , Dhananjai Singh , Kavyasree, Amardeep Kumar
roll_no :   DA25M526    , DA25M523       , DA25M510       , DA25M506 ,  DA25M502

```

## 🎯 Overview

This repository implements an end-to-end, modular MLOps platform for predictive maintenance, organized as a single monolithic codebase encompassing the complete machine learning lifecycle. The system leverages Apache Airflow for pipeline orchestration, Apache Spark for large-scale data processing, MLflow for experiment tracking and model registry management, DVC for dataset and model versioning, FastAPI for model serving, Docker for containerization, and Prometheus/Grafana for monitoring and observability.

Despite its monolithic structure, the platform is designed with clear modular boundaries between components — data ingestion, preprocessing, feature engineering, model training, deployment, and monitoring — enabling each stage to be developed, tested, and executed independently while remaining part of a unified, version-controlled repository.

## 🎓 Learning Objectives
1. Structure an ML project repository with proper Git practices (branching, PRs, .gitignore secrets management)
2. End to End Data Engineering Pipeline
3. Write multi-stage Dockerfiles optimized for ML workloads
4. Deploy PyTorch training jobs on Kubernetes using Jobs and persistent storage
5. Serve a trained model via a Kubernetes Deployment with health checks
6. Use ConfigMaps and Secrets for environment-specific configuration

## 📋 Prerequisites

| Requirement | Specified Version / Component |
| :--- | :--- |
| OS / Environment | Windows 11 + WSL2 (Ubuntu) |
| Java| OpenJDK 17 (`JAVA_HOME`) |
| Python Tooling | Python 3.12 (via `uv venv`) |
| Big Data Engine | Apache Spark 3.5.6 (`SPARK_HOME`) |
| Container Engine | Docker Desktop with WSL2 integration |
| Kubernetes CLI | `kubectl` + `minikube` (Docker driver) |
| Data Tracking | Git LFS + DVC |
| Services Stack | Airflow, MLflow, Kafka, Prometheus, Grafana |

## 📁 Repository Architecture

```plaintext
predictive-maintenance-mlops/
│
├── .github/            
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
├── notebooks/                 
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
│       └── setup_project_strcuture.py
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
├── mlflow/             
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
├── monitoring/            
│   ├── prometheus/
│   │   └── prometheus.yml
│   │
│   ├── dashboards/
│   │   └── machine_failure_dashboard.json
│   │
│   └── logs/
│       └── application.log
│
├── tests/                        
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_api.py
│   └── test_monitoring.py
│
├── docker/                
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── configs/               
│   ├── airflow_config.yaml
│   ├── spark_config.yaml
│   ├── model_config.yaml
│   └── deployment_config.yaml
│
├── project_documents/      
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
├── .env                    
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── Makefile
```

## Architecture
<img src="project_documents/assets/AI4CPS MLOps Project.png" width="800">

##  🚀 Quick Start
```bash
# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install training requirements
pip install -r requirements-dev.txt
# Install serving requirements
pip install -r requirements-dev.txt
```

## Docker Run
```bash
cd docker
docker compose down
docker compose up -d
docker logs airflow # check the logs of docker
minikube start --driver=docker  # start minikube with docker
minikube status # check status
```

## Exposed Endpoints
```pliantext
| Service    | URL                                            | Login Detail                |
| ---------- | ---------------------------------------------- | --------------------------- |
| Airflow    | [http://localhost:8080](http://localhost:8080) | admin :                     |
| MLflow     | [http://localhost:5000](http://localhost:5000) |                             |
| Prometheus | [http://localhost:9090](http://localhost:9090) |                             |
| Grafana    | [http://localhost:3000](http://localhost:3000) | admin : admin               |
```

## 🤝 Contributing Guidelines
We welcome contributions! To keep our codebase clean and maintain a smooth workflow, please strictly follow these step-by-step contribution guidelines.

---

### 1. Issue & Branch Creation
1. **Find or Create an Issue:** All work must be linked to an issue. If no issue exists, create one describing the bug or feature.
2. **Assign Labels & Project:** Ensure the issue has appropriate **Labels** (e.g., `bug`, `feature`, `documentation`) and is added to the repository **GitHub Project**.
3. **Branch from `develop`:** Create a new branch directly from the `develop` branch using the issue number and a descriptive name:
```bash
   git checkout develop
   git pull origin develop
   git checkout -b <issue-number>-<short-description>
   # Example: git checkout -b 42-add-feature-engineering
```

### 2. Development & Code Formatting
```bash
# 1. Write clean, modular, and documented code following the project structure.

# 2. Format Code with Ruff: Before committing, you must format all Python code using ruff to ensure consistency:

# Install ruff
uv pip install ruff
# Format code
ruff format .
# Check for lint issues
ruff check .

# 3. Synchronize Before Raising a PR Always pull the latest changes from develop into your feature branch before opening a Pull Request to prevent merge conflicts:
git checkout <your-branch-name>
git pull origin develop
git push origin <your-branch-name>
```

###  4. Creating a Pull Request (PR)
```bash
# Fill all the required details propely , title , description , reviewer , label , project.
```

