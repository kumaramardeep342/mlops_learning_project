# Team 54 : Predictive Maintenance of Industrial Milling Machine

``` bash
# repo owner details
GitHub ID : sjagga001sj , NusrathGit , dhananjAI-coder , kavyasreek9 ,  kumaramardeep342
name :      Sagar Jagga , Nusrath Fatima , Dhananjai Singh , Kavyasree, Amardeep Kumar
roll_no :   DA25M526    , DA25M523       , DA25M510       , DA25M506 ,  DA25M502

```

## 🎯 Overview

This repository implements an end-to-end, modular MLOps platform for predictive maintenance, organized as unified codebase encompassing the complete machine learning lifecycle. The system leverages Apache Airflow for pipeline orchestration, Apache Spark for large-scale data processing, MLflow for experiment tracking and model registry management, DVC for dataset and model versioning, FastAPI for model serving, Docker for containerization, and Prometheus/Grafana for monitoring and observability.

The codebase structure, the platform is designed with clear modular boundaries between components: data ingestion, preprocessing, feature engineering, model training, deployment, and monitoring — enabling each stage to be developed, tested, and executed independently while remaining part of a unified, version-controlled repository.

### 🎓 Learning Objectives
1. Structure an ML project repository with proper Git practices (branching, gitlfs , PRs, .gitignore , github actions , issues ,  secrets and project management)
2. End to End Data Engineering Pipeline
3. Write multi-stage Dockerfiles optimized for ML workloads
4. Deploy PyTorch training jobs on Kubernetes using Jobs and persistent storage
5. Serve a trained model via a Kubernetes Deployment with health checks
6. Use ConfigMaps and Secrets for environment-specific configuration
7. Metrics analyzed via prometheus and grafana.

### 📋 Prerequisites

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

## System architecture
                            GitHub
                                |
                                v
                    +-----------------------+
                    | GitHub Actions CI/CD  |
                    +-----------+-----------+
                                |
                                v
    +-----------------------------------------------------------+
    |                       Apache Airflow                      |
    +-----------------------------------------------------------+
            |                  |                     |
            v                  v                     v

    Data Validation   Spark ETL Pipeline   Feature Engineering

            |                  |                     |
            +------------------+---------------------+
                            |
                            v

                    Processed Dataset

                            |
                            v

                    Model Training

                            |
            +----------------+---------------+
            |                                |
            v                                v

        Random Forest                    XGBoost

            |                                |
            +----------------+---------------+
                            |
                            v

                        MLflow Server

                            |
                            v

                        DVC Repository

                            |
                            v

                        Model Registry

                            |
                            v

                        FastAPI Server

                            |
                        Docker Container

                            |
                            v

                        Prometheus Server

                            |
                            v

                        Monitoring

## System architecture diagram

<img src="project_documents/assets/AI4CPS MLOps Project.png" width="800">

## Setup and installation instructions
###  🚀 Quick Start
```bash
# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install training requirements
pip install -r requirements-dev.txt
# Install serving requirements
```
## Steps to run pipelines and services

### Kubernetes minikube services
```bash
# work with minikube

# Apply Manifests & Deploy to Minikube
minikube start --driver=docker  # start minikube with docker
minikube status # check status

# Apply all deployment files
kubectl apply -f k8s/base/

# full pipleline , not recomended due to high memory usage and system capacity below is recommened
kubectl apply -f k8s/services/

# during training phase
# Turn OFF monitoring
kubectl scale deployment prometheus grafana --replicas=0
# Turn ON pipeline & tracking components
kubectl scale deployment airflow mlflow kafka zookeeper --replicas=1

# suring inference phase
# Turn OFF training & orchestration
kubectl scale deployment airflow mlflow kafka zookeeper --replicas=0
# Turn ON monitoring & serving
kubectl scale deployment prometheus grafana --replicas=1

# run training pipeline
kubectl apply -f k8s/pipeline/

# Check status of your Pods and Services
kubectl get pods
kubectl get svc

# Access UI services on your host machine
# run the following commands in separate terminal windows to get the local URLs:
# Mlflow UI
minikube service mlflow-service --url
# Airflow UI
minikube service airflow-service --url
# Prometheus UI
minikube service prometheus-service --url
# Grafana UI
minikube service grafana-service --url
```
### Exposed Endpoints
| Service | Local URL | Port-Forward Command | Default Credentials |
| :--- | :--- | :--- | :--- |
| **MLflow** | `http://localhost:5000` | `kubectl port-forward svc/mlflow-service 5000:5000` | None |
| **Airflow** | `http://localhost:8080` | `kubectl port-forward svc/airflow-service 8080:8080` | None *(Standalone mode auto-logs in)* |
| **Grafana** | `http://localhost:3000` | `kubectl port-forward svc/grafana-service 3000:3000` | **User:** `admin` / **Password:** `admin` |
| **Prometheus** | `http://localhost:9090` | `kubectl port-forward svc/prometheus-service 9090:9090` | None |
| **Kafka** | `localhost:9092` | `kubectl port-forward svc/kafka-service 9092:9092` | Internal Broker Port |

## API usage instructions
    TODO

## Docker execution commands
### Docker Run
```bash
cd docker
docker compose down
docker compose up -d
docker logs airflow # check the logs of docker
```

### Exposed Endpoints
```pliantext
| Service    | URL                                            | Login Detail                |
| ---------- | ---------------------------------------------- | --------------------------- |
| Airflow    | [http://localhost:8080](http://localhost:8080) | admin :                     |
| MLflow     | [http://localhost:5000](http://localhost:5000) |                             |
| Prometheus | [http://localhost:9090](http://localhost:9090) |                             |
| Grafana    | [http://localhost:3000](http://localhost:3000) | admin : admin               |
```

## Folder structure and dependencies
### 📁 Repository Architecture
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
├── k8s/ 
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

### Kubernetes Strcuture 
```plaintext
configs/
├── airflow_config.yaml
├── spark_config.yaml
└── training_config.yaml

k8s/
├── base/
│   ├── configmap.yaml
│   ├── grafana-configmap.yaml
│   ├── kafka-configmap.yaml
│   ├── mlflow-configmap.yaml
│   ├── namespace.yaml
│   └── prometheus-configmap.yaml
├── pipeline/
│   ├── hpa.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── training-job.yaml
└── services/
    ├── airflow-deployment.yaml
    ├── airflow-service.yaml
    ├── grafana-deployment.yaml
    ├── grafana-service.yaml
    ├── kafka-deployment.yaml
    ├── kafka-service.yaml
    ├── mlflow-deployment.yaml
    ├── mlflow-service.yaml
    ├── prometheus-deployment.yaml
    └── prometheus-service.yaml
```
### Dependencies
<img src="project_documents/setup/dependency strcuture.png" width="800">

## 🤝 Contributing Guidelines
We welcome contributions! To keep our codebase clean and maintain a smooth workflow, please strictly follow these step-by-step contribution guidelines.

---

### Issue & Branch Creation
1. **Find or Create an Issue:** All work must be linked to an issue. If no issue exists, create one describing the bug or feature.
2. **Assign Labels & Project:** Ensure the issue has appropriate **Labels** (e.g., `bug`, `feature`, `documentation`) and is added to the repository **GitHub Project**.
3. **Branch from `develop`:** Create a new branch directly from the `develop` branch using the issue number and a descriptive name:
```bash
   git checkout develop
   git pull origin develop
   git checkout -b <issue-number>-<short-description>
   # Example: git checkout -b 42-add-feature-engineering
```

### Development & Code Formatting
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

###  Creating a Pull Request (PR)
```bash
# Fill all the required details propely , title , description , reviewer , label , project.
```