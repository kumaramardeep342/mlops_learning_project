# Team 54 : Predictive Maintenance of Industrial Milling Machine

<img src="project_documents/assets/AI4CPS MLOps Project.png" width="800">

``` bash
# repo owner details
GitHub ID :    kumaramardeep342 ( Global Owner) , kavyasreek9 , dhananjAI-coder , NusrathGit , sjagga001sj (Global Vice Owner)
name :         Amardeep Kumar , Kavyasree Madhuri Badimela , Dhananjai Singh ,Nusrath Fatima , Sagar Jagga 
roll_no :      DA25M502 , DA25M506 , DA25M510  ,  DA25M523     ,   DA25M526    

```

## 🎯 Overview

This repository implements an end-to-end, modular MLOps platform for predictive maintenance, organized as unified codebase encompassing the complete machine learning lifecycle. The system leverages Apache Airflow for pipeline orchestration, Apache Spark for large-scale data processing, MLflow for experiment tracking and model registry management, DVC for dataset and model versioning, FastAPI for model serving, Docker for containerization, and Prometheus/Grafana for monitoring and observability.

The codebase structure, the platform is designed with clear modular boundaries between components: data ingestion, preprocessing, feature engineering, model training, deployment, and monitoring — enabling each stage to be developed, tested, and executed independently while remaining part of a unified, version-controlled repository.

### Objectives
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
# low level design
                              GitHub
                                |
                                | Push / PR
                                v
                     +-----------------------+
                     |   GitHub Actions      |
                     |        CI/CD          |
                     +-----------+-----------+
                                 |
                                 | Trigger / Deploy
                                 v
                     +-----------------------+
                     |       Airflow         |
                     |  Workflow Orchestration|
                     +-----------+-----------+
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
       Data Validation      Spark ETL        Feature Engineering
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                        Processed Dataset
                                 |
                                 v
                         Model Training
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
       Random Forest           MLP                LSTM
        Spark ML + Ray       PyTorch            PyTorch
                            torch.compile       torch.compile
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                    +-------------------------+
                    |        MLflow            |
                    | Experiment Tracking      |
                    | Metrics → SQLite        |
                    | Model Artifacts          |
                    +-----------+-------------+
                                |
                                v
                       Model Registry
                                |
                                v
                    +-------------------------+
                    |     FastAPI Server      |
                    |                         |
                    | /health                 |
                    | /predict                |
                    | /metrics                |
                    +-----------+-------------+
                                |
                                v
                  +-----------------------------+
                  | Docker + Kubernetes         |
                  |          Minikube           |
                  +-------------+---------------+
                                |
                                v
                    +-------------------------+
                    | Prometheus + Grafana     |
                    |                         |
                    | • Request count         |
                    | • Latency               |
                    | • Errors                |
                    | • Model metrics         |
                    | • System health         |
                    +-----------+-------------+
                                |
                                v
                           Monitoring
                                |
                                |
                                +------------------+
                                                   |
                                                   v
                                           Feedback / Retraining
                                                   |
                                                   +----> Airflow

## System Architecture Diagram

<img src="project_documents/architecture/v1/System Architecture.png" width="800">

## Setup and installation instructions
###  🚀 Quick Start
```bash
# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install requirements for venv
pip install -r requirements/requirements-dev.txt
# Install requirements for uv
uv lock # run this if already uv setup in env for safe dependency installation
uv sync --all-extras --all-groups
```
## Steps to run pipelines and services

### Run Setup without Docker and Kubernetes
```bash
# 1. train the model locally.
python src/training/train.py
# output will look like below.
{"epoch": 1, "train_loss": 1.1102, "train_accuracy": 0.6205, "val_loss": 0.8658, "val_accuracy": 0.7079}
{"event": "checkpoint_saved", "path": "checkpoints/lstm.pt"}
# 2. start the fastapi server
uvicorn src.inference.serve:app --host 0.0.0.0 --port 8080 --reload
# 3. check the health endpoint
curl http://localhost:8080/health
# output will come like this
{"status":"healthy","model_loaded":true}
# 4. check the prediction endpoint , please keep the test_image.jpeg in project directory to test from local.
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"air_temperature":300.0,"process_temperature":310.0,"rotational_speed":1500,"torque":40.0,"tool_wear":100}'
# output will come like  this 
{"prediction":0,"message":"No machine failure predicted","mode":"RandomForest","accuracy":0.9765 }
```

### Kubernetes minikube services
``` bash
# setup spark in minikube
# 1. Point shell to Minikube's Docker daemon
minikube start --driver=docker  # start minikube with docker
minikube status # check status
eval $(minikube docker-env)

# 2. Build image using the specific Dockerfile path
docker build -f docker/Dockerfile -t custom-airflow-spark:latest .

# 3. Apply updated deployment and restart Airflow
kubectl apply -f airflow-deployment.yaml -n ml-training
kubectl rollout restart deployment/airflow

# 4. Stream logs to verify startup
kubectl logs -f deployment/airflow

# pointing the image to latest airflow images
# 1. Update deployment image to custom-airflow-spark:latest
kubectl set image deployment/airflow airflow=custom-airflow-spark:latest -n ml-training

# 2. Set imagePullPolicy to Never so Minikube doesn't search Docker Hub
kubectl patch deployment airflow -n ml-training -p '{"spec":{"template":{"spec":{"containers":[{"name":"airflow","imagePullPolicy":"Never"}]}}}}'

# 3. Force restart the deployment to terminate the old pod
kubectl rollout restart deployment/airflow -n ml-training

# verify the changes applied
# 1. Confirm a NEW pod is running (AGE should be a few seconds)
kubectl get pods -n ml-training -l app=airflow

# 2. Test Java version
kubectl exec -it deployment/airflow -n ml-training -- java -version

# 3. Test PySpark version
kubectl exec -it deployment/airflow -n ml-training -- python3 -c "import pyspark; print(pyspark.__version__)"

```

```bash
# work with minikube

# Apply Manifests & Deploy to Minikube
minikube start --driver=docker  # start minikube with docker
minikube status # check status

# Apply all deployment files
kubectl apply -f k8s/base/ -n ml-training

# full pipleline , not recomended due to high memory usage and system capacity below is recommened
kubectl apply -f k8s/services/ -n ml-training 
kubectl apply -f k8s/monitoring/grafana.yaml -n ml-training

# issues comes like 
# The Service "prometheus-service" is invalid: spec.ports[0].nodePort: Invalid value: 30090: provided port is already allocated
# 1. Delete the conflicting service from the default namespace
kubectl delete service prometheus-service -n default
# 2. Apply the service to the ml-training namespace
kubectl apply -f k8s/services/prometheus-service.yaml -n ml-training

# during training phase
# Turn OFF monitoring
kubectl scale deployment prometheus grafana --replicas=0
# Turn ON pipeline & tracking components
kubectl scale deployment airflow mlflow kafka zookeeper --replicas=1

# suring inference phase
# Turn OFF training & orchestration
kubectl scale deployment airflow mlflow kafka zookeeper --replicas=0 -n ml-training
# Turn ON monitoring & serving
kubectl scale deployment prometheus grafana --replicas=1 -n ml-training

# Check status of your Pods and Services
kubectl get pods -n ml-training
kubectl get svc -n ml-training

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
| **Grafana** | `http://localhost:3000` | `kubectl port-forward -n ml-training svc/grafana-service 3000:3000` | **User:** `admin` / **Password:** `admin` |
| **Prometheus** | `http://localhost:9090` | `kubectl port-forward svc/prometheus-service 9090:9090` | None |
| **Kafka** | `localhost:9092` | `kubectl port-forward svc/kafka-service 9092:9092` | Internal Broker Port |

## API usage instructions
### Kubernetes Training Job + Kubernetes Model Serving +  End-to-End Validation

``` bash 
# 1. Start Minikube , don't run in python env/project path , run in nomral WSL , once minikube started execute the command from step 2 from  python env/project path  becuase it need Dockerfile.train and Dockerfile.serve to rebuild the images.
minikube start

# 2. Point local Docker CLI to Minikube's Docker daemon
eval $(minikube -p minikube docker-env)

# 3. Rebuild your Docker images inside Minikube's Docker daemon
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# 2. Rebuild training image with no-cache to force config update
docker build --no-cache -f docker/Dockerfile.train -t mlops-train:v1 .
docker build --no-cache -f docker/Dockerfile.serve -t mlops-serve:v1 .

```

```bash
# 1. Run a temporary helper container mounting mlops-pvc as root
kubectl run pvc-fixer --image=alpine --restart=Never -n ml-training \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "pvc-fixer",
        "image": "alpine",
        "command": ["sleep", "3600"],
        "securityContext": {"runAsUser": 0},
        "volumeMounts": [{"name": "storage", "mountPath": "/app/data"}]
      }],
      "volumes": [{"name": "storage", "persistentVolumeClaim": {"claimName": "mlops-pvc"}}]
    }
  }'

# 2. Wait until the helper pod is running
kubectl wait --for=condition=Ready pod/pvc-fixer -n ml-training --timeout=30s

# 3. Create directory and adjust permissions
kubectl exec -n ml-training pvc-fixer -- mkdir -p /app/data/processed
kubectl exec -n ml-training pvc-fixer -- chmod -R 777 /app/data

# 4. Copy data_features.csv into the PVC
kubectl cp data/processed/data_features.csv ml-training/pvc-fixer:/app/data/processed/data_features.csv

# 5. Clean up the helper pod
kubectl delete pod pvc-fixer -n ml-training

```

```bash 
# Demonstrate the full workflow running on your Kubernetes cluster:
# run training pipeline at once 
kubectl apply -f k8s/pipeline/ -n ml-training

# 1 Apply all manifests , skip if you have already ran k8s/base
kubectl apply -f k8s/base/namespace.yaml -n ml-training
kubectl apply -f k8s/base/configmap.yaml -n ml-training
kubectl apply -f k8s/pipeline/training-job.yaml -n ml-training

# Wait for Training to Finish...
# Watch job progress until COMPLETIONS shows 1/1
kubectl get job mlops-train-job -n ml-training -w
# (Press Ctrl+C once COMPLETIONS shows 1/1).
# Check Live Training Logs:
kubectl logs -f job/mlops-train-job -n ml-training
# Check Pod Health & Events:
kubectl get pods -n ml-training
# once succesfully finishes then move to setp 2 below.

# 2 Once training completes, deploy the serving layer , skip you have already ran k8s/pipeline/
kubectl apply -f k8s/pipeline/serving-deployment.yaml -n ml-training
kubectl apply -f k8s/pipeline/serving-service.yaml -n ml-training
kubectl apply -f k8s/pipeline/hpa.yaml -n ml-training

# 3 Verify pods are running and healthy:
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 4 Test the prediction endpoint:
# Port-forward for local testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"air_temperature":300.0,"process_temperature":310.0,"rotational_speed":1500,"torque":40.0,"tool_wear":100}'

# 5. if issue come port 8080 is used and  Unable to listen on port 8080 then try below one

kubectl port-forward svc/model-serving 8081:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"air_temperature":300.0,"process_temperature":310.0,"rotational_speed":1500,"torque":40.0,"tool_wear":100}'
```

``` bash
# get the grafana password
kubectl exec -it deployment/grafana -n ml-training -- env | grep -i admin
# Model Serving API
kubectl port-forward svc/model-serving 8080:80 -n ml-training
# Grafana Dashboard
kubectl port-forward svc/grafana-service 3000:3000 -n ml-training
# or,
minikube service grafana-service --url
# Prometheus UI
minikube service prometheus-service --url
# or,
kubectl port-forward svc/prometheus-service 9090:9090 -n ml-training
# Total predictions served
http_requests_total{handler="/predict"}
# Prediction request rate (per second over 5 min):
rate(http_requests_total{handler="/predict"}[5m])
# Average prediction duration (seconds)
rate(http_request_duration_seconds_sum{handler="/predict"}[5m]) / rate(http_request_duration_seconds_count{handler="/predict"}[5m])
# check metrics in terminal
curl http://localhost:8080/metrics
# to monitor 
for i in {1..20}; do   curl -s -X POST http://localhost:8080/predict     -H "Content-Type: appli
cation/json"     -d '{"air_temperature": 298.1, "process_temperature": 308.6, "rotational_speed": 1500, "torque": 40.0, "too
l_wear": 0}' > /dev/null;   echo "Request $i sent";   sleep 1; done


## Docker execution commands
### Docker Run
```bash
cd docker
docker compose down
docker compose up -d
docker logs airflow # check the logs of docker
```

### Exposed Endpoints
use `docker log` to get the Airflow password.
```pliantext
| Service    | URL                                            | Login Detail                |
| ---------- | ---------------------------------------------- | --------------------------- |
| Airflow    | [http://localhost:8080](http://localhost:8080) | admin :                     |
| MLflow     | [http://localhost:5000](http://localhost:5000) |                             |
| Prometheus | [http://localhost:9090](http://localhost:9090) |                             |
| Grafana    | [http://localhost:3000](http://localhost:3000) | admin : admin               |
```
### Docker Containerization + Serving 

```bash
# verify docker daemon , docker is running fine :
docker info
# 1.Build and test the training image:
# Build image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
# Run training with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

#2.build and test the serving image
# Build image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving container
docker run -d --rm -p 8080:8080 \
  --name serve-container \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

#3.Test prediction endpoint
# Check health
curl http://localhost:8080/health
# Send test image for prediction
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"air_temperature":300.0,"process_temperature":310.0,"rotational_speed":1500,"torque":40.0,"tool_wear":100}'

```


## Folder structure and dependencies
### 📁 Repository Architecture
```plaintext
predictive-maintenance-mlops/
│
├── .github/  
│   └── ISSUE_TEMPLATE/          
│   └── workflows/
│   │    ├── ci.yml
│   │    └── cd.yml
├── ── CODEOWNERS
├── ── CODEOWNERS
│
├── airflow/
│   ├── dags/
│   │   ├── data_ENGINEERING_dag.py
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   ├── processed/
│        └── data_features.csv
│
├── spark_jobs/
│   ├── feature_engineering_job.py
│
├── notebooks/                 
│   ├── Predictive Maintenance for Industrial Milling Machines.ipynb
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── consumer.py
│   │   └── prodcuer.py
│   │
│   ├── preprocessing/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │
│   ├── training/
│   │   ├── model.py
│   │   ├── train.py
│   │
│   ├── inference/
│   │   ├── serve.py
│   │
│   └── utils/              
│       ├── logger.py
│       └── setup_project_strcuture.py
│
├── checkpoints/                     
│   ├── best_model/
│   ├── lstm.pt
│   │── mlp.pt
│   └── model_metrics/
│
├── mlflow.db             
│
├── .dvc/
│
├── monitoring/            
│   └── logs/
│       └── application.log
│
├── tests/                        
│   ├── integration/
│   ├── unit/
│
├── docker/                
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── Dockerfile.train
│   └── Dockerfile.serve
│   └── .dockerignore
│
├── k8s/ 
│
├── configs/               
│   ├── grafana_configmap.yaml
│   ├── spark_config.yaml
│   ├── training_config.yaml
│
├── project_documents/      
│   ├── architecture/v1/
│   │   ├── HLD.png
│   │   ├── LLD.png
│   │   └── System_Architecture.png
│   │
│   ├── assets/
│   ├── guides/
│   └── reports/
│   └── setup/                  
├── .gitignore
├── .dvcignore
├── .gitattributes
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements/
```

### Kubernetes Strcuture 
```plaintext
configs/
├── grafana_configmap.yaml
├── spark_config.yaml
└── training_config.yaml

k8s/
├── base/
│   ├── configmap.yaml
│   ├── kafka-configmap.yaml
│   ├── mlflow-configmap.yaml
│   ├── namespace.yaml
│   └── prometheus-configmap.yaml
├── pipeline/
│   ├── hpa.yaml
│   ├── prometheus-rbac.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── training-job.yaml
├── monitoring/
│   ├── grafana.yaml.yaml
└── services/
    ├── airflow-deployment.yaml
    ├── airflow-service.yaml
    ├── kafka-deployment.yaml
    ├── kafka-service.yaml
    ├── mlflow-deployment.yaml
    ├── mlflow-service.yaml
    ├── prometheus-deployment.yaml
    └── prometheus-service.yaml
    └── pvc.yaml
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