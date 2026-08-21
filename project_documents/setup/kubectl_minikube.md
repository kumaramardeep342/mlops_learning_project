```bash
# kubectl setup
# install the current stable Linux binary:
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
#remove the downloaded file
rm kubectl
kubectl version --client
# it will come like this Client Version: ...
```

```bash
# Install Minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
#remove the downloaded file
rm minikube-linux-amd64
minikube version
# it will like this minikube version: v1.38.1
minikube start --driver=docker  # start minikube with docker
minikube status # check status
# it should come lime this.
# type: Control Plane
# host: Running
# kubelet: Running
# apiserver: Running
# kubeconfig: Configured
```

```bash
# work with minikube

# Apply Manifests & Deploy to Minikube
minikube start --driver=docker

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

| Service | Local URL | Port-Forward Command | Default Credentials |
| :--- | :--- | :--- | :--- |
| **MLflow** | `http://localhost:5000` | `kubectl port-forward svc/mlflow-service 5000:5000` | None |
| **Airflow** | `http://localhost:8080` | `kubectl port-forward svc/airflow-service 8080:8080` | None *(Standalone mode auto-logs in)* |
| **Grafana** | `http://localhost:3000` | `kubectl port-forward svc/grafana-service 3000:3000` | **User:** `admin` / **Password:** `admin` |
| **Prometheus** | `http://localhost:9090` | `kubectl port-forward svc/prometheus-service 9090:9090` | None |
| **Kafka** | `localhost:9092` | `kubectl port-forward svc/kafka-service 9092:9092` | Internal Broker Port |
