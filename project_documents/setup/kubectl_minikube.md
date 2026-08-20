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
kubectl apply -f k8s/
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