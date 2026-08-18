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