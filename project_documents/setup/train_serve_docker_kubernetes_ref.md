## Without Docker and Kubernetes
```bash
# 1. train the model locally.
python src/train.py
# output will look like below.
{"epoch": 1, "train_loss": 1.1102, "train_accuracy": 0.6205, "val_loss": 0.8658, "val_accuracy": 0.7079}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt"}
# 2. start the fastapi server
uvicorn src.serve:app --host 0.0.0.0 --port 8080 --reload
# 3. check the health endpoint
curl http://localhost:8080/health
# output will come like this
{"status":"healthy","model_loaded":true}
# 4. check the prediction endpoint , please keep the test_image.jpeg in project directory to test from local.
curl -X POST http://localhost:8080/predict -F "image=@test_image.jpeg"
# output will come like  this 
{"filename":"test_frog_image.jpeg","prediction":"frog","confidence":1.0,"probabilities":{"airplane":0.0,"automobile":0.0,"bird":0.0,"cat":0.0,"deer":0.0,"dog":0.0,"frog":1.0,"horse":0.0,"ship":0.0,"truck":0.0}}
```

## Docker Containerization + Serving 

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
curl -X POST http://localhost:8080/predict -F "image=@test_image.jpeg"

```
## Kubernetes Training Job + Kubernetes Model Serving +  End-to-End Validation

```bash 
# 1. Start Minikube , don't run in python env/project path , run in nomral WSL , once minikube started execute the command from step 2 from  python env/project path  becuase it need Dockerfile.train and Dockerfile.serve to rebuild the images.
minikube start

# 2. Point local Docker CLI to Minikube's Docker daemon
eval $(minikube -p minikube docker-env)

# 3. Rebuild your Docker images inside Minikube's Docker daemon
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

```bash 
# Demonstrate the full workflow running on your Kubernetes cluster:
# 1 Apply all manifests:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# Wait for Training to Finish...
# Watch job progress until COMPLETIONS shows 1/1
kubectl get job mlops-train-job -n ml-training -w
# (Press Ctrl+C once COMPLETIONS shows 1/1).
# Check Live Training Logs:
kubectl logs -f job/mlops-train-job -n ml-training
# Check Pod Health & Events:
kubectl get pods -n ml-training
# once succesfully finishes then move to setp 2 below.

# 2 Once training completes, deploy the serving layer:
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# 3 Verify pods are running and healthy:
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# 4 Test the prediction endpoint:
# Port-forward for local testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8080/predict -F "image=@test_image.jpeg"

# 5. if issue come port 8080 is used and  Unable to listen on port 8080 then try below one

kubectl port-forward svc/model-serving 8081:80 -n ml-training
# Send a prediction request
curl -X POST http://localhost:8081/predict -F "image=@test_image.jpeg"
```

```bash
# useful kubectl commands
# describe pad 
kubectl describe pod airflow-59f8ff95f-rq6f4
# delete pod
kubectl delete pod mlflow-697784bf8b-bgz44
# restart the deployment
kubectl rollout restart deployment/mlflow
# delete the deployment
kubectl delete deployment mlflow