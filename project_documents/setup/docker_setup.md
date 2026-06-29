# Step 1 : Go to docker folder
```
cd ~/projects/mlops_project/mlops_learning_project/docker
pwd
ls
```

```
output :
Dockerfile
docker-compose.yml
.dockerignore
```

# Step 2 : Validate compose file
```
docker compose config
```

# Step 3 : Pull images (downloads containers)
```
docker compose pull
```

```
output :
apache/airflow:3.0.2
confluentinc/cp-zookeeper:7.6.0
confluentinc/cp-kafka:7.6.0
ghcr.io/mlflow/mlflow:v3.1.0
prom/prometheus
grafana/grafana
```

```
docker images
```

# Step 4 : Start services
```
docker compose up -d
docker ps
```
```
output :
airflow
zookeeper
kafka
mlflow
prometheus
grafana
```
# step 5 : URL 

| Service    | URL                                            |
| ---------- | ---------------------------------------------- |
| Airflow    | [http://localhost:8080](http://localhost:8080) |
| MLflow     | [http://localhost:5000](http://localhost:5000) |
| Prometheus | [http://localhost:9090](http://localhost:9090) |
| Grafana    | [http://localhost:3000](http://localhost:3000) |

# Step 6 : Stop everything
```
docker compose down
docker compose down -v # this remove volume as well , deletes conatiner data don't run danger.
```