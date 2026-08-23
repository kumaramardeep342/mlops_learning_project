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