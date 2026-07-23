                    Windows 11
                         │
              Docker Desktop Installed
                         │
────────────────────────────────────────────────────────

                    WSL Ubuntu 24.04
                         │
         ┌───────────────┴────────────────┐
         │                                │
      Project Code                  uv Virtual Env
         │                                │
         ├── Python 3.12
         ├── Spark 3.5.6
         ├── DVC
         ├── Git
         └── VS Code Remote

                         │
                 Docker Containers
                         │
         ├── Kafka
         ├── Airflow
         ├── MLflow Tracking Server
         ├── Prometheus
         └── Grafana

                         │
──────────────────────────────────────────────

                 GitHub Repository
           (Code + DVC Metadata only)

                         │

                 Google Drive
          (Datasets + Models via DVC)