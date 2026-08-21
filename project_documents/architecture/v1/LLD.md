# low level design
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