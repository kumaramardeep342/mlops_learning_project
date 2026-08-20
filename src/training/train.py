"""
src/train.py

Training pipeline:

1. MLP       -> PyTorch + torch.compile
2. LSTM      -> PyTorch + torch.compile
3. RF        -> Spark ML
4. Ray       -> parallel RF experiments
5. MLflow    -> experiment tracking
"""

import time
from pathlib import Path

import mlflow
import ray
import torch
import torch.nn as nn
import yaml

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from src.training.model import (
    MLP,
    LSTMModel,
    get_random_forest
)

from src.preprocessing.dataset import get_dataset


from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# Configuration
# ============================================================

def load_config(path):

    with open(path, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# Generic PyTorch Training
# ============================================================

def train_pytorch_model(
    model,
    train_loader,
    epochs,
    device,
    is_lstm=False
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            # LSTM expects:
            # [batch, sequence, features]
            if is_lstm and inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        logger.info(
            f"Epoch {epoch + 1}/{epochs} "
            f"loss={total_loss:.4f}"
        )


# ============================================================
# MLP
# ============================================================

def train_mlp(config):

    logger.info("Starting MLP training")

    start_time = time.time()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, _ = get_dataset(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"]
    )

    # Find number of input features
    sample_x, _ = next(
        iter(train_loader)
    )

    input_size = sample_x.shape[-1]

    model = MLP(
        input_size=input_size,
        hidden_size=config["mlp"]["hidden_size"]
    )

    model = model.to(device)

    # --------------------------------------------------------
    # torch.compile
    # --------------------------------------------------------

    if config["torch_compile"]["enabled"]:

        logger.info(
            "Applying torch.compile() to MLP"
        )

        model = torch.compile(
            model,
            mode=config["torch_compile"]["mode"]
        )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="MLP"
    ):

        mlflow.log_param(
            "model",
            "MLP"
        )

        mlflow.log_param(
            "hidden_size",
            config["mlp"]["hidden_size"]
        )

        mlflow.log_param(
            "epochs",
            config["training"]["epochs"]
        )

        mlflow.log_param(
            "torch_compile",
            config["torch_compile"]["enabled"]
        )

        train_pytorch_model(
            model=model,
            train_loader=train_loader,
            epochs=config["training"]["epochs"],
            device=device
        )

        training_time = (
            time.time() - start_time
        )

        mlflow.log_metric(
            "training_time",
            training_time
        )

    logger.info(
        f"MLP completed in "
        f"{training_time:.2f} seconds"
    )


# ============================================================
# LSTM
# ============================================================

def train_lstm(config):

    logger.info("Starting LSTM training")

    start_time = time.time()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, _ = get_dataset(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"]
    )

    sample_x, _ = next(
        iter(train_loader)
    )

    # If data is:
    # [batch, features]
    #
    # sequence length = 1

    if sample_x.dim() == 2:

        input_size = sample_x.shape[1]

    else:

        input_size = sample_x.shape[2]

    model = LSTMModel(
        input_size=input_size,
        hidden_size=config["lstm"]["hidden_size"],
        num_layers=config["lstm"]["num_layers"]
    )

    model = model.to(device)

    # --------------------------------------------------------
    # torch.compile
    # --------------------------------------------------------

    if config["torch_compile"]["enabled"]:

        logger.info(
            "Applying torch.compile() to LSTM"
        )

        model = torch.compile(
            model,
            mode=config["torch_compile"]["mode"]
        )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="LSTM"
    ):

        mlflow.log_param(
            "model",
            "LSTM"
        )

        mlflow.log_param(
            "hidden_size",
            config["lstm"]["hidden_size"]
        )

        mlflow.log_param(
            "num_layers",
            config["lstm"]["num_layers"]
        )

        mlflow.log_param(
            "epochs",
            config["training"]["epochs"]
        )

        mlflow.log_param(
            "torch_compile",
            config["torch_compile"]["enabled"]
        )

        train_pytorch_model(
            model=model,
            train_loader=train_loader,
            epochs=config["training"]["epochs"],
            device=device,
            is_lstm=True
        )

        training_time = (
            time.time() - start_time
        )

        mlflow.log_metric(
            "training_time",
            training_time
        )

    logger.info(
        f"LSTM completed in "
        f"{training_time:.2f} seconds"
    )


# ============================================================
# Random Forest
# ============================================================

@ray.remote(num_cpus=1)
def train_random_forest(
    config,
    num_trees,
    max_depth
):

    logger.info(
        f"Starting RF trial: "
        f"trees={num_trees}, "
        f"depth={max_depth}"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Spark
    # --------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName(
            f"RF_{num_trees}_{max_depth}"
        )
        .master("local[1]")
        .getOrCreate()
    )

    try:

        # ----------------------------------------------------
        # Load data
        # ----------------------------------------------------

        df = spark.read.csv(
            config["data"]["data_path"],
            header=True,
            inferSchema=True
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        assembler = VectorAssembler(
            inputCols=config["data"]["feature_columns"],
            outputCol="features"
        )

        data = assembler.transform(df)

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = get_random_forest(
            num_trees=num_trees,
            max_depth=max_depth
        )

        # ----------------------------------------------------
        # MLflow
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=(
                f"RF_{num_trees}_{max_depth}"
            )
        ):

            mlflow.log_param(
                "model",
                "RandomForest"
            )

            mlflow.log_param(
                "num_trees",
                num_trees
            )

            mlflow.log_param(
                "max_depth",
                max_depth
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(data)

            training_time = (
                time.time() - start_time
            )

            mlflow.log_metric(
                "training_time",
                training_time
            )

        logger.info(
            f"RF completed: "
            f"trees={num_trees}, "
            f"depth={max_depth}"
        )

        return {
            "model": "RandomForest",
            "num_trees": num_trees,
            "max_depth": max_depth,
            "status": "completed"
        }

    finally:

        spark.stop()


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config_path = Path(
        "configs/training_config.yaml"
    )

    config = load_config(
        config_path
    )

    # --------------------------------------------------------
    # MLflow experiment
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    logger.info(
        "Training pipeline started"
    )

    # ========================================================
    # MLP
    # ========================================================

    train_mlp(config)

    # ========================================================
    # LSTM
    # ========================================================

    train_lstm(config)

    # ========================================================
    # Random Forest + Ray
    # ========================================================

    logger.info(
        "Starting Ray Random Forest experiments"
    )

    ray.init()

    futures = []

    for num_trees in config["ray"]["num_trees"]:

        for max_depth in config["ray"]["max_depth"]:

            future = train_random_forest.remote(
                config,
                num_trees,
                max_depth
            )

            futures.append(future)

    logger.info(
        f"Submitted {len(futures)} "
        f"RF experiments to Ray"
    )

    # Wait for all Ray tasks
    results = ray.get(futures)

    logger.info(
        f"Ray results: {results}"
    )

    ray.shutdown()

    logger.info(
        "Training pipeline completed"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()



