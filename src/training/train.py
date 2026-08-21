"""
src/training/train.py

Pipeline:

1. Load AI4I dataset
2. Train/validation split
3. Standardize features for PyTorch models
4. Train MLP using torch.compile
5. Train LSTM using torch.compile
6. Train multiple Random Forest configurations using Ray
7. Compare F1 scores
8. Select best model
9. Save only best model
10. Save model metrics
"""

import json
import shutil
import time
from pathlib import Path

import mlflow
import ray
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

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

def load_config():

    with open(
        "configs/training_config.yaml",
        "r"
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# Prepare PyTorch data
# ============================================================

def create_loader(
    X,
    y,
    config
):

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    y = torch.tensor(
        y,
        dtype=torch.long
    )

    dataset = TensorDataset(
        X,
        y
    )

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )


# ============================================================
# Train PyTorch model
# ============================================================

def train_torch_model(
    model,
    train_loader,
    config,
    is_lstm=False
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    # --------------------------------------------------------
    # torch.compile
    # --------------------------------------------------------

    if config["torch_compile"]["enabled"]:

        try:

            logger.info(
                "Applying torch.compile()"
            )

            model = torch.compile(
                model,
                backend="inductor"
            )

            logger.info(
                "torch.compile enabled"
            )

        except Exception as e:

            logger.warning(
                f"torch.compile failed: {e}"
            )

            logger.warning(
                "Continuing with normal PyTorch"
            )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    targets = (
        train_loader.dataset
        .tensors[1]
    )

    class_counts = torch.bincount(
        targets,
        minlength=2
    ).float()

    class_weights = (
        len(targets)
        /
        (2 * class_counts)
    )

    class_weights = class_weights.to(
        device
    )

    logger.info(
        f"Class counts: "
        f"{class_counts.tolist()}"
    )

    logger.info(
        f"Class weights: "
        f"{class_weights.tolist()}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):

        total_loss = 0.0

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            # LSTM expects:
            # [batch, sequence, features]

            if is_lstm:

                inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

        logger.info(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss={total_loss:.4f}"
        )

    return model, device


# ============================================================
# Evaluate PyTorch model
# ============================================================

def evaluate_torch_model(
    model,
    X_val,
    y_val,
    device,
    is_lstm=False
):

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    ).to(device)

    if is_lstm:

        X_val = X_val.unsqueeze(1)

    model.eval()

    with torch.no_grad():

        outputs = model(
            X_val
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    predictions = (
        predictions
        .cpu()
        .numpy()
    )

    score = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    return float(score)


# ============================================================
# Train MLP / LSTM
# ============================================================

def run_torch_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    is_lstm=False
):

    logger.info(
        f"Starting {name} training"
    )

    start_time = time.time()

    train_loader = create_loader(
        X_train,
        y_train,
        config
    )

    model, device = train_torch_model(
        model,
        train_loader,
        config,
        is_lstm
    )

    f1 = evaluate_torch_model(
        model,
        X_val,
        y_val,
        device,
        is_lstm
    )

    training_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.log_param(
        "model",
        name
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "training_time",
        training_time
    )

    logger.info(
        f"{name} completed | "
        f"F1={f1:.4f}"
    )

    return {
        "model": name,
        "f1_score": f1,
        "training_time": training_time,
        "model_object": model
    }


# ============================================================
# Random Forest - Ray
# ============================================================

@ray.remote(num_cpus=1)
def run_random_forest(
    train_df,
    val_df,
    config,
    num_trees,
    max_depth
):

    logger.info(
        f"RF started | "
        f"trees={num_trees} | "
        f"depth={max_depth}"
    )

    start_time = time.time()

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
        # Pandas -> Spark
        # ----------------------------------------------------

        train = spark.createDataFrame(
            train_df
        )

        val = spark.createDataFrame(
            val_df
        )

        # ----------------------------------------------------
        # Feature vector
        # ----------------------------------------------------

        assembler = VectorAssembler(
            inputCols=config["data"]["feature_columns"],
            outputCol="features"
        )

        train = assembler.transform(
            train
        )

        val = assembler.transform(
            val
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label_column = (
            config["data"]["label_column"]
        )

        train = train.withColumn(
            "label",
            col(label_column).cast("double")
        )

        val = val.withColumn(
            "label",
            col(label_column).cast("double")
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        rf = get_random_forest(
            num_trees=num_trees,
            max_depth=max_depth
        )

        model = rf.fit(
            train
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        predictions = model.transform(
            val
        )

        rows = predictions.select(
            "label",
            "prediction"
        ).collect()

        y_true = [
            row["label"]
            for row in rows
        ]

        y_pred = [
            row["prediction"]
            for row in rows
        ]

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        training_time = (
            time.time() - start_time
        )

        logger.info(
            f"RF completed | "
            f"trees={num_trees} | "
            f"depth={max_depth} | "
            f"F1={f1:.4f}"
        )

        return {
            "model": "RandomForest",
            "num_trees": num_trees,
            "max_depth": max_depth,
            "f1_score": float(f1),
            "training_time": training_time
        }

    finally:

        spark.stop()


# ============================================================
# Save best model
# ============================================================

def save_best_model(
    best,
    torch_model,
    train_df,
    config
):

    model_dir = Path(
        config["output"]["model_dir"]
    )

    metrics_dir = Path(
        config["output"]["metrics_dir"]
    )

    # --------------------------------------------------------
    # Remove previous best model
    # --------------------------------------------------------

    if model_dir.exists():

        shutil.rmtree(
            model_dir
        )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # PyTorch
    # ========================================================

    if best["model"] in [
        "MLP",
        "LSTM"
    ]:

        model_to_save = torch_model

        # torch.compile wrapper
        if hasattr(
            model_to_save,
            "_orig_mod"
        ):

            model_to_save = (
                model_to_save._orig_mod
            )

        torch.save(
            model_to_save.state_dict(),
            model_dir / "model.pt"
        )

        logger.info(
            f"{best['model']} saved"
        )

    # ========================================================
    # Random Forest
    # ========================================================

    elif best["model"] == "RandomForest":

        spark = (
            SparkSession.builder
            .appName(
                "SaveBestRF"
            )
            .master("local[1]")
            .getOrCreate()
        )

        try:

            train = spark.createDataFrame(
                train_df
            )

            assembler = VectorAssembler(
                inputCols=config["data"]["feature_columns"],
                outputCol="features"
            )

            train = assembler.transform(
                train
            )

            label_column = (
                config["data"]["label_column"]
            )

            train = train.withColumn(
                "label",
                col(label_column).cast("double")
            )

            rf = get_random_forest(
                num_trees=best["num_trees"],
                max_depth=best["max_depth"]
            )

            model = rf.fit(
                train
            )

            model.write().overwrite().save(
                str(model_dir)
            )

            logger.info(
                "Best Random Forest saved"
            )

        finally:

            spark.stop()

    # ========================================================
    # Best model metadata
    # ========================================================

    metadata = {
        "model": best["model"],
        "f1_score": best["f1_score"],
        "model_path": str(model_dir)
    }

    if best["model"] == "RandomForest":

        metadata["num_trees"] = (
            best["num_trees"]
        )

        metadata["max_depth"] = (
            best["max_depth"]
        )

    with open(
        metrics_dir / "best_model.json",
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


# ============================================================
# Main
# ============================================================

def main():

    config = load_config()

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    logger.info(
        "================================"
    )

    logger.info(
        "Training pipeline started"
    )

    logger.info(
        "================================"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = get_dataset()

    if df is None:

        logger.error(
            "Dataset loading failed"
        )

        return

    feature_columns = (
        config["data"]["feature_columns"]
    )

    label_column = (
        config["data"]["label_column"]
    )

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df[
        feature_columns +
        [label_column]
    ].dropna()

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[label_column]
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train_df[
        feature_columns
    ].values

    y_train = train_df[
        label_column
    ].values

    X_val = val_df[
        feature_columns
    ].values

    y_val = val_df[
        label_column
    ].values

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    logger.info(
        "Feature scaling completed"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # ========================================================
    # MLP
    # ========================================================

    with mlflow.start_run(
        run_name="MLP"
    ):

        mlp_result = run_torch_model(
            name="MLP",
            model=MLP(
                input_size=len(
                    feature_columns
                )
            ),
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            config=config
        )

    results.append(
        mlp_result
    )

    # ========================================================
    # LSTM
    # ========================================================

    with mlflow.start_run(
        run_name="LSTM"
    ):

        lstm_result = run_torch_model(
            name="LSTM",
            model=LSTMModel(
                input_dim=len(
                    feature_columns
                )
            ),
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            config=config,
            is_lstm=True
        )

    results.append(
        lstm_result
    )

    # ========================================================
    # Random Forest + Ray
    # ========================================================

    logger.info(
        "Starting Random Forest experiments with Ray"
    )

    ray.init(
        ignore_reinit_error=True
    )

    futures = []

    for num_trees in (
        config["ray"]["num_trees"]
    ):

        for max_depth in (
            config["ray"]["max_depth"]
        ):

            futures.append(
                run_random_forest.remote(
                    train_df,
                    val_df,
                    config,
                    num_trees,
                    max_depth
                )
            )

    logger.info(
        f"Submitted {len(futures)} "
        f"RF experiments"
    )

    rf_results = ray.get(
        futures
    )

    ray.shutdown()

    logger.info(
        "Ray shutdown complete"
    )

    results.extend(
        rf_results
    )

    # ========================================================
    # Remove model objects before JSON
    # ========================================================

    clean_results = []

    for result in results:

        clean_results.append({
            key: value
            for key, value in result.items()
            if key != "model_object"
        })

    # ========================================================
    # Select best model
    # ========================================================

    best = max(
        clean_results,
        key=lambda x: x["f1_score"]
    )

    logger.info(
        f"BEST MODEL = {best['model']}"
    )

    logger.info(
        f"BEST F1 = {best['f1_score']:.4f}"
    )

    # ========================================================
    # Find PyTorch model
    # ========================================================

    torch_model = None

    if best["model"] in [
        "MLP",
        "LSTM"
    ]:

        for result in results:

            if result["model"] == best["model"]:

                torch_model = (
                    result["model_object"]
                )

                break

    # ========================================================
    # Save best model
    # ========================================================

    save_best_model(
        best=best,
        torch_model=torch_model,
        train_df=train_df,
        config=config
    )

    # ========================================================
    # Save all results
    # ========================================================

    metrics_dir = Path(
        config["output"]["metrics_dir"]
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        metrics_dir / "model_results.json",
        "w"
    ) as file:

        json.dump(
            clean_results,
            file,
            indent=4
        )

    logger.info(
        "model_results.json saved"
    )

    logger.info(
        "================================"
    )

    logger.info(
        "Training pipeline completed"
    )

    logger.info(
        "================================"
    )


if __name__ == "__main__":

    main()