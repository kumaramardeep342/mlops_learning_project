"""
Predictive Maintenance Training Pipeline

MLP       -> PyTorch + torch.compile
LSTM      -> PyTorch + torch.compile + sequences
RF        -> Spark ML
Ray       -> Parallel RF experiments
MLflow    -> Experiment tracking

All trained PyTorch models are saved in checkpoints/.
The best model is copied/saved under checkpoints/best_model/.
"""
import os
import json
import shutil
import time
from pathlib import Path

import mlflow
import numpy as np
import ray
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from src.training.model import MLP, LSTMModel, get_random_forest
from src.preprocessing.dataset import get_dataset
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================
# CONFIG
# ============================================================


def load_config():

    with open("configs/training_config.yaml", "r") as f:
        return yaml.safe_load(f)
    


# ============================================================
# SEQUENCES FOR LSTM
# ============================================================


def create_sequences(X, y, seq_len):

    Xs = []
    ys = []

    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])

        ys.append(y[i + seq_len])

    return np.array(Xs), np.array(ys)


# ============================================================
# TORCH COMPILE
# ============================================================


def compile_model(model, config):

    if not config["torch_compile"]["enabled"]:
        return model

    try:
        logger.info("Applying torch.compile()")

        model = torch.compile(model, backend="inductor")

        logger.info("torch.compile enabled")

    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")

    return model


# ============================================================
# PYTORCH TRAINING
# ============================================================


def train_torch_model(model, X_train, y_train, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model = compile_model(model, config)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()

        logger.info(f"Epoch {epoch + 1}/{epochs} | Loss={loss.item():.4f}")

    return model, device


# ============================================================
# PYTORCH EVALUATION
# ============================================================


def evaluate_model(model, X_test, y_test, device):

    model.eval()

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_test)

        predictions = torch.argmax(outputs, dim=1)

    predictions = predictions.cpu().numpy()

    return float(accuracy_score(y_test, predictions))


# ============================================================
# MLP
# ============================================================


def train_mlp(X_train, y_train, X_test, y_test, input_size, config):

    logger.info("Starting MLP training")

    start = time.time()

    model = MLP(input_size)

    with mlflow.start_run(run_name="MLP"):
        model, device = train_torch_model(model, X_train, y_train, config)

        accuracy = evaluate_model(model, X_test, y_test, device)

        training_time = time.time() - start

        mlflow.log_param("model", "MLP")

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("training_time", training_time)

    logger.info(f"MLP Accuracy = {accuracy:.4f}")

    return {
        "model": "MLP",
        "accuracy": accuracy,
        "training_time": training_time,
        "model_object": model,
    }


# ============================================================
# LSTM
# ============================================================


def train_lstm(X, y, input_size, config):

    logger.info("Starting LSTM training")

    start = time.time()

    seq_len = config["lstm"]["sequence_length"]

    X_seq, y_seq = create_sequences(X, y, seq_len)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    model = LSTMModel(input_size)

    with mlflow.start_run(run_name="LSTM"):
        model, device = train_torch_model(model, X_train, y_train, config)

        accuracy = evaluate_model(model, X_test, y_test, device)

        training_time = time.time() - start

        mlflow.log_param("model", "LSTM")

        mlflow.log_param("sequence_length", seq_len)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("training_time", training_time)

    logger.info(f"LSTM Accuracy = {accuracy:.4f}")

    return {
        "model": "LSTM",
        "accuracy": accuracy,
        "training_time": training_time,
        "model_object": model,
    }


# ============================================================
# RANDOM FOREST + RAY
# ============================================================


@ray.remote(num_cpus=1)
def train_random_forest(train_data, test_data, config, num_trees, max_depth):

    start = time.time()

    logger.info(f"RF started | trees={num_trees} | depth={max_depth}")

    spark = (
        SparkSession.builder.appName(f"RF_{num_trees}_{max_depth}")
        .master("local[1]")
        .getOrCreate()
    )

    try:
        train_df = spark.createDataFrame(train_data)

        test_df = spark.createDataFrame(test_data)

        assembler = VectorAssembler(
            inputCols=config["data"]["feature_columns"], outputCol="features"
        )

        train_df = assembler.transform(train_df)

        test_df = assembler.transform(test_df)

        label_column = config["data"]["label_column"]

        train_df = train_df.withColumn("label", col(label_column).cast("double"))

        test_df = test_df.withColumn("label", col(label_column).cast("double"))

        rf = get_random_forest(num_trees, max_depth)

        model = rf.fit(train_df)

        predictions = model.transform(test_df)

        rows = predictions.select("label", "prediction").collect()

        y_true = [row["label"] for row in rows]

        y_pred = [row["prediction"] for row in rows]

        accuracy = accuracy_score(y_true, y_pred)

        training_time = time.time() - start

        logger.info(
            f"RF completed | "
            f"trees={num_trees} | "
            f"depth={max_depth} | "
            f"accuracy={accuracy:.4f}"
        )

        return {
            "model": "RandomForest",
            "num_trees": num_trees,
            "max_depth": max_depth,
            "accuracy": float(accuracy),
            "training_time": training_time,
        }

    finally:
        spark.stop()


# ============================================================
# SAVE PYTORCH CHECKPOINT
# ============================================================


def save_torch_checkpoint(model, path):

    # torch.compile creates a wrapper.
    # Save the original PyTorch model.

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    torch.save(model.state_dict(), path)


# ============================================================
# SAVE BEST MODEL
# ============================================================


def save_best_model(best, mlp_result, lstm_result, rf_train_data, config):

    checkpoints_dir = Path(config["output"]["checkpoints_dir"])

    best_dir = checkpoints_dir / "best_model"

    # Remove previous best model

    if best_dir.exists():
        shutil.rmtree(best_dir)

    best_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # MLP / LSTM wins
    # --------------------------------------------------------

    if best["model"] == "MLP":
        source = checkpoints_dir / "mlp.pt"

        shutil.copy2(source, best_dir / "model.pt")

    elif best["model"] == "LSTM":
        source = checkpoints_dir / "lstm.pt"

        shutil.copy2(source, best_dir / "model.pt")

    # --------------------------------------------------------
    # Random Forest wins
    # --------------------------------------------------------

    else:
        spark = (
            SparkSession.builder.appName("SaveBestRF").master("local[1]").getOrCreate()
        )

        try:
            train_df = spark.createDataFrame(rf_train_data)

            assembler = VectorAssembler(
                inputCols=config["data"]["feature_columns"], outputCol="features"
            )

            train_df = assembler.transform(train_df)

            label_column = config["data"]["label_column"]

            train_df = train_df.withColumn("label", col(label_column).cast("double"))

            rf = get_random_forest(best["num_trees"], best["max_depth"])

            model = rf.fit(train_df)

            model.write().overwrite().save(str(best_dir))

        finally:
            spark.stop()

    # --------------------------------------------------------
    # Best model information
    # --------------------------------------------------------

    best_info = {
        "model": best["model"],
        "accuracy": best["accuracy"],
        "model_path": str(best_dir),
    }

    if best["model"] == "RandomForest":
        best_info["num_trees"] = best["num_trees"]

        best_info["max_depth"] = best["max_depth"]

    metrics_dir = Path(config["output"]["metrics_dir"])

    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "best_model.json", "w") as f:
        json.dump(best_info, f, indent=4)


# ============================================================
# MAIN
# ============================================================


def main():

    config = load_config()

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    logger.info("Training pipeline started")

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    checkpoints_dir = Path(config["output"]["checkpoints_dir"])

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    # df = get_dataset()
    
    # Pass data_path directly from training_config.yaml
    df = get_dataset(config["data"]["data_path"])

    if df is None:
        logger.error("Dataset loading failed")

        return

    features = config["data"]["feature_columns"]

    label = config["data"]["label_column"]

    df = df[features + [label]].dropna()

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[label]
    )

    X_train = train_df[features].values

    y_train = train_df[label].values

    X_test = test_df[features].values

    y_test = test_df[label].values

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # ========================================================
    # MLP
    # ========================================================

    mlp_result = train_mlp(
        X_train_scaled, y_train, X_test_scaled, y_test, len(features), config
    )

    # ALWAYS SAVE MLP
    save_torch_checkpoint(mlp_result["model_object"], checkpoints_dir / "mlp.pt")

    logger.info("MLP checkpoint saved")

    # ========================================================
    # LSTM
    # ========================================================

    # Use the same scaler for the complete sequence data.

    X_all_scaled = scaler.transform(df[features].values)

    y_all = df[label].values

    lstm_result = train_lstm(X_all_scaled, y_all, len(features), config)

    # ALWAYS SAVE LSTM
    save_torch_checkpoint(lstm_result["model_object"], checkpoints_dir / "lstm.pt")

    logger.info("LSTM checkpoint saved")

    # ========================================================
    # RANDOM FOREST + RAY
    # ========================================================

    logger.info("Starting Random Forest experiments with Ray")

    ray.init(ignore_reinit_error=True)

    futures = []

    for trees in config["ray"]["num_trees"]:
        for depth in config["ray"]["max_depth"]:
            future = train_random_forest.remote(train_df, test_df, config, trees, depth)

            futures.append(future)

    rf_results = ray.get(futures)

    ray.shutdown()

    logger.info("Ray shutdown complete")

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    results = [
        {
            "model": "MLP",
            "accuracy": mlp_result["accuracy"],
            "training_time": mlp_result["training_time"],
        },
        {
            "model": "LSTM",
            "accuracy": lstm_result["accuracy"],
            "training_time": lstm_result["training_time"],
        },
    ]

    results.extend(rf_results)

    # ========================================================
    # BEST MODEL
    # ========================================================

    best = max(results, key=lambda x: x["accuracy"])

    logger.info(f"BEST MODEL = {best['model']}")

    logger.info(f"BEST ACCURACY = {best['accuracy']:.4f}")

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    save_best_model(best, mlp_result, lstm_result, train_df, config)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_dir = Path(config["output"]["metrics_dir"])

    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("Model results saved")

    logger.info("================================")

    logger.info("Training pipeline completed")

    logger.info("================================")


if __name__ == "__main__":
    main()
