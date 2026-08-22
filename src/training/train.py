"""
Training pipeline

MLP  -> PyTorch + torch.compile
LSTM -> PyTorch + torch.compile + sequences
RF   -> Spark ML + Ray parallel experiments
MLflow -> experiment tracking
"""

import json
import os
import shutil
import time
from pathlib import Path

import mlflow
import numpy as np
import ray
import torch
import torch.nn as nn
import yaml
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing.dataset import get_dataset
from src.training.model import MLP, LSTMModel, get_random_forest
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# CONFIG WITH DYNAMIC FALLBACKS
# ============================================================


def load_config():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    # Environment variables override YAML for Kubernetes / production parity
    config["data"]["data_path"] = os.getenv(
        "DATA_PATH", config["data"]["data_path"]
    )
    config["mlflow"]["tracking_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
    )

    # Handle output paths dynamically (checkpoints vs model_dir)
    default_model_dir = config["output"].get(
        "model_dir", config["output"].get("checkpoints_dir", "checkpoints")
    )
    config["output"]["model_dir"] = os.getenv("MODEL_DIR", default_model_dir)

    default_metrics_dir = config["output"].get(
        "metrics_dir", "checkpoints/model_metrics"
    )
    config["output"]["metrics_dir"] = os.getenv(
        "METRICS_DIR", default_metrics_dir
    )

    return config


# ============================================================
# SEQUENCE CREATION
# ============================================================


def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return (np.array(Xs), np.array(ys))


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
# TRAIN PYTORCH MODEL
# ============================================================


def train_torch_model(model, X_train, y_train, config, is_lstm=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = compile_model(model, config)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        inputs = X_train.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        logger.info(f"Epoch {epoch + 1}/{epochs} | Loss={loss.item():.4f}")

    return model, device


# ============================================================
# EVALUATE PYTORCH
# ============================================================


def evaluate_torch_model(model, X_test, y_test, device):
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
        accuracy = evaluate_torch_model(model, X_test, y_test, device)
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

    X_seq, y_seq = create_sequences(X, y, config["lstm"]["sequence_length"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    model = LSTMModel(input_size)

    with mlflow.start_run(run_name="LSTM"):
        model, device = train_torch_model(
            model, X_train, y_train, config, is_lstm=True
        )
        accuracy = evaluate_torch_model(model, X_test, y_test, device)
        training_time = time.time() - start

        mlflow.log_param("model", "LSTM")
        mlflow.log_param("sequence_length", config["lstm"]["sequence_length"])
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
# RANDOM FOREST - RAY
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
        train_df = train_df.withColumn(
            "label", col(label_column).cast("double")
        )
        test_df = test_df.withColumn("label", col(label_column).cast("double"))

        rf = get_random_forest(num_trees, max_depth)
        model = rf.fit(train_df)

        predictions = model.transform(test_df)
        rows = predictions.select("label", "prediction").collect()

        y_true = [r["label"] for r in rows]
        y_pred = [r["prediction"] for r in rows]

        accuracy = accuracy_score(y_true, y_pred)
        training_time = time.time() - start

        logger.info(
            f"RF completed | trees={num_trees} | depth={max_depth} | accuracy={accuracy:.4f}"
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
# SAVE BEST MODEL
# ============================================================


def save_best_model(
    best, results, config, mlp_result=None, lstm_result=None, rf_train_data=None
):
    model_dir = Path(config["output"]["model_dir"])
    metrics_dir = Path(config["output"]["metrics_dir"])

    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if best["model"] in ["MLP", "LSTM"]:
        model = (
            mlp_result["model_object"]
            if best["model"] == "MLP"
            else lstm_result["model_object"]
        )

        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        torch.save(model.state_dict(), model_dir / "model.pt")

    else:
        spark = (
            SparkSession.builder.appName("SaveBestModel")
            .master("local[1]")
            .getOrCreate()
        )

        try:
            train_df = spark.createDataFrame(rf_train_data)
            assembler = VectorAssembler(
                inputCols=config["data"]["feature_columns"], outputCol="features"
            )
            train_df = assembler.transform(train_df)

            label_column = config["data"]["label_column"]
            train_df = train_df.withColumn(
                "label", col(label_column).cast("double")
            )

            rf = get_random_forest(best["num_trees"], best["max_depth"])
            model = rf.fit(train_df)
            model.write().overwrite().save(str(model_dir))

        finally:
            spark.stop()

    best_info = {
        "model": best["model"],
        "accuracy": best["accuracy"],
        "model_path": str(model_dir),
    }

    if best["model"] == "RandomForest":
        best_info["num_trees"] = best["num_trees"]
        best_info["max_depth"] = best["max_depth"]

    with open(metrics_dir / "best_model.json", "w") as f:
        json.dump(best_info, f, indent=4)


# ============================================================
# MAIN
# ============================================================


def main():
    config = load_config()

    # Dynamic MLflow tracking connection
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Pass the resolved data_path explicitly to dataset loader
    df = get_dataset(data_path=config["data"]["data_path"])

    if df is None:
        logger.error(
            f"Failed to load dataset from path: {config['data']['data_path']}"
        )
        return

    features = config["data"]["feature_columns"]
    label = config["data"]["label_column"]

    df = df[features + [label]].dropna()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[label]
    )

    X_train = train_df[features].values
    y_train = train_df[label].values

    X_test = test_df[features].values
    y_test = test_df[label].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP Training
    mlp_result = train_mlp(
        X_train_scaled, y_train, X_test_scaled, y_test, len(features), config
    )

    # LSTM Training
    X_scaled = StandardScaler().fit_transform(df[features].values)
    y = df[label].values
    lstm_result = train_lstm(X_scaled, y, len(features), config)

    # Random Forest Training via Ray
    logger.info("Starting Ray Random Forest experiments")
    ray.init(ignore_reinit_error=True)

    futures = []
    for trees in config["ray"]["num_trees"]:
        for depth in config["ray"]["max_depth"]:
            future = train_random_forest.remote(
                train_df, test_df, config, trees, depth
            )
            futures.append(future)

    rf_results = ray.get(futures)
    ray.shutdown()

    # Results Consolidation
    results = [
        {
            "model": mlp_result["model"],
            "accuracy": mlp_result["accuracy"],
            "training_time": mlp_result["training_time"],
        },
        {
            "model": lstm_result["model"],
            "accuracy": lstm_result["accuracy"],
            "training_time": lstm_result["training_time"],
        },
    ]
    results.extend(rf_results)

    best = max(results, key=lambda x: x["accuracy"])
    logger.info(f"BEST MODEL = {best['model']}")
    logger.info(f"BEST ACCURACY = {best['accuracy']:.4f}")

    save_best_model(best, results, config, mlp_result, lstm_result, train_df)

    metrics_dir = Path(config["output"]["metrics_dir"])
    with open(metrics_dir / "model_results.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info("================================")
    logger.info("Training pipeline completed")
    logger.info("================================")


if __name__ == "__main__":
    main()