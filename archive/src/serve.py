"""
src/serve.py

FastAPI inference service.

Loads the automatically selected best model.
"""

import json
from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from src.training.model import MLP, LSTMModel


# ============================================================
# Configuration
# ============================================================

with open("model_metrics/best_model.json") as f:
    best = json.load(f)


MODEL_TYPE = best["model"]
MODEL_PATH = best["model_path"]


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="Predictive Maintenance API")


# ============================================================
# Input
# ============================================================


class MachineData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# ============================================================
# Load model
# ============================================================

if MODEL_TYPE == "MLP":
    model = MLP(input_size=5)
    model.load_state_dict(
        torch.load(MODEL_PATH + "/model.pt", map_location="cpu", weights_only=True)
    )

    model.eval()


elif MODEL_TYPE == "LSTM":
    model = LSTMModel(input_dim=5)
    model.load_state_dict(
        torch.load(MODEL_PATH + "/model.pt", map_location="cpu", weights_only=True)
    )

    model.eval()


elif MODEL_TYPE == "RandomForest":
    spark = (
        SparkSession.builder.appName("PredictiveMaintenanceAPI")
        .master("local[1]")
        .getOrCreate()
    )

    model = RandomForestClassificationModel.load(MODEL_PATH)
    assembler = VectorAssembler(
        inputCols=[
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ],
        outputCol="features",
    )


else:
    raise ValueError(f"Unknown model: {MODEL_TYPE}")


# ============================================================
# Health check
# ============================================================


@app.get("/")
def home():

    return {"status": "running", "model": MODEL_TYPE, "accuracy": best["accuracy"]}


# ============================================================
# Prediction
# ============================================================


@app.post("/predict")
def predict(data: MachineData):

    values = [
        [
            data.air_temperature,
            data.process_temperature,
            data.rotational_speed,
            data.torque,
            data.tool_wear,
        ]
    ]

    # --------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------

    if MODEL_TYPE in ["MLP", "LSTM"]:
        X = torch.tensor(values, dtype=torch.float32)

        if MODEL_TYPE == "LSTM":
            X = X.unsqueeze(1)

        with torch.no_grad():
            output = model(X)

            prediction = int(torch.argmax(output, dim=1).item())

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    else:
        df = spark.createDataFrame(
            values,
            [
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear",
            ],
        )

        df = assembler.transform(df)

        result = model.transform(df).select("prediction").first()

        prediction = int(result["prediction"])

    return {
        "prediction": prediction,
        "message": "Machine failure predicted"
        if prediction == 1
        else "No machine failure predicted",
        "model": MODEL_TYPE,
        "accuracy": best["accuracy"],
    }
