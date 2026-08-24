"""
src/serve.py
FastAPI inference service.
Loads the automatically selected best model.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.training.model import MLP, LSTMModel

# ============================================================
# Configuration & Globals
# ============================================================

metrics_dir = os.getenv("METRICS_DIR", "/app/checkpoints/model_metrics")
metrics_file = Path(metrics_dir) / "best_model.json"

if not metrics_file.exists():
    metrics_file = Path("checkpoints/model_metrics/best_model.json")

with open(metrics_file) as f:
    best = json.load(f)

MODEL_TYPE = best["model"]
MODEL_PATH = os.getenv("MODEL_DIR", best["model_path"])

model = None
spark = None
assembler = None

# ============================================================
# Lifespan Event (Model Loading)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, spark, assembler

    if MODEL_TYPE == "MLP":
        model = MLP(input_size=5)
        model.load_state_dict(
            torch.load(Path(MODEL_PATH) / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()

    elif MODEL_TYPE == "LSTM":
        model = LSTMModel(input_size=5)
        model.load_state_dict(
            torch.load(Path(MODEL_PATH) / "model.pt", map_location="cpu", weights_only=True)
        )
        model.eval()

    elif MODEL_TYPE == "RandomForest":
        from pyspark.ml.classification import RandomForestClassificationModel
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("PredictiveMaintenanceAPI")
            .config("spark.driver.memory", "512m")
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

    yield

    if spark is not None:
        spark.stop()

# ============================================================
# FastAPI Setup
# ============================================================

app = FastAPI(title="Predictive Maintenance API", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)

class MachineData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/")
def home():
    return {"status": "running", "model": MODEL_TYPE, "accuracy": best["accuracy"]}

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

    if MODEL_TYPE in ["MLP", "LSTM"]:
        X = torch.tensor(values, dtype=torch.float32)
        if MODEL_TYPE == "LSTM":
            X = X.unsqueeze(1)
        with torch.no_grad():
            output = model(X)
            prediction = int(torch.argmax(output, dim=1).item())

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

# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("src.inference.serve:app", host="0.0.0.0", port=port, reload=False)