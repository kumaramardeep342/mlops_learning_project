"""
src/preprocessing/preprocess.py

Loads raw AI4I 2020 data, drops identifier columns, handles missing values,
encodes categorical columns, scales numeric features, and splits into
train/test sets. Failure-mode columns (TWF, HDF, PWF, OSF, RNF) are kept
in the output for a future failure-mode attribution model, but are not
used as the training target here.

Run either as:
    python -m src.preprocessing.preprocess      (recommended)
or directly:
    python src/preprocessing/preprocess.py
"""

import os
import sys

# --- path fix: lets this file run directly (not just via `python -m`) ---
if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/ai4i2020.csv"
PROCESSED_DIR = "data/processed"

ID_COLS = ["UDI", "Product ID"]

NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COLS = ["Type"]
TARGET_COL = "Machine failure"

FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data file not found at '{path}'. "
            f"Run 'dvc pull' or place the CSV there manually."
        )
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.drop(columns=ID_COLS, errors="ignore")
    logger.info(f"Dropped ID columns: {ID_COLS}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_before = df.isnull().sum().sum()
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        logger.info(
            f"Encoded '{col}': classes {list(le.classes_)} -> {list(range(len(le.classes_)))}"
        )
    return df


def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    logger.info(f"Scaled numeric columns with StandardScaler: {NUMERIC_COLS}")
    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Saved train ({train_df.shape[0]} rows, "
        f"{int(train_df[TARGET_COL].sum())} failures) -> {train_path}"
    )
    logger.info(
        f"Saved test ({test_df.shape[0]} rows, "
        f"{int(test_df[TARGET_COL].sum())} failures) -> {test_path}"
    )

    return train_df, test_df


def run_preprocessing():
    df = load_data()
    df = handle_missing_values(df)
    df = encode_categorical(df)
    # df = scale_numeric(df)
    split_and_save(df)
    logger.info("Preprocessing pipeline completed successfully")


if __name__ == "__main__":
    run_preprocessing()
