# import pandas as pd

# from src.utils.logger import get_logger


# logger = get_logger(__name__)


# def get_dataset():

#     try:
#         df = pd.read_csv("data/raw/ai4i2020.csv")

#         logger.info("Dataset loaded successfully")

#         return df

#     except Exception as e:
#         logger.error(f"Dataset loading failed: {e}")

#         return None


"""
src/preprocessing/dataset.py

Loads the Spark feature-engineered data (produced by preprocess.py ->
feature_engineering_job.py) for model training, instead of reading the
raw CSV directly — this connects the Airflow-orchestrated preprocessing
pipeline to the actual training step.

If the feature-engineered data hasn't been generated yet, this falls
back to the raw CSV with a warning, so training doesn't hard-crash.

Run the pipeline first to get the connected behavior:
    python src/preprocessing/preprocess.py
    python spark_jobs/feature_engineering_job.py
"""

import os
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAIN_FEATURES_PATH = "data/features/train_features.parquet"
TEST_FEATURES_PATH = "data/features/test_features.parquet"

# fallback only — used if the pipeline hasn't been run yet
RAW_PATH = "data/raw/ai4i2020.csv"


def get_dataset(split: str = "train") -> pd.DataFrame:
    """
    Loads the feature-engineered dataset produced by the preprocessing
    pipeline (preprocess.py -> feature_engineering_job.py).

    Args:
        split: "train" or "test" — which split to load.

    Returns:
        pandas DataFrame with cleaned, scaled features plus the derived
        columns (power, temp_diff, overstrain_indicator). Returns None
        if loading fails entirely.
    """
    path = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH

    if not os.path.exists(path):
        logger.warning(
            f"Feature-engineered data not found at {path}. "
            f"Falling back to raw CSV — run preprocess.py then "
            f"feature_engineering_job.py for the fully connected pipeline."
        )
        try:
            df = pd.read_csv(RAW_PATH)
            logger.info(f"Loaded raw dataset as fallback: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Fallback raw dataset loading also failed: {e}")
            return None

    try:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {split} feature-engineered dataset from {path}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return None
