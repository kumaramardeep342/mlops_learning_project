"""
src/preprocessing/dataset.py

Dataset loading utility for Predictive Maintenance.
"""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataset():

    """
    Load AI4I 2020 Predictive Maintenance dataset.
    """

    try:

        data_path = (
            "data/raw/ai4i2020.csv"
        )

        df = pd.read_csv(
            data_path
        )

        logger.info(
            "Dataset loaded successfully."
        )

        logger.info(
            f"Dataset shape: {df.shape}"
        )

        return df

    except Exception as e:

        logger.error(
            f"There is an issue with data loading: {e}"
        )

        return None