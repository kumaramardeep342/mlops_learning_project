import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_dataset(data_path="data/raw/ai4i2020.csv"):
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Dataset loading failed from {data_path}: {e}")
        return None