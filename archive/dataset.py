import pandas as pd
from src.utils.logger import get_logger


logger = get_logger(__name__)
def get_dataset():

    try:
        df = pd.read_csv("data/raw/ai4i2020.csv")
        logger.info("Dataset loaded successfully")
        return df

    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        return None
