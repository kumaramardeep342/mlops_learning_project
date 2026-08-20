from src.utils.logger import get_logger
import pandas as pd
logger = get_logger(__name__)

def get_dataset():
    """
    Load the dataset from a CSV file.
    """
    try:
        # checking how data looks like - displaying 5 rows
        df = pd.read_csv("data/raw/ai4i2020.csv")
        logger.info("dataset loaded successfully !!.")
        return df
    except Exception as e:
        logger.error(f"There is an issue with Data Loading :{e}")
        return None