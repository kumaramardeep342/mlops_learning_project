"""
src/logger.py

Centralized logging utility for the Predictive Maintenance MLOps project.
Usage in any module:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from unittest.mock import call

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "monitoring", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "application.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with both console and rotating-file output.

    Args:
        name: usually pass __name__ from the calling module.
        level: logging level (default INFO).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger



# Ek common logging function — har file jab kuch print/log karna chahe (progress, error, warning), 
# to isi function ko call karegi, taaki:
# Log terminal pe bhi dikhe aur ek file (application.log) 
# me bhi save ho jaye — isliye baad me bhi dekh sako kya hua tha
# Kyun chahiye
# print() sirf terminal pe dikhta hai, terminal band karte hi gayab
#test karne ka command 
# python -c "from src.utils.logger import get_logger; log = get_logger('test'); log.info('Logger working!'); log.warning('This is a warning')"