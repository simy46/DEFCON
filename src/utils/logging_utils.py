import logging
import os
from datetime import datetime
from config.consts import LOG_FILE_PREFIX


def get_logger():
    logger = logging.getLogger(LOG_FILE_PREFIX)
    timestamp = datetime.now().strftime("%d-%H-%M-%S")

    if logger.hasHandlers():
        return logger, timestamp

    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)

    log_filename = f"logs/{LOG_FILE_PREFIX}_{timestamp}.log"

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, timestamp
