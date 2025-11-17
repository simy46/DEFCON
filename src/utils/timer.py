import time
import logging
from config.consts import LOG_FILE_PREFIX

logger = logging.getLogger(LOG_FILE_PREFIX)

class Timer:
    def __init__(self, msg=""):
        self.msg = msg
    
    def __enter__(self):
        logger.info(f"\n[TIMER START] {self.msg}")
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_value, traceback):
        end = time.time()
        logger.info(f"[TIMER END] {self.msg} ({end - self.start:.2f} sec)\n")
        return False