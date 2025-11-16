import pandas as pd
import os
from config.consts import OUTPUT_DIR

def save_submission(preds, timestamp: str) -> str:
    """
    Save predictions to a timestamped CSV file.

    Parameters
    preds : NDArray
        Predictions as integers (0/1).
    timestamp : str
        Timestamp string used for file naming.

    Returns
    str
        Path to the saved CSV file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"{OUTPUT_DIR}/submission_{timestamp}.csv"
    pd.Series(preds).to_csv(filename, index=False, header=False)

    return filename
