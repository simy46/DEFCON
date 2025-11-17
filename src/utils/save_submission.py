import pandas as pd
import os
from config.consts import IDS_KEY, OUTPUT_DIR

def save_submission(preds, metadata_test, timestamp: str) -> str:
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

    df = pd.DataFrame({
        "id": metadata_test[IDS_KEY].values,
        "label": preds.astype(int)
    })


    filename = f"{OUTPUT_DIR}/submission_{timestamp}.csv"
    df.to_csv(filename, index=False)

    return filename
