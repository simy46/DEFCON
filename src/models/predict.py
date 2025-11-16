import pandas as pd
from datetime import datetime
import os

def predict_model(model, X_test, metadata_test, timestamp):
    preds = model.predict(X_test).astype(int)

    df = pd.DataFrame({
        "label": preds
    })

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%d-%H-%M")
    filename = f"outputs/submission_{timestamp}.csv"
    df.to_csv(filename, index=False)

    return df
