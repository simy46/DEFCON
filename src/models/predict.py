import pandas as pd

def predict_model(model, X_test, metadata_test):
    preds = model.predict(X_test)

    df = pd.DataFrame({
        "id": metadata_test["ID"],
        "label": preds
    })

    df.to_csv("outputs/submission.csv", index=False)
    return df
