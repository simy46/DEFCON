from numpy.typing import NDArray
import numpy as np
from pandas import DataFrame

def predict_model(model, X_test: NDArray) -> NDArray:
    """
    Run predictions on the test set using the trained model.

    Parameters
    model : any
        Trained scikit-learn compatible model with predict().
    X_test : NDArray
        Test feature matrix.

    Returns
    NDArray[int]
        Vector of predictions (0/1).
    """
    preds: NDArray = model.predict(X_test)
    return preds.astype(int)
