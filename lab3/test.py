import os
import pandas as pd
import mlflow
from sklearn.datasets import load_iris

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.labs.itmo.loc")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = os.getenv("MODEL_URI", "models:/IrisRandomForest/latest")
    model = mlflow.pyfunc.load_model(model_uri)

    iris = load_iris(as_frame=True)
    X = iris.data

    preds = model.predict(X)

    # Tests
    assert len(preds) == len(X), "Prediction length mismatch"
    unique = set(pd.Series(preds).unique())
    assert unique.issubset({0, 1, 2}), f"Unexpected classes: {unique}"
    assert pd.isna(pd.Series(preds)).sum() == 0, "Predictions contain NaN"

    print("OK: model loaded from registry and passed tests.")
    print("MODEL_URI:", model_uri)

if __name__ == "__main__":
    main()
