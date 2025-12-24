import os
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.labs.itmo.loc")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("itmo-lab3-iris-v3")

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameters 
    n_estimators = int(os.getenv("N_ESTIMATORS", "100"))
    max_depth = int(os.getenv("MAX_DEPTH", "5"))

    with mlflow.start_run():
        # log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # train
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print("Logged run_id:", run_id)
        print("accuracy:", acc, "f1_macro:", f1)

        # register model in Model Registry
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name="IrisRandomForest")
        print("Registered model version:", result.version)

if __name__ == "__main__":
    main()
