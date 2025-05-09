import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

REGISTERED_MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"

def load_data(path):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(0, 5))
    return df, Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

def load_params(path="config/params.yaml"):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    return params["svd"]

def main():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("SVD-Recommender")

    # Autenticazione per DagsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    params = load_params()

    with mlflow.start_run() as run:   # noqa: F841
        model = SVD(**params)
        raw_df, data = load_data("data/processed/SVD/ratings.csv")

        # Cross-validation
        cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
        rmse = np.mean(cv_results["test_rmse"])
        mae = np.mean(cv_results["test_mae"])

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_params(params)

        # Fit finale
        trainset = data.build_full_trainset()
        model.fit(trainset)

        # Salvataggio locale
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/svd_model.pkl")

        # Logging modello
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="svd_model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        mlflow.log_artifact("data/processed/SVD/ratings.csv", artifact_path="training_data")

        print("✅ Modello loggato su MLflow")

        # Alias e tag con MlflowClient
        client = MlflowClient()
        latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME)[-1].version

        client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, latest_version)
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=latest_version,
            key="avg_rmse",
            value=str(round(rmse, 4))
        )

        print(f"🏷️ Alias '{MODEL_ALIAS}' assegnato alla versione {latest_version}")
        print("🎯 Modello pronto per il deployment")

if __name__ == "__main__":
    main()