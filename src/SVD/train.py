import os
import sys
from collections import defaultdict

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from surprise import SVD, Dataset, Reader

# Setup import per il logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.log_config import setup_logger

logger = setup_logger("train")
load_dotenv()


def load_config(path="config/SVD/train.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_best_params(path="config/SVD/params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f).get("svd", {})


def precision_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rec_k = sum(est >= threshold for est, _ in ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold and est >= threshold)
                              for est, true_r in ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0

    return np.mean(list(precisions.values()))


def prepare_dataset(df):
    reader = Reader(rating_scale=(0, 5))
    return Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)


def main():
    logger.info("Avvio training del modello SVD")
    config = load_config()
    best_params = load_best_params()

    train_path = config["train_path"]
    test_path = config["test_path"]
    k = config.get("k", 10)
    threshold = config.get("threshold", 3.5)
    model_name = config["model_name"]
    model_alias = config["model_alias"]
    experiment_name = config["experiment_name"]

    logger.info(f"Caricamento train set da: {train_path}")
    df_train = pd.read_csv(train_path)
    logger.info(f"Caricamento test set da: {test_path}")
    df_test = pd.read_csv(test_path)

    train_data = prepare_dataset(df_train)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    # Allenamento sul training set
    logger.info(f"Addestramento su train set con parametri: {best_params}")
    algo = SVD(**best_params)
    trainset = train_data.build_full_trainset()
    algo.fit(trainset)

    # Valutazione su test set
    logger.info("Valutazione su test set (Leave-One-Out)")
    testset = list(df_test[["userId", "itemId", "rating"]].itertuples(index=False, name=None))
    predictions = algo.test(testset)
    precision = precision_at_k(predictions, k=k, threshold=threshold)

    logger.info(f"Precision@{k} = {precision:.4f}")

    # Riaddestramento sul dataset completo
    logger.info("Riaddestramento su train + test")
    full_df = pd.concat([df_train, df_test])
    full_data = prepare_dataset(full_df)
    final_trainset = full_data.build_full_trainset()
    final_model = SVD(**best_params)
    final_model.fit(final_trainset)

    # Salvataggio modello
    os.makedirs("models", exist_ok=True)
    model_path = "models/svd_model.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"Modello salvato in {model_path}")

    # Log su MLflow
    with mlflow.start_run() as run:  # noqa: F841
        mlflow.log_params(best_params)
        mlflow.log_metric(f"Precision{k}", precision)

        mlflow.log_artifact("config/SVD/train.yaml")
        mlflow.log_artifact("config/SVD/params.yaml")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(test_path, artifact_path="test_data")
        mlflow.log_artifact(train_path, artifact_path="train_data")

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="svd_model",
            registered_model_name=model_name
        )

        full_path = "data/processed/SVD/full_train.csv"
        full_df.to_csv(full_path, index=False)
        mlflow.log_artifact(full_path, artifact_path="full_data")

        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name)[-1].version
        client.set_registered_model_alias(model_name, model_alias, latest_version)
        client.set_model_version_tag(model_name, latest_version, "avg_precision", str(round(precision, 4)))

        logger.info(f"Alias '{model_alias}' assegnato alla versione {latest_version}")
        logger.info("Modello loggato su MLflow con successo")


if __name__ == "__main__":
    main()