import os
from collections import defaultdict

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from surprise import SVD, Dataset, Reader
from surprise.model_selection import KFold

REGISTERED_MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"


def load_data(path):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(0, 5))
    return df, Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)


def load_params(path="config/params.yaml"):
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    # Expected keys: n_splits, k, threshold, plus SVD hyperparameters
    return params.get("svd", {})


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}
    for uid, ratings in user_est_true.items():
        # Sort by estimated rating descending
        ratings.sort(key=lambda x: x[0], reverse=True)

        # Relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in ratings)
        # Recommended in top k
        n_rec_k = sum((est >= threshold) for (est, _) in ratings[:k])
        # Relevant and recommended
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def main():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("SVD-Recommender")

    # DagsHub auth
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    params = load_params()
    n_splits = params.get('n_splits', 5)
    k = params.get('k', 10)
    threshold = params.get('threshold', 3.5)
    # Remove non-SVD keys
    svd_kwargs = {key: params[key] for key in params if key not in ('n_splits', 'k', 'threshold')}

    with mlflow.start_run() as run: # noqa: F841
        # Cross-validation custom for Precision@k and Recall@k
        raw_df, data = load_data("data/processed/SVD/ratings.csv")
        kf = KFold(n_splits=n_splits)
        fold_precisions = []
        fold_recalls = []
        for trainset, testset in kf.split(data):
            algo = SVD(**svd_kwargs)
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
            fold_precisions.append(np.mean(list(precisions.values())))
            fold_recalls.append(np.mean(list(recalls.values())))

        avg_precision = np.mean(fold_precisions)
        avg_recall = np.mean(fold_recalls)

        # Log metrics and params
        mlflow.log_metric(f"Precision{k}", avg_precision)
        mlflow.log_metric(f"Recall{k}", avg_recall)
        mlflow.log_params(params)

        # Final training on full dataset
        trainset = data.build_full_trainset()
        final_model = SVD(**svd_kwargs)
        final_model.fit(trainset)

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_model, "models/svd_model.pkl")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="svd_model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        mlflow.log_artifact("data/processed/SVD/ratings.csv", artifact_path="training_data")
        print("✅ Modello loggato su MLflow con Precision@k e Recall@k")

        # Set alias and tags
        client = MlflowClient()
        latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME)[-1].version

        client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, latest_version)
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=latest_version,
            key="avg_precision",
            value=str(round(avg_precision, 4))
        )
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=latest_version,
            key="avg_recall",
            value=str(round(avg_recall, 4))
        )

        print(f"🏷️ Alias '{MODEL_ALIAS}' assegnato alla versione {latest_version}")
        print("🎯 Modello pronto per il deployment")


if __name__ == "__main__":
    main()
