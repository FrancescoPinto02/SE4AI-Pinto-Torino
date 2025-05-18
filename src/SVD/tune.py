import itertools
import os
import sys
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from surprise import SVD, Dataset, Reader
from surprise.model_selection import KFold

# Path hack per importare log_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.log_config import setup_logger

logger = setup_logger("tune")
load_dotenv()


def load_config(path="config/SVD/tune.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_best_params(best_params, path="config/SVD/params.yaml"):
    with open(path, "w") as f:
        yaml.dump({"svd": best_params}, f)
    logger.info(f"Parametri migliori salvati in {path}")


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in ratings)
        n_rec_k = sum(est >= threshold for est, _ in ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold and est >= threshold)
                              for est, true_r in ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0
    return precisions, recalls


def evaluate_params(params, data, cv=5, k=10, threshold=3.5):
    kf = KFold(n_splits=cv)
    fold_precisions = []
    fold_recalls = []
    for trainset, testset in kf.split(data):
        algo = SVD(**params)
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
        fold_precisions.append(np.mean(list(precisions.values())))
        fold_recalls.append(np.mean(list(recalls.values())))
    return np.mean(fold_precisions), np.mean(fold_recalls)


def main():
    logger.info("Avvio tuning iperparametri SVD con MLflow")
    config = load_config()

    train_path = config["train_path"]
    cv = config.get("cv", 5)
    k = config.get("k", 10)
    threshold = config.get("threshold", 3.5)
    param_grid = config.get("param_grid", {})

    df = pd.read_csv(train_path)
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

    # Imposta tracking URI ed esperimento
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("svd_recommender")

    combinations = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())

    logger.info(f"Totale combinazioni da testare: {len(combinations)}")

    best_score = 0
    best_params = {}
    best_run_id = None

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        logger.info(f"[{i}/{len(combinations)}] Test parametri: {params}")

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            avg_precision, avg_recall = evaluate_params(params, data, cv=cv, k=k, threshold=threshold)

            mlflow.log_params(params)
            mlflow.log_metric(f"Precision{k}", avg_precision)
            mlflow.log_metric(f"Recall{k}", avg_recall)

            logger.info(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}")

            if avg_precision > best_score:
                best_score = avg_precision
                best_params = params
                best_run_id = run_id

    save_best_params(best_params)
    logger.info(f"Parametri migliori: {best_params} → Precision@{k}: {best_score:.4f}")

    if best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("label", "Best")
            mlflow.set_tag("best_precision", str(round(best_score, 4)))
            logger.info(f"Run '{best_run_id}' etichettata come Best")

    logger.info("Tuning completato con successo")


if __name__ == "__main__":
    main()


