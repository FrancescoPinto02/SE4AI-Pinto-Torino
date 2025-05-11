import itertools
import os
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from surprise import SVD, Dataset, Reader
from surprise.model_selection import KFold

# Constants for evaluation
DEFAULT_CV = 10
DEFAULT_K = 10
DEFAULT_THRESHOLD = 3.5

# Load the dataset
def load_data(path):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(0, 5))
    return Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

# Load the parameters from config file
def load_param_grid(path="config/grid_search.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("param_grid", {})

# Save the Best Parameters inside the config file
def update_best_params(best_params, path="config/params.yaml"):
    with open(path, "w") as f:
        yaml.dump({"svd": best_params}, f)

# Calculate the Precision@K and the Recall@K
def precision_recall_at_k(predictions, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD):
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


def evaluate_params(params, data, n_splits=DEFAULT_CV, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD):
    fold_precisions = []
    fold_recalls = []
    kf = KFold(n_splits=n_splits)
    for trainset, testset in kf.split(data):
        algo = SVD(**params)
        algo.fit(trainset)
        predictions = algo.test(testset)
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
        fold_precisions.append(np.mean(list(precisions.values())))
        fold_recalls.append(np.mean(list(recalls.values())))
    return np.mean(fold_precisions), np.mean(fold_recalls)


def main():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("svd_recommender")

    data = load_data("data/processed/SVD/ratings.csv")
    param_grid = load_param_grid()

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    print(f"🔍 Testing {len(combinations)} combinations...")

    best_score = 0
    best_params = {}
    best_run_id = None

    for combo in combinations:
        params = dict(zip(keys, combo))
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            avg_precision, avg_recall = evaluate_params(params, data)

            mlflow.log_params(params)
            mlflow.log_metric(f"Precision{DEFAULT_K}", avg_precision)
            mlflow.log_metric(f"Recall{DEFAULT_K}", avg_recall)

            print(f"Params: {params} → Precision@{DEFAULT_K}: {avg_precision:.4f}, Recall@{DEFAULT_K}: {avg_recall:.4f}")

            if avg_precision > best_score:
                best_score = avg_precision
                best_params = params
                best_run_id = run_id

    print(f"\n🏆 Best params: {best_params} → Precision@{DEFAULT_K}: {best_score:.4f}")
    update_best_params(best_params)

    # Tag the best run
    if best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("label", "Best")
            mlflow.set_tag("best_precision", str(round(best_score, 4)))
        print(f"🏷️ Run '{best_run_id}' etichettata con label=Best")


if __name__ == "__main__":
    main()


