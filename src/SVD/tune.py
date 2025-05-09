import itertools
import os

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate


def load_data(path):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(0, 5))
    return Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

def load_param_grid(path="config/grid_search.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["param_grid"]

def update_best_params(best_params, path="config/params.yaml"):
    with open(path, "w") as f:
        yaml.dump({"svd": best_params}, f)

def main():
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("svd_recommender")

    client = MlflowClient()   # noqa: F841
    data = load_data("data/processed/SVD/ratings.csv")
    param_grid = load_param_grid()

    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    print(f"🔍 Testing {len(combinations)} combinations...")

    best_score = float("inf")
    best_params = {}
    best_run_id = None

    for combo in combinations:
        params = dict(zip(keys, combo))
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            model = SVD(**params)
            results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=10, verbose=False)

            rmse = sum(results["test_rmse"]) / len(results["test_rmse"])
            mae = sum(results["test_mae"]) / len(results["test_mae"])

            mlflow.log_params(params)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)

            print(f"Params: {params} → RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            if rmse < best_score:
                best_score = rmse
                best_params = params
                best_run_id = run_id

    print(f"\n🏆 Best params: {best_params} → RMSE: {best_score:.4f}")
    update_best_params(best_params)

    # Etichetta la migliore run
    if best_run_id:
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("label", "Best")
            mlflow.set_tag("best_rmse", str(round(best_score, 4)))
        print(f"🏷️ Run '{best_run_id}' etichettata con label=Best")

if __name__ == "__main__":
    main()


