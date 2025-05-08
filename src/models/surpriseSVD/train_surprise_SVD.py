import itertools
import os

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Caricamento variabili da .env
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("svd_recommender")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

DATA_PATH = "data/processed/surpriseSVD/reviews_clean.csv"
MODEL_OUTPUT_PATH = "models/svd_model.pkl"
PARAM_OUTPUT_PATH = "models/svd_best_params.txt"

def load_data(path):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(df["score"].min(), df["score"].max()))
    return Dataset.load_from_df(df[["userId", "gameId", "score"]], reader)

def main():
    data = load_data(DATA_PATH)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    param_grid = {
        "n_factors": [50, 100],
        "reg_all": [0.02, 0.1],
        "lr_all": [0.002, 0.005]
    }

    best_score = float("inf")
    best_params = None
    best_model = None

    all_combinations = list(itertools.product(
        param_grid["n_factors"],
        param_grid["reg_all"],
        param_grid["lr_all"]
    ))

    for n_factors, reg_all, lr_all in all_combinations:
        params = {
            "n_factors": n_factors,
            "reg_all": reg_all,
            "lr_all": lr_all
        }

        with mlflow.start_run():
            mlflow.log_params(params)

            model = SVD(**params)
            model.fit(trainset)
            predictions = model.test(testset)

            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)

            mlflow.log_metrics({
                "rmse": rmse,
                "mae": mae
            })

            print(f"🔍 Params: {params} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

            if rmse < best_score:
                best_score = rmse
                best_params = params
                best_model = model

    # Salvataggio del miglior modello
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    with open(PARAM_OUTPUT_PATH, "w") as f:
        f.write(str(best_params))

    print("\n🏆 Miglior combinazione trovata:")
    print(f"Params: {best_params}")
    print(f"RMSE: {best_score:.4f}")
    print(f"✅ Modello salvato in {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()

