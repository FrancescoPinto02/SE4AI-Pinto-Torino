import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Setup
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("svd_recommender")

app = FastAPI(title="Game Recommender API")

# Config
MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"
DATA_PATH = "data/processed/surpriseSVD/reviews_clean.csv"
GAMES_PATH = "data/raw/games.csv"

# Cache model & data
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
reviews_df = pd.read_csv(DATA_PATH)
games_df = pd.read_csv(GAMES_PATH)

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, n: int = 10):
    if user_id not in reviews_df["userId"].unique():
        raise HTTPException(status_code=404, detail="User ID non trovato")

    all_items = reviews_df["gameId"].unique()
    reviewed = reviews_df[reviews_df["userId"] == user_id]["gameId"].unique()
    to_predict = [iid for iid in all_items if iid not in reviewed]

    predictions = [
        (iid, model.predict(user_id, iid).est)
        for iid in to_predict
    ]

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    result = pd.DataFrame(top_n, columns=["gameId", "score"])
    result = result.merge(games_df[["_id", "title"]], left_on="gameId", right_on="_id", how="left")

    return [
        {"title": row["title"], "predicted_score": round(row["score"], 2)}
        for _, row in result.iterrows()
    ]