import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient

# --- Load environment ---
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("svd_recommender")

MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"

# --- Load model from MLflow ---
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# --- Load training data artifact from MLflow ---
def load_training_data_from_registry(model_name: str, alias: str) -> pd.DataFrame:
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias)
    run_id = model_version.run_id
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="training_data/ratings.csv"
    )
    return pd.read_csv(path)

# Carica il dataset una sola volta all’avvio
ratings_df = load_training_data_from_registry(MODEL_NAME, MODEL_ALIAS)
print(f"✅ Dataset caricato: {ratings_df.shape[0]} righe")

# --- FastAPI app ---
app = FastAPI(title="Game Recommender API")

# CORS settings
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, n: int = 10):
    print(f"🔍 Richiesta per user_id: {user_id}")

    user_reviews = ratings_df[ratings_df["userId"] == user_id]

    if user_reviews.empty:
        raise HTTPException(status_code=404, detail="User ID non trovato o senza recensioni")

    reviewed_game_ids = user_reviews["itemId"].astype(str).unique()
    all_game_ids = ratings_df["itemId"].astype(str).unique()
    to_predict = [gid for gid in all_game_ids if gid not in reviewed_game_ids]

    print(f"📚 Giochi totali nel dataset: {len(all_game_ids)}")
    print(f"🎯 Giochi da raccomandare: {len(to_predict)}")

    predictions = [
        (gid, model.predict(user_id, gid).est)
        for gid in to_predict
    ]

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return [{"gameId": gid, "predicted_score": round(float(score), 2)} for gid, score in top_n]