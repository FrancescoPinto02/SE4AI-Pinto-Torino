import os
from typing import List

import mlflow
import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

# --- Load environment ---
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("svd_recommender")

# --- Config ---
MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"

# MongoDB credentials
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB")
MONGO_COLLECTION_REVIEWS = "reviews"

# --- FastAPI app ---
# --- FastAPI app ---
app = FastAPI(title="Game Recommender API")

# CORS settings
origins = [
    "http://localhost:5173",   # React development server
    "http://127.0.0.1:5173"    # Alternative localhost format
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permetti solo queste origini
    allow_credentials=True,
    allow_methods=["*"],    # Permetti tutti i metodi (GET, POST, PUT, DELETE, ecc.)
    allow_headers=["*"],    # Permetti tutti gli header
)

# --- Load model from MLflow ---
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


def get_mongo_reviews(user_id: str) -> pd.DataFrame:
    uri = (
        f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/"
        f"{MONGO_DB}?authSource={MONGO_AUTH_DB}&retryWrites=true&w=majority"
    )
    client = MongoClient(uri)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION_REVIEWS]

    try:
        query = {"userId": ObjectId(user_id)}
    except Exception:
        raise HTTPException(status_code=400, detail="Formato user_id non valido")

    results = list(collection.find(query))
    return pd.DataFrame(results)


def get_all_game_ids() -> List[str]:
    uri = (
        f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/"
        f"{MONGO_DB}?authSource={MONGO_AUTH_DB}&retryWrites=true&w=majority"
    )
    client = MongoClient(uri)
    db = client[MONGO_DB]
    games = db["games"].find({}, {"_id": 1})
    return [str(g["_id"]) for g in games]


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, n: int = 10):
    print(f"🔍 Richiesta per user_id: {user_id}")

    reviews_df = get_mongo_reviews(user_id)
    print(f"📄 Recensioni trovate: {len(reviews_df)}")

    if reviews_df.empty:
        raise HTTPException(status_code=404, detail="User ID non trovato o senza recensioni")

    # Converti gameId recensiti in stringhe per confronto coerente
    reviewed_game_ids = reviews_df["gameId"].apply(str).unique()
    all_game_ids = get_all_game_ids()
    print(f"📚 Giochi totali nel catalogo: {len(all_game_ids)}")

    to_predict = [gid for gid in all_game_ids if gid not in reviewed_game_ids]
    print(f"🎯 Giochi da raccomandare: {len(to_predict)}")

    predictions = [
        (gid, model.predict(user_id, gid).est)
        for gid in to_predict
    ]

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return [{"gameId": gid, "predicted_score": round(float(score), 2)} for gid, score in top_n]





