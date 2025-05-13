import socket

import mlflow
import pandas as pd
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from py_eureka_client.eureka_client import EurekaClient
import asyncio

from src.log_config import setup_logger

# Configurazione e Inizializzazione
load_dotenv()

TRACKING_URI = "https://dagshub.com/FrancescoPinto02/SE4AI-Pinto-Torino.mlflow"
EXPERIMENT_NAME = "svd_recommender"
MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"

if not TRACKING_URI:
    raise RuntimeError("MLFLOW_TRACKING_URI non definito nel file .env")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

logger = setup_logger()
logger.info("Avvio applicazione Game Recommender API")

# Caricamento Modello
logger.info(f"Caricamento modello '{MODEL_NAME}' con alias '{MODEL_ALIAS}' da MLflow")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# Caricamento Dataset

def load_training_data_from_registry(model_name: str, alias: str) -> pd.DataFrame:
    logger.info("Download dataset di training da MLflow registry")
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias)
    path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id,
        artifact_path="training_data/ratings.csv"
    )
    df = pd.read_csv(path)
    logger.info(f"Dataset caricato: {df.shape[0]} righe, {df['itemId'].nunique()} giochi")
    return df

ratings_df = load_training_data_from_registry(MODEL_NAME, MODEL_ALIAS)

# Calcolo Items per Fallback
logger.info("Calcolo fallback ponderato basato su popolarità e qualità")
C = ratings_df["rating"].mean()
m = 20

game_stats = (
    ratings_df.groupby("itemId")
    .agg(v=("rating", "count"), R=("rating", "mean"))
    .reset_index()
)

game_stats["WR"] = game_stats.apply(
    lambda x: (x["v"] / (x["v"] + m)) * x["R"] + (m / (x["v"] + m)) * C,
    axis=1
)

popular_fallback_games = [
    {"gameId": str(row["itemId"]), "predicted_score": round(row["WR"], 2), "fallback": True}
    for _, row in game_stats.sort_values(by="WR", ascending=False).iterrows()
]

# FastAPI Setup
app = FastAPI(title="Game Recommender API")

ip = socket.gethostbyname(socket.gethostname())
logger.info(f"IP del server: {ip}")

# Registrazione su Eureka
async def start_eureka_client():
    try:
        eureka_client = EurekaClient(
            eureka_server="http://discovery-server:8761/eureka",
            app_name="recommendation-service",
            instance_port=8000,
            instance_ip=ip,
            instance_id="recommendation-service",
            renewal_interval_in_secs=5,
            duration_in_secs=10,
            metadata={"version": "1.0.0"},
        )
        await eureka_client.start()
        logger.info("Servizio di raccomandazione registrato su Eureka con successo.")
        while True:
            await asyncio.sleep(30)
    except Exception as e:
        logger.error(f"Errore durante la registrazione su Eureka: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("Avvio registrazione su Eureka")
    asyncio.create_task(start_eureka_client())

@app.get("/health")
async def health_check():
    return {"status": "UP"}

# Prometheus Metrics
instrumentator = Instrumentator().instrument(app).expose(app)
PREDICTED_SCORE_AVG = Gauge("predicted_score_avg", "Media dei punteggi previsti per una richiesta")
RECOMMENDATIONS_SENT = Counter("recommendations_total", "Totale Items raccomandati")
RECOMMENDATION_CLICKS = Counter("recommendation_clicks_total", "Click su raccomandazioni mostrate")
FALLBACK_USERS = Counter("fallback_users_total", "Numero utenti serviti con fallback")
PREDICTION_REQUEST = Counter("prediction_request", "Numero richieste per raccomandazioni")

# Cache Raccomandazioni
recommendation_cache = TTLCache(maxsize=1000, ttl=600)

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, n: int = 10):
    logger.info(f"Richiesta raccomandazioni | user_id={user_id}, top={n}")
    PREDICTION_REQUEST.inc(1)

    cache_key = f"{user_id}_{n}"
    if cache_key in recommendation_cache:
        cached_response = recommendation_cache[cache_key]
        logger.info(f"Cache HIT per user_id={user_id} | Raccomandazioni: {cached_response}")
        RECOMMENDATIONS_SENT.inc(n)
        return cached_response

    logger.info(f"Cache MISS per user_id={user_id}")
    user_reviews = ratings_df[ratings_df["userId"] == user_id]

    if user_reviews.empty:
        logger.warning(f"Cold start per user_id={user_id} - fallback attivato")
        FALLBACK_USERS.inc(1)
        fallback_top_n = popular_fallback_games[:n]
        logger.info(f"Fallback restituito per user_id={user_id} | Raccomandazioni: {fallback_top_n}")
        RECOMMENDATIONS_SENT.inc(n)
        return fallback_top_n

    reviewed_game_ids = user_reviews["itemId"].astype(str).unique()
    all_game_ids = ratings_df["itemId"].astype(str).unique()
    to_predict = [gid for gid in all_game_ids if gid not in reviewed_game_ids]

    predictions = [
        (gid, model.predict(user_id, gid).est)
        for gid in to_predict
    ]

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    if top_n:
        avg_score = sum(score for _, score in top_n) / len(top_n)
        PREDICTED_SCORE_AVG.set(avg_score)

    response = [
        {"gameId": gid, "predicted_score": round(float(score), 2), "fallback": False}
        for gid, score in top_n
    ]

    recommendation_cache[cache_key] = response
    logger.info(f"Raccomandazioni inviate per user_id={user_id}: {response}")

    logger.info(f"Raccomandazioni salvate in cache per user_id={user_id}")
    RECOMMENDATIONS_SENT.inc(n)
    return response

@app.post("/feedback")
def receive_feedback(user_id: str = Query(...), item_id: str = Query(...)):
    logger.info(f"Feedback ricevuto | user_id={user_id}, item_id={item_id}")
    RECOMMENDATION_CLICKS.inc()
    return {
        "status": "ok",
        "message": "Feedback ricevuto correttamente",
        "data": {"user_id": user_id, "item_id": item_id}
    }