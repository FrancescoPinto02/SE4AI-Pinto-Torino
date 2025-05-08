import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

# Carica variabili di ambiente
load_dotenv()

# Config MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("svd_recommender")
REGISTERED_MODEL_NAME = "SVD-Recommender"
MODEL_ALIAS = "champion"

# Path file
DATA_PATH = "data/processed/surpriseSVD/reviews_clean.csv"
GAMES_PATH = "data/raw/games.csv"

def load_model_from_mlflow():
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    print(f"📦 Caricamento modello da MLflow: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)

def recommend(user_id, model, reviews_df, games_df, n=10):
    all_items = reviews_df["gameId"].unique()
    items_reviewed = reviews_df[reviews_df["userId"] == user_id]["gameId"].unique()
    items_to_predict = [iid for iid in all_items if iid not in items_reviewed]

    predictions = [
        (iid, model.predict(user_id, iid).est)
        for iid in items_to_predict
    ]

    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    result = pd.DataFrame(top_n, columns=["gameId", "predicted_score"])
    result = result.merge(games_df[["_id", "title"]], left_on="gameId", right_on="_id", how="left")
    return result[["title", "predicted_score"]]

def print_user_history(user_id, reviews_df, games_df):
    user_reviews = reviews_df[reviews_df["userId"] == user_id]
    merged = user_reviews.merge(games_df, left_on="gameId", right_on="_id", how="left")
    print(f"\n🎮 Giochi già valutati da {user_id}:")
    if merged.empty:
        print("  Nessuno.")
    else:
        for i, row in merged.iterrows():
            print(f"  - {row['title']} (score: {row['score']})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True, help="User ID per cui generare raccomandazioni")
    args = parser.parse_args()

    reviews_df = pd.read_csv(DATA_PATH)
    games_df = pd.read_csv(GAMES_PATH)

    if args.user_id not in reviews_df["userId"].unique():
        print(f"❌ User ID '{args.user_id}' non trovato nel dataset.")
        return

    model = load_model_from_mlflow()
    print_user_history(args.user_id, reviews_df, games_df)

    recommendations = recommend(args.user_id, model, reviews_df, games_df)
    print(f"\n✅ Raccomandazioni per l'utente {args.user_id}:")
    for i, row in recommendations.iterrows():
        print(f"{i + 1}. {row['title']} (score previsto: {row['predicted_score']:.2f})")

if __name__ == "__main__":
    main()

