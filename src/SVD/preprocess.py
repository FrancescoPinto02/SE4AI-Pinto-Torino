import os

import numpy as np
import pandas as pd


def preprocess_reviews(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Rimozione colonne inutili
    df = df.drop(columns=["_id", "author", "text", "date", "_class"], errors="ignore")

    # Rinomina per compatibilità con Surprise
    df = df.rename(columns={
        "userId": "userId",
        "gameId": "itemId",
        "score": "rating"
    })

    # Rimuove duplicati e nulli
    df.drop_duplicates(subset=["userId", "itemId"], inplace=True)
    df.dropna(subset=["userId", "itemId", "rating"], inplace=True)

    # Winsorization (limita outlier a percentili 5 e 95)
    p5, p95 = np.percentile(df["rating"], [5, 95])
    df["rating"] = df["rating"].clip(lower=p5, upper=p95)

    # Normalizzazione in scala 0–5
    min_score = df["rating"].min()
    max_score = df["rating"].max()
    df["rating"] = (df["rating"] - min_score) / (max_score - min_score) * 5
    df["rating"] = df["rating"].clip(0, 5)

    # Crea cartella destinazione
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Dati preprocessati salvati in {output_path}")

if __name__ == "__main__":
    preprocess_reviews("data/raw/reviews.csv", "data/processed/SVD/ratings.csv")