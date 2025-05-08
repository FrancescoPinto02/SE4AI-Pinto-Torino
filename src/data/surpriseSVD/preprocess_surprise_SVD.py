import os

import pandas as pd

RAW_PATH = "data/raw/reviews.csv"
OUTPUT_PATH = "data/processed/surpriseSVD/reviews_clean.csv"

def load_and_clean():
    df = pd.read_csv(RAW_PATH)

    # Mantieni tutti i punteggi, inclusi gli 0
    df = df[["userId", "gameId", "score"]]
    df = df.dropna(subset=["userId", "gameId", "score"])

    return df

def save(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ File salvato: {OUTPUT_PATH}")

if __name__ == "__main__":
    df_clean = load_and_clean()
    save(df_clean)

