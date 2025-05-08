import ast
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

RAW_REVIEWS_PATH = "data/raw/reviews.csv"
RAW_GAMES_PATH = "data/raw/games.csv"
OUTPUT_DIR = "data/processed/lightfm"

# Caricamento Dataset Raw
def load_data():
    reviews = pd.read_csv(RAW_REVIEWS_PATH)
    games = pd.read_csv(RAW_GAMES_PATH)
    return reviews, games

# Encoding degli ID per LightFM
def encode_ids(reviews):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    reviews["user_idx"] = user_encoder.fit_transform(reviews["userId"])
    reviews["item_idx"] = item_encoder.fit_transform(reviews["gameId"])
    return reviews, user_encoder, item_encoder

# Creazione matrice di Interazione User-Item
def build_interaction_matrix(reviews):
    # Normalizza lo score tra 0 e 1
    norm_score = (reviews["score"] - reviews["score"].min()) / (reviews["score"].max() - reviews["score"].min())
    interactions = sparse.coo_matrix(
        (norm_score, (reviews["user_idx"], reviews["item_idx"]))
    )
    return interactions

# Funzione per gestire Parsing di Liste/Singole categorie
def safe_parse_list(val):
    if pd.isnull(val):
        return []

    # Caso 1: lista ben formattata
    if isinstance(val, str) and val.strip().startswith("[") and val.strip().endswith("]"):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass  # Se fallisce, tenta fallback

    # Caso 2: stringa singola (es. "Action RPG")
    try:
        return [v.strip() for v in val.split(",") if v.strip()]
    except Exception:
        return []


# Estrazuibe features Games
def extract_item_features(games, item_encoder):
    games = games.copy()
    games["item_idx"] = item_encoder.transform(games["_id"])

    # Parsing stringhe tipo "['Action', 'Adventure']"
    def parse_list_column(col):
        return col.apply(safe_parse_list)

    games["genre"] = parse_list_column(games["genre"])
    games["platforms"] = parse_list_column(games["platforms"])
    games["publishers"] = parse_list_column(games["publishers"])

    # One-Hot Encoding
    mlb = MultiLabelBinarizer()
    genre_enc = mlb.fit_transform(games["genre"])
    platforms_enc = mlb.fit_transform(games["platforms"])
    publishers_enc = mlb.fit_transform(games["publishers"])

    # Concatenazione One-Hot Encoded Features
    features = np.hstack([genre_enc, platforms_enc, publishers_enc])
    features_matrix = sparse.csr_matrix(features)

    # Reindicizza secondo l’ordine item_idx
    idx_sorted = games.sort_values("item_idx").index
    features_matrix = features_matrix[idx_sorted, :]

    return features_matrix

# Salvataggio dell`output
def save_outputs(interactions, item_features):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sparse.save_npz(os.path.join(OUTPUT_DIR, "interactions.npz"), interactions)
    sparse.save_npz(os.path.join(OUTPUT_DIR, "item_features.npz"), item_features)
    print("✅ Preprocessing completato e file salvati in:", OUTPUT_DIR)


if __name__ == "__main__":
    reviews, games = load_data()
    reviews, user_encoder, item_encoder = encode_ids(reviews)
    interactions = build_interaction_matrix(reviews)
    item_features = extract_item_features(games, item_encoder)
    save_outputs(interactions, item_features)