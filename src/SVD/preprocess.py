import os
import sys

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from surprise import Dataset, Reader
from surprise.model_selection import LeaveOneOut

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.log_config import setup_logger

logger = setup_logger("preprocess")
load_dotenv()


def load_config(path="config/SVD/preprocess.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(df, rating_col="score", min_rating=0, max_rating=5, winsorize_percentiles=(5, 95)):
    logger.info("Inizio preprocessing: rimozione colonne inutili e nulli")
    df = df.drop(columns=["_id", "author", "text", "date", "_class"], errors="ignore")

    df = df.rename(columns={
        "userId": "userId",
        "gameId": "itemId",
        rating_col: "rating"
    })

    df.drop_duplicates(subset=["userId", "itemId"], inplace=True)
    df.dropna(subset=["userId", "itemId", "rating"], inplace=True)

    logger.info("Winsorization dei rating")
    p5, p95 = np.percentile(df["rating"], winsorize_percentiles)
    df["rating"] = df["rating"].clip(lower=p5, upper=p95)

    logger.info(f"Normalizzazione rating in range [{min_rating}, {max_rating}]")
    min_score = df["rating"].min()
    max_score = df["rating"].max()
    df["rating"] = (df["rating"] - min_score) / (max_score - min_score) * (max_rating - min_rating)
    df["rating"] = df["rating"].clip(min_rating, max_rating)

    return df


def split_leave_one_out(df):
    logger.info("Split leave-one-out con Surprise")
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

    loo = LeaveOneOut(n_splits=1, random_state=42)
    trainset, testset = next(loo.split(data))
    logger.info(f"Split effettuato: {trainset.n_ratings} training, {len(testset)} test")

    train_df = pd.DataFrame(trainset.build_testset(), columns=["userId", "itemId", "rating"])
    test_df = pd.DataFrame(testset, columns=["userId", "itemId", "rating"])

    return train_df, test_df


def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Dataset salvato in {path}")


def main():
    logger.info("Avvio preprocessing completo")
    config = load_config()

    df = pd.read_csv(config["input_path"])
    df = preprocess_data(
        df,
        rating_col=config.get("rating_column", "score"),
        min_rating=config.get("min_rating", 0),
        max_rating=config.get("max_rating", 5),
        winsorize_percentiles=config.get("winsorize_percentiles", [5, 95])
    )

    train_df, test_df = split_leave_one_out(df)

    save_df(train_df, config["output_train_path"])
    save_df(test_df, config["output_test_path"])
    logger.info("Preprocessing completato con successo")


if __name__ == "__main__":
    main()