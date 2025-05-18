import os
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.log_config import setup_logger

# Setup logger
logger = setup_logger("data_ingestion")

load_dotenv()


def build_mongo_uri(db_key: str, db_name: str) -> str:
    user = os.getenv(f"MONGO_USER_{db_key}")
    password = os.getenv(f"MONGO_PASSWORD_{db_key}")
    cluster = os.getenv(f"MONGO_CLUSTER_{db_key}")
    auth_db = os.getenv(f"MONGO_AUTH_DB_{db_key}")

    if not all([user, password, cluster, auth_db]):
        logger.error(f"Variabili d'ambiente mancanti per '{db_key}'")
        raise ValueError(f"Variabili d'ambiente mancanti per '{db_key}'")

    return f"mongodb+srv://{user}:{password}@{cluster}/{db_name}?authSource={auth_db}&retryWrites=true&w=majority"


def fetch_and_save(uri: str, db_name: str, collection_name: str, output_path: str):
    logger.info(f"Connessione a MongoDB per collection '{collection_name}' nel DB '{db_name}'")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    data = list(collection.find({}))
    if not data:
        logger.warning(f"La collection '{collection_name}' nel DB '{db_name}' è vuota.")
        return

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Salvata collection '{collection_name}' da DB '{db_name}' in '{output_path}'")


def main(config_path: str = "config/ingestion.yaml"):
    logger.info("Avvio processo di data ingestion")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Errore nella lettura del file di configurazione: {e}")
        return

    for source in config.get("sources", []):
        db_key = source["db_key"]
        db_name = source["db_name"]
        try:
            uri = build_mongo_uri(db_key, db_name)
        except Exception as e:
            logger.error(f"Errore nella costruzione della URI per '{db_key}': {e}")
            continue

        for col in source.get("collections", []):
            try:
                collection_name = col["name"]
                output_path = col["output"]
                fetch_and_save(uri, db_name, collection_name, output_path)
            except Exception as e:
                logger.error(f"Errore nella fetch della collection '{col['name']}': {e}")

    logger.info("Data ingestion completata.")


if __name__ == "__main__":
    main()