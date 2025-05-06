import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

#TODO: Utilizzare l`API Gateway quando i microservizi saranno pronti

def build_mongo_uri():
    load_dotenv()
    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASSWORD")
    cluster = os.getenv("MONGO_CLUSTER")
    db_name = os.getenv("MONGO_DB")
    auth_db = os.getenv("MONGO_AUTH_DB")

    return f"mongodb+srv://{user}:{password}@{cluster}/{db_name}?authSource={auth_db}&retryWrites=true&w=majority"


def fetch_and_save_collection(collection_name, output_path):
    uri = build_mongo_uri()
    client = MongoClient(uri)
    db = client[os.getenv("MONGO_DB")]
    collection = db[collection_name]
    data = list(collection.find({}))
    df = pd.DataFrame(data)

    df.to_csv(output_path, index=False)
    print(f"✅ Salvata collection '{collection_name}' in {output_path}")


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    fetch_and_save_collection("games", "data/raw/games.csv")
    fetch_and_save_collection("reviews", "data/raw/reviews.csv")