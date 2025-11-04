import pandas as pd
import time
from pymongo import MongoClient
from src.app.model_inference import predict
from src.utils.db import get_db
from src.utils.alerts import send_alert


# Connect to MongoDB
db = get_db()
tweets_collection = db.tweets


def ingest_from_csv(csv_path="data/train.csv"):
    """
    Read tweets from train.csv (no header, tweet text in the last column)
    and insert them into MongoDB with sentiment.
    """

    # Load CSV with no header (6 columns total)
    df = pd.read_csv(csv_path, header=None, encoding="latin-1").head(10000)


    # Rename columns for clarity — your file has 6 columns
    df.columns = ["id", "date", "query", "user", "flag", "text"]

    print(f"📥 Loaded {len(df)} tweets from {csv_path}")

    count = 0
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue

        # Run sentiment analysis
        sentiment = predict(text)

        # Create document for MongoDB
        data = {
            "id": f"csv_{count}",
            "text": text,
            "created_at": (
                str(row["date"])
                if not pd.isna(row["date"])
                else time.strftime("%Y-%m-%d %H:%M:%S")
            ),
            "sentiment": sentiment,
        }

        # Insert into MongoDB
        tweets_collection.insert_one(data)
        print(f"[{sentiment['label'].upper()}] {text[:80]}...")

        # Optional: Alert for strongly negative tweets
        if sentiment["label"].lower() == "negative" and sentiment.get("confidence", 0) > 0.9:
            send_alert(
                "⚠️ Negative Tweet Alert",
                f"Tweet: {text}<br>Confidence: {sentiment['confidence']}",
            )

        count += 1

    print(f"✅ Finished inserting {count} tweets into MongoDB.")


if __name__ == "__main__":
    ingest_from_csv()
