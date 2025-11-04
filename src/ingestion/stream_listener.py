import os
import time
from tweepy import StreamingClient, StreamRule
from pymongo import MongoClient
from dotenv import load_dotenv
from src.ingestion.preprocess import clean_tweet
from src.app.model_inference import predict
from src.utils.alerts import send_alert

# Load environment variables
load_dotenv()

# Environment variables
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "twitter_sentiment")

# Database setup
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
tweets_collection = db.tweets


class MyStream(StreamingClient):
    """Custom Tweepy stream client."""

    def on_connect(self):
        print("✅ Connected to Twitter stream API.")

    def on_tweet(self, tweet):
        try:
            text = clean_tweet(tweet.text)
            sentiment = predict(text)

            data = {
                "id": str(tweet.id),
                "text": text,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment,
            }
            tweets_collection.insert_one(data)
            print(f"[{sentiment['label'].upper()}] {text}")

            # Optional: Send alert for very negative tweets
            if sentiment["label"] == "negative" and sentiment["confidence"] > 0.9:
                send_alert(
                    "⚠️ Negative Tweet Alert",
                    f"Tweet: {text}<br>Confidence: {sentiment['confidence']}",
                )

        except Exception as e:
            print("❌ Error processing tweet:", e)

    def on_errors(self, errors):
        print("⚠️ Stream error:", errors)


def start_stream(keyword="AI"):
    """Start streaming tweets containing a specific keyword."""
    if not BEARER_TOKEN:
        raise ValueError("Missing Twitter Bearer Token. Set it in your .env file.")

    stream = MyStream(bearer_token=BEARER_TOKEN)
    print(f"🚀 Starting stream for keyword: '{keyword}'")

    # Remove old rules
    rules = stream.get_rules().data
    if rules:
        rule_ids = [r.id for r in rules]
        stream.delete_rules(rule_ids)

    # Add new rule
    stream.add_rules(StreamRule(f"{keyword} -is:retweet lang:en"))

    # Start filtering stream
    try:
        stream.filter(tweet_fields=["created_at", "lang"])
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped by user.")
    except Exception as e:
        print("❌ Stream failed:", e)


if __name__ == "__main__":
    kw = input("Enter keyword or hashtag to track (default: AI): ") or "AI"
    start_stream(kw)
