from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
import os

from src.app.model_inference import predict
from src.utils.db import get_db

# Load environment variables
load_dotenv()

app = FastAPI(title="Twitter Sentiment Analysis API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
db = get_db()
tweets_collection = db.tweets


@app.get("/")
async def root():
    return {"message": "✅ Twitter Sentiment API is running!"}


@app.get("/recent")
async def get_recent_tweets(limit: int = 50):
    """Fetch most recent tweets from MongoDB."""
    data = list(tweets_collection.find().sort("created_at", -1).limit(limit))
    for doc in data:
        doc["_id"] = str(doc["_id"])
    return data


@app.post("/analyze")
async def analyze_text(text: str):
    """Perform sentiment analysis on custom text."""
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = predict(text)
    return {"text": text, "sentiment": result}


@app.post("/annotate/{tweet_id}")
async def annotate_tweet(tweet_id: str):
    """Update sentiment for a stored tweet by its ID."""
    tweet = tweets_collection.find_one({"id": tweet_id})
    if not tweet:
        raise HTTPException(status_code=404, detail="Tweet not found.")
    sentiment = predict(tweet["text"])
    tweets_collection.update_one({"id": tweet_id}, {"$set": {"sentiment": sentiment}})
    return {"id": tweet_id, "sentiment": sentiment}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
