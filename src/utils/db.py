from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def get_db():
    """Connect to MongoDB and return the database instance."""
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "twitter_sentiment")
    client = MongoClient(mongo_uri)
    return client[db_name]
