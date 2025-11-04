from transformers import pipeline
import torch
import os

# Load model name from environment or use default
MODEL_NAME = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment")

# Initialize model once on import
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)

# Map model labels to friendly names (for RoBERTa Twitter model)
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}

def predict(text: str):
    """Run sentiment prediction on a text string."""
    result = sentiment_pipeline(text, truncation=True)[0]
    label = LABEL_MAP.get(result["label"], result["label"])
    score = round(float(result["score"]), 4)
    return {"label": label, "confidence": score}
