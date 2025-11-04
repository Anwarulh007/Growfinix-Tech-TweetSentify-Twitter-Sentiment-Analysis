import streamlit as st
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import requests
import plotly.express as px

# Load environment variables
load_dotenv()

# --- Config ---
API_URL = "http://127.0.0.1:8000"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "twitter_sentiment")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
tweets_collection = db.tweets

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Twitter Sentiment Project", layout="wide")

# --- Header ---
st.markdown(
    """
    <div style="text-align: center; padding: 15px; background-color: #1DA1F2; border-radius: 10px; color: white;">
        <h1> TweetSentify - Twitter Sentiment Analysis Project</h1>
        <p style="font-size: 20px;">Analyze and classify tweet sentiments intelligently.</p>
        <p style="font-size: 18px;">
            TweetSentify is a Real-Time Sentiment Analysis System that processes tweets from live Twitter streams or CSV files, 
            classifies them into Positive, Negative, or Neutral sentiments using AI models, 
            and visualizes the insights beautifully.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🧩 Project Architecture")

# --- Dynamically load the image from assets folder ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_PATH = os.path.join(BASE_DIR, "assets", "senti.png")

# --- Display the image ---
st.image(
    IMAGE_PATH,
    caption="TweetSentify System Architecture",
    use_container_width=True
)
# --- Tabs ---
tabs = st.tabs([" Dashboard", " Analyze Text"])

# --- Dashboard Tab ---
with tabs[0]:
    st.markdown("###  Sentiment Overview")
    tweets = list(tweets_collection.find().sort("created_at", -1).limit(200))

    if tweets:
        df = pd.DataFrame([
            {
                "Text": t.get("text", ""),
                "Sentiment": (
                    t["sentiment"]["label"]
                    if isinstance(t.get("sentiment"), dict)
                    else t.get("sentiment", "")
                ),
                "Confidence": (
                    t["sentiment"].get("confidence", "")
                    if isinstance(t.get("sentiment"), dict)
                    else ""
                ),
                "Created At": t.get("created_at", ""),
            }
            for t in tweets
        ])

        # --- Pie Chart ---
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            color="Sentiment",
            color_discrete_map={
                "positive": "#2ECC71",
                "negative": "#E74C3C",
                "neutral": "#F1C40F",
            },
            title="Sentiment Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Display Recent Tweets ---
        st.markdown("### 🧾 Recent Tweets")

        for _, row in df.head(15).iterrows():
            sentiment_color = {
                "positive": "#D4EFDF",
                "negative": "#FADBD8",
                "neutral": "#FCF3CF",
            }.get(row["Sentiment"].lower(), "#EAECEE")

            emoji = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "⚪"
            }.get(row["Sentiment"].lower(), "⚪")

            st.markdown(
                f"""
                <div style="background-color:{sentiment_color}; 
                            padding:12px; 
                            border-radius:10px; 
                            margin-bottom:8px;
                            box-shadow:0 2px 6px rgba(0,0,0,0.1);">
                    <b>{emoji} {row['Sentiment'].capitalize()}</b><br>
                    <span style="font-size:16px;">{row['Text']}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("No tweets found in MongoDB yet. Please ingest the data first.")

# --- Analyze Text Tab ---
with tabs[1]:
    st.markdown("### 💬 Analyze Custom Text")

    st.write(
        "Type any tweet or sentence below to see how the model predicts its sentiment in real-time."
    )

    user_text = st.text_area("Enter your text:", height=120)

    if st.button("Analyze Sentiment"):
        if user_text.strip():
            try:
                resp = requests.post(f"{API_URL}/analyze", params={"text": user_text})
                if resp.status_code == 200:
                    result = resp.json()["sentiment"]
                    label = result["label"].capitalize()
                    confidence = result["confidence"]

                    color_map = {
                        "Positive": "#2ECC71",
                        "Negative": "#E74C3C",
                        "Neutral": "#F1C40F",
                    }

                    st.markdown(
                        f"""
                        <div style="padding:20px; 
                                    border-radius:12px; 
                                    background-color:{color_map.get(label, '#EAECEE')};
                                    box-shadow: 0 3px 8px rgba(0,0,0,0.1); 
                                    text-align:center;">
                            <h3>Sentiment: {label}</h3>
                            <p style="font-size:18px;">Confidence: <b>{confidence:.2f}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("⚠️ Failed to get response from API.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text for analysis.")
