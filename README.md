# Twitter Sentiment Analysis using NLP & Streamlit

A complete end-to-end project for performing sentiment analysis on tweets using a fine-tuned transformer model (BERT).
Includes data ingestion from CSV, MongoDB storage, real-time analysis, and a Streamlit dashboard for visualization.

# Features

✅ Load tweet data from CSV or Twitter API

✅ Perform sentiment classification (Positive / Negative / Neutral)

✅ Store analyzed tweets in MongoDB

✅ Interactive Streamlit dashboard with:

# Recent tweet table

Sentiment distribution (bar + pie chart)

Text analysis input box



# Tech Stack
Component	Technology Used
Language	Python
Framework	Streamlit
NLP Model	HuggingFace Transformers
Database	MongoDB
Visualization	Plotly + Streamlit
Alerts	SendGrid API
⚙️ Setup Instructions
1️) Clone / Open the Project
git clone <repo-url>
cd twitter_sentiment_project

2️) Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3️) Install Dependencies
pip install -r requirements.txt

4️) Configure Environment

Create .env file:

MONGO_URI=mongodb://localhost:27017
MONGO_DB=twitter_sentiment
SENDGRID_API_KEY=

5️) Start MongoDB

Run MongoDB locally or via Docker.

6️) Ingest Tweets
python -m src.ingestion.csv_ingest


7)Run the FastAPI Backend

Start the API server for custom text analysis:

uvicorn src.app.main:app --reload

8) Launch Dashboard
streamlit run src/streamlit/dashboard.py


Then visit:
👉 http://localhost:8501

# Output Preview

✅ Bar & Pie charts of sentiment distribution
✅ Table of recent tweets
✅ Real-time text analyzer
✅ Optional negative tweet alerts

# Future Enhancements

Integrate live Twitter API streaming

Add date/time sentiment trend graph

Support multilingual tweet analysis

Include model fine-tuning option

## ⚙️ Setup Instructions

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<your-username>/twitter_sentiment_project.git
cd twitter_sentiment_project
