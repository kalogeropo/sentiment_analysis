import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy.sql import text
from transformers import pipeline

# Load .env file
load_dotenv()

# Get DB credentials
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")
dialect = os.getenv("DB_DIALECT")  # e.g., mysql or postgresql

connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
engine = create_engine(connection_string)
query = text("SELECT id,id_acc, context, language, rev_id FROM reviews_new")
df = pd.read_sql(query, engine)
print(df.head())

# Load the model
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english",
    framework="pt", device=-1
)


# Function to get sentiment label
def get_sentiment(text):
    try:
        if not isinstance(text, str) or text.strip() == "":
            return "unknown"
        result = sentiment_analysis(text[:512])[0]  # Truncate to 512 tokens
        return result["label"]
    except Exception as e:
        print(f"Error on text: {text}\n{e}")
        return "error"



# Apply to your DataFrame
df["sentiment"] = df["context"].apply(get_sentiment)

# Show the results
print(df[["context", "sentiment"]].head())


print(df.head())
print(df.columns)
