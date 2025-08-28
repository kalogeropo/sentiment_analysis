import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

DetectorFactory.seed = 0

reviews_df = pd.read_csv('reviews_new.csv', low_memory=False)

sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def preprocess_and_analyze1(review):
    if pd.isna(review):
        return {"sentiment": {"label": None, "score": None}, "emotion": [], "aspects": []}
    s = str(review).strip()
    if s == "":
        return {"sentiment": {"label": None, "score": None}, "emotion": [], "aspects": []}
    try:
        lang = detect(s)
    except LangDetectException:
        lang = "en"
    translated = s
    if lang != "en":
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(s)
            if translated is None:
                translated = s
        except Exception:
            translated = s
    try:
        sent_out = sentiment_analyzer(translated, truncation=True, max_length=512)
        sentiment = sent_out[0] if isinstance(sent_out, list) and len(sent_out) > 0 else sent_out
    except Exception:
        sentiment = {"label": None, "score": None}
    try:
        emo_out = emotion_analyzer(translated, truncation=True, max_length=512)
        emotion = [emo_out] if isinstance(emo_out, dict) else emo_out
    except Exception:
        emotion = []
    keywords = ["room", "staff", "location", "bed", "noise", "view", "breakfast", "food", "transport", "pool", "bar", "restaurant"]
    aspects = [k for k in keywords if k in translated.lower()]
    return {"sentiment": sentiment, "emotion": emotion, "aspects": aspects}


sen_labels = []
sen_scores = []
emo_scores = []
emo_labels = []
aspects = []

for i in range(len(reviews_df)):
    result = preprocess_and_analyze1(reviews_df['context'].iloc[i])
    sent = result["sentiment"]
    if isinstance(sent, dict):
        sen_scores.append(sent.get("score"))
        sen_labels.append(sent.get("label"))
    else:
        sen_scores.append(None)
        sen_labels.append(None)

    emo = result["emotion"]
    if isinstance(emo, list) and len(emo) > 0 and isinstance(emo[0], dict):
        emo_labels.append(emo[0].get("label"))
        emo_scores.append(emo[0].get("score"))
    else:
        emo_labels.append(None)
        emo_scores.append(None)

    aspects.append(result["aspects"])

reviews_df['sentiment_scores'] = sen_scores
reviews_df['sentiment_labels'] = sen_labels
reviews_df['emotion_scores'] = emo_scores
reviews_df['emotion_labels'] = emo_labels
reviews_df['aspects'] = aspects

