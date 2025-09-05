import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

DetectorFactory.seed = 0

reviews_df = pd.read_csv('reviews_new.csv', low_memory=False)

device = 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=device)
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)

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

for i in range(5000):
    print("Proccessing Review: ", i)
    result = preprocess_and_analyze1(reviews_df['context'].iloc[i])
    sent = result["sentiment"]
    try:
        sen_scores.append(sent["score"])
        sen_labels.append(sent["label"])
    except:
        sen_scores.append(None)
        sen_labels.append(None)

    emo = result["emotion"]
    first_result = emo[0][0]
    try:
        label = first_result['label']
        score = first_result['score']
        emo_labels.append(label)
        emo_scores.append(score)
    except:
        emo_labels.append(None)
        emo_scores.append(None)

    aspects.append(result["aspects"])

reviews_df_reduced = reviews_df.iloc[:5000]

reviews_df_reduced['sentiment_scores'] = sen_scores
reviews_df_reduced['sentiment_labels'] = sen_labels
reviews_df_reduced['emotion_scores'] = emo_scores
reviews_df_reduced['emotion_labels'] = emo_labels
reviews_df_reduced['aspects'] = aspects

reviews_df_reduced.to_csv("reviews_df_Roberta1.csv", index=False)

