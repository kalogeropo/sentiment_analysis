import pandas as pd
import langid
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langdetect import detect
from deep_translator import GoogleTranslator

reviews_df = pd.read_csv('reviews_new.csv', low_memory=False)

nltk.download('punkt')
nltk.download('stopwords')

device = 0 if torch.cuda.is_available() else -1

sentiment_model = "nlptown/bert-base-multilingual-uncased-sentiment"
classifier = pipeline("sentiment-analysis", model=sentiment_model, device=device)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
kw_model = KeyBERT(model=embedding_model)

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def extract_keywords(text, lang='en'):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=lang,
        top_n=5
    )
    return [kw for kw, _ in keywords]

def preprocess_and_analyze2(review):
    lang, _ = langid.classify(review)
    sentiment = classifier(review)[0]
    lang = detect(review)
    if lang != "en":
        translated = GoogleTranslator(source='auto', target='en').translate(review)
    else:
        translated = review
    aspects = []
    keywords = ["room", "staff", "location", "bed", "noise", "view", "breakfast", "food", "transport", "pool", "bar", "restaurant"]
    for keyword in keywords:
        if keyword in translated.lower():
            aspects.append(keyword)
        
    ensure_nltk_data()
    tokens = word_tokenize(review.lower())
    sw = set(stopwords.words('english') if lang == 'en' else [])
    keywords = [word for word in tokens if word.isalpha() and word not in sw]

    key_phrases = extract_keywords(review, lang='english')

    return {
        "sentiment": sentiment,
        "key_phrases": key_phrases,
        "aspects": aspects
    }


sen_labels = []
sen_scores = []
aspects = []

for i in range(5000):
    print("Processing Review:", i)
    try:
        result = preprocess_and_analyze2(reviews_df['context'][i])
        score = result["sentiment"]["score"]
        label = result["sentiment"]["label"]
        sen_scores.append(score)
        sen_labels.append(label)
        aspects.append(result["aspects"])
    except:
        sen_scores.append(None)
        sen_labels.append(None)
        aspects.append(None)

reviews_df_reduced = reviews_df.iloc[:5000]

reviews_df_reduced['sen_scores'] = sen_scores
reviews_df_reduced['sen_labels'] = sen_labels
reviews_df_reduced['aspects'] = aspects

reviews_df_reduced.to_csv("reviews_df_KeyBert1.csv", index=False)