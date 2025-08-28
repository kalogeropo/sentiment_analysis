# evaluators/metrics_evaluator.py
import time
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional, Dict, Callable, Union    
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sacrebleu import corpus_bleu, corpus_chrf

class TranslationMetricsEvaluator:
    def __init__(self, translator: Callable[[str], str], name: str = "Translator"):
        """
        translator: a function or callable object that takes a string and returns the translated string
        """
        self.translator = translator
        self.name = name

    def evaluate(self,
                 source_texts: List[str],
                 reference_texts: List[str],
                 verbose: bool = False,
                 return_predictions: bool = False
                 ) -> Union[Dict, tuple]:

        predictions = []
        timings = []

        for src in source_texts:
            start = time.time()
            translated = self.translator(src)
            end = time.time()

            predictions.append(translated)
            timings.append(end - start)

            if verbose:
                print(f"\nOriginal:   {src}")
                print(f"Reference:  {reference_texts[source_texts.index(src)]}")
                print(f"Predicted:  {translated} (Time: {end - start:.4f}s)")

        bleu = corpus_bleu(predictions, [reference_texts])
        chrf = corpus_chrf(predictions, [reference_texts])

        result = {
            "model": self.name,
            "num_samples": len(source_texts),
            "avg_time_per_text": round(sum(timings) / len(timings), 4),
            "BLEU": round(bleu.score, 2),
            "chrF": round(chrf.score, 2),
            "BLEU_details": bleu.format(),
        }

        if return_predictions:
            from pandas import DataFrame
            df = DataFrame({
                "Source": source_texts,
                "Reference": reference_texts,
                "Prediction": predictions
            })

        return result, df if return_predictions else None

class MetricsEvaluator:
    def __init__(self, predictor: Callable[[str], str], name: str = "UnnamedModel"):
        """
        :param predictor: A callable object or function with a method `predict(text: str) -> str`
        :param name: Name of the model for reporting
        """
        self.predictor = predictor
        self.name = name

    def evaluate(self,
                 texts: List[str],
                 true_labels: List[str],
                 task_type: str = "classification",  
                 verbose: bool = False,
                 return_predictions: bool = False
                 ) -> Dict | tuple:
        predicted_labels = []
        timings = []

        for text in texts:
            start = time.time()
            prediction = self.predictor(text)
            end = time.time()

            predicted_labels.append(prediction)
            timings.append(end - start)

            if verbose:
                print(f"{task_type.title()} | Text: {text}")
                print(f"Predicted: {prediction} (Time: {end - start:.4f}s)\n")

        accuracy = accuracy_score(true_labels, predicted_labels)
        avg_time = sum(timings) / len(timings)

        result = {
            "model": self.name,
            "task": task_type,
            "accuracy": round(accuracy, 4),
            "avg_time_per_text": round(avg_time, 4),
            "max_time": round(max(timings), 4),
            "min_time": round(min(timings), 4),
            "num_samples": len(texts),
            "report": classification_report(true_labels, predicted_labels, output_dict=True, zero_division=0)
        }

        if return_predictions:
            from pandas import DataFrame
            df = DataFrame({
                    "Text": texts,
                    "True Label": true_labels,
                    "Predicted Label": predicted_labels
                })

        return result, df if return_predictions else None




def analyze_prediction_df(df: pd.DataFrame) -> dict:
    """
    Analyze predictions and performance for a classification task.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns "Text", "True Label", "Predicted Label"
        plot (bool): Whether to show matplotlib plots
    
    Returns:
        dict: A dictionary of key statistics and report tables
    """
    results = {}

    # Add text length
    df["Text Length"] = df["Text"].astype(str).apply(len)

    # Classification Report
    report_dict = classification_report(df["True Label"], df["Predicted Label"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    results["classification_report"] = report_df

    # Confusion Matrix
    labels = sorted(set(df["True Label"]).union(set(df["Predicted Label"])))
    cm = confusion_matrix(df["True Label"], df["Predicted Label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    results["confusion_matrix"] = cm_df

    # Accuracy
    accuracy = (df["True Label"] == df["Predicted Label"]).mean()
    results["accuracy"] = round(accuracy, 4)

    # Basic Distributions
    results["true_label_distribution"] = df["True Label"].value_counts()
    results["predicted_label_distribution"] = df["Predicted Label"].value_counts()

    # Avg text length
    results["avg_text_length_by_true_label"] = df.groupby("True Label")["Text Length"].mean().round(2)
    results["avg_text_length_by_predicted_label"] = df.groupby("Predicted Label")["Text Length"].mean().round(2)

    return results

def analyze_language_sentiment_dataframe(df: pd.DataFrame) -> dict:
    """
    Perform statistical analysis on a DataFrame with Text, Language, and Sentiment columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'Text', 'Language', 'Sentiment'.
        plot (bool): Whether to display bar plots (default: True)

    Returns:
        dict: Summary of computed statistics and distributions.
    """

    stats = {}

    # Add text length column
    df['Text Length'] = df['Text'].astype(str).apply(len)

    # Basic Info
    stats["shape"] = df.shape
    stats["columns"] = df.columns.tolist()

    # Class Distributions
    stats["language_distribution"] = df['Language'].value_counts()
    stats["sentiment_distribution"] = df['Sentiment'].value_counts()

    # Text Length Stats
    stats["text_length_stats"] = df['Text Length'].describe()

    # Grouped Analysis
    stats["avg_length_per_language"] = df.groupby('Language')['Text Length'].mean().round(2)
    stats["avg_length_per_sentiment"] = df.groupby('Sentiment')['Text Length'].mean().round(2)

    # Crosstab
    stats["language_vs_sentiment"] = pd.crosstab(df['Language'], df['Sentiment'])

    return stats