import pandas as pd

from language_detectors.langdetect_detector import LangDetectDetector
from language_detectors.langid_detector import LangIDDetector
from language_detectors.roberta_detector import RobertaDetector
from language_detectors.fasttext_detector import FastTextDetector

from translators.translators import OpusMtTranslator

from utils.metrics_analyzer import MetricsEvaluator ,analyze_language_sentiment_dataframe, analyze_prediction_df, TranslationMetricsEvaluator

from sklearn.metrics import confusion_matrix

def get_detector(name: str):
    name = name.lower()
    
    if name == "langdetect":
        return LangDetectDetector()
    elif name == "langid":
        return LangIDDetector()
    elif name == "roberta":
        return RobertaDetector()
    elif name == "fasttext":
        return FastTextDetector() 
    else:
        raise ValueError(f"Unknown detector name: {name}")

def run_language_evaluation(df,verbose=False, return_predictions=False):
    print("\n LANGUAGE DETECTION RESULTS\n" + "-"*40)
    texts = df["Text"].tolist()
    labels = df["Language"].tolist()

    for model_name in ["langdetect", "langid", "roberta", "fasttext"]:
        detector = get_detector(model_name)

        evaluator = MetricsEvaluator(detector.detect_language, name=model_name)
        result, new_df = evaluator.evaluate(texts, labels, task_type="language",verbose=verbose, return_predictions=return_predictions)
       
        print(f"{result['model']:>15} | Accuracy: {result['accuracy']:.4f} | Avg Time: {result['avg_time_per_text']:.4f}s")
        print("---------------------------------------------------------------------------------------------------------------")
        #print(result['report'])
        if new_df is not None:
            print(new_df.head())
            new_df_stats = analyze_prediction_df(new_df)
            print(new_df_stats)

            # new_df.to_csv(f"./test_data/{model_name}_predictions.csv", index=False)
            # print(f"Predictions saved to {model_name}_predictions.csv\n")


if __name__ == "__main__":
    #Load the sample data
    df = pd.read_csv("Libeccio_ML/sentiment_analysis/test_data/sample_texts.csv")
    
    summary = analyze_language_sentiment_dataframe(df)
    print(summary)
    # Run language detection evaluation
    run_language_evaluation(df,return_predictions=True)

    df = pd.read_csv("Libeccio_ML/sentiment_analysis/test_data/multilingual_translation_eval_with_reference.csv")
    source_texts = df["Text"].tolist()
    reference_texts = df["Reference"].tolist()
    print(df.head())
    
    for lang in df["Language"].unique():
        subset = df[df["Language"] == lang]
        translator = OpusMtTranslator(lang, "en")
        evaluator = TranslationMetricsEvaluator(translator.translate, name=f"{lang}-to-en")
        results, _ = evaluator.evaluate(subset["Text"].tolist(), subset["Reference"].tolist())
        print(results)

