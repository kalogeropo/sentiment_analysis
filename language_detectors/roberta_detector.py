from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

from language_detectors.language_detector import LanguageDetectorBase


class RobertaDetector(LanguageDetectorBase):
    def __init__(self, model_name="papluca/xlm-roberta-base-language-detection"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def detect_language(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        return self.id2label[label_id]
