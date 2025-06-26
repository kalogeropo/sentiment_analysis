from langdetect import detect

from language_detectors.language_detector import LanguageDetectorBase


class LangDetectDetector(LanguageDetectorBase):
    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"
