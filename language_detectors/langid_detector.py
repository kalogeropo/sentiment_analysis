import langid

from language_detectors.language_detector import LanguageDetectorBase

class LangIDDetector(LanguageDetectorBase):
    def detect_language(self, text: str) -> str:
        lang, _ = langid.classify(text)
        return lang