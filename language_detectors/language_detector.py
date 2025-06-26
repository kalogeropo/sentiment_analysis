from abc import ABC, abstractmethod

class LanguageDetectorBase(ABC):
    
    @abstractmethod
    def detect_language(self, text: str) -> str:
        """Detect the language of a given text"""
        pass






