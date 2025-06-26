import os
import urllib.request
import fasttext
from language_detectors.language_detector import LanguageDetectorBase

class FastTextDetector(LanguageDetectorBase):

    def __init__(self, model_path="data/lid.176.ftz"):
        self.model_path = model_path
        self._ensure_model_exists()
        self.model = fasttext.load_model(self.model_path)

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            print(f"FastText model not found at {self.model_path}. Downloading...")
            directory = os.path.dirname(self.model_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def detect_language(self, text: str) -> str:
        label = self.model.predict(text, k=1)[0][0]  # Only take top-1 label
        return label.replace("__label__", "")
