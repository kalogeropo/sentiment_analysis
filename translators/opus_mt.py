from transformers import MarianMTModel, MarianTokenizer

class OpusMtTranslator:
    """
    Translator using Helsinki-NLP MarianMT models via Hugging Face.
    Example: source_lang = 'fr', target_lang = 'en'
    """
    def __init__(self, source_lang: str, target_lang: str):
        self.src = source_lang.lower()
        self.tgt = target_lang.lower()
        self.model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.tgt}"

        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)
        except Exception as e:
            self.tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-mul-{self.tgt}")
            self.model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-mul-{self.tgt}")

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True)
        translated_tokens = self.model.generate(**inputs)
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)