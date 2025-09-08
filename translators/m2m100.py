from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class M2M100Translator:
    """
    Meta M2M100 (many-to-many, autodetect not required but supports src/tgt tokens).
    Uses ISO-like codes (e.g., 'en', 'fr', 'el', 'de', 'zh').
    """
    def __init__(self, source_lang: str, target_lang: str, model_name: str = "facebook/m2m100_418M"):
        self.src = source_lang.lower()
        self.tgt = target_lang.lower()
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        self.tokenizer.src_lang = self.src
        encoded = self.tokenizer([text], return_tensors="pt", truncation=True)
        forced_id = self.tokenizer.get_lang_id(self.tgt)
        out = self.model.generate(**encoded, forced_bos_token_id=forced_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
