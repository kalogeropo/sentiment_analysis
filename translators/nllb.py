from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class NLLBTranslator:
    """Meta NLLB-200 distilled."""
    _CODE = {
        "en": "eng_Latn", "fr": "fra_Latn", "it": "ita_Latn", "es": "spa_Latn",
        "de": "deu_Latn", "el": "ell_Grek", "ru": "rus_Cyrl", "ar": "arb_Arab",
        "tr": "tur_Latn", "nl": "nld_Latn", "pt": "por_Latn", "pl": "pol_Latn",
        "zh": "zho_Hans",
    }

    def __init__(self, source_lang: str, target_lang: str, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.src = source_lang.lower()
        self.tgt = target_lang.lower()

        if self.src not in self._CODE or self.tgt not in self._CODE:
            raise ValueError(f"NLLB unsupported mapping: src={self.src}, tgt={self.tgt}")

        self.src_code = self._CODE[self.src]
        self.tgt_code = self._CODE[self.tgt]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        # Set source language
        self.tokenizer.src_lang = self.src_code

        # Encode input
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)

        # Convert target language code to id (replacement for lang_code_to_id)
        forced_id = self.tokenizer.convert_tokens_to_ids(self.tgt_code)

        # Generate
        out = self.model.generate(**encoded, forced_bos_token_id=forced_id)

        # Decode output
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

