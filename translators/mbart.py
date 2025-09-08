from typing import Dict
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class MBartTranslator:
    """
    mBART-50 many-to-many.
    Requires mBART language codes (map below).
    """
    _MBART_CODE: Dict[str, str] = {
        "en": "en_XX", "fr": "fr_XX", "it": "it_IT", "es": "es_XX",
        "de": "de_DE", "el": "el_GR", "ru": "ru_RU", "ar": "ar_AR",
        "tr": "tr_TR", "nl": "nl_XX", "pt": "pt_XX", "pl": "pl_PL",
        "zh": "zh_CN",
    }

    def __init__(self, source_lang: str, target_lang: str, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        self.src = source_lang.lower()
        self.tgt = target_lang.lower()
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)

        if self.src not in self._MBART_CODE or self.tgt not in self._MBART_CODE:
            raise ValueError(f"Unsupported language (src={self.src}, tgt={self.tgt}) for mBART mapping.")

        self.src_code = self._MBART_CODE[self.src]
        self.tgt_code = self._MBART_CODE[self.tgt]

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        self.tokenizer.src_lang = self.src_code
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        forced_id = self.tokenizer.lang_code_to_id[self.tgt_code]
        out = self.model.generate(**encoded, forced_bos_token_id=forced_id)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
