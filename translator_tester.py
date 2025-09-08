# translation_tester.py
import csv
from translators import OpusMtTranslator, M2M100Translator, NLLBTranslator, MBartTranslator

TRANSLATORS = {
    "opus": OpusMtTranslator,
    "m2m100": M2M100Translator,
    "nllb": NLLBTranslator,
    "mbart": MBartTranslator,
}

def run_tests(input_csv="test_data/chat_generated_text_translations.csv", output_csv="results.csv"):
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_data = list(reader)

    results = []
    for row in test_data:
        text_id = row["id"]
        src_lang = row["lang"]
        text = row["text"]
        reference = row["english_reference"]

        for name, cls in TRANSLATORS.items():
            try:
                translator = cls(src_lang, "en")
                translated = translator.translate(text)
            except Exception as e:
                translated = f"ERROR: {e}"

            results.append({
                "id": text_id,
                "src_lang": src_lang,
                "translator": name,
                "input_text": text,
                "reference": reference,
                "translation": translated,
            })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "src_lang", "translator", "input_text", "reference", "translation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Testing completed. Results saved to {output_csv}")


if __name__ == "__main__":
    run_tests()
