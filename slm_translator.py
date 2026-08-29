"""
slm_translator.py
------------------
Translates English text to an Indic language (Tamil, Malayalam, etc.)
using AI4Bharat's IndicTrans2 distilled model -- a Small Language Model
(200M params) purpose-built for English<->Indic translation.

Model card: ai4bharat/indictrans2-en-indic-dist-200M

Swap MODEL_NAME for the 1B version ("ai4bharat/indictrans2-en-indic-1B")
if you have more compute and want higher quality.

FLORES-200 language codes used by IndicTrans2 (a few common ones):
    English    -> eng_Latn
    Tamil      -> tam_Taml
    Malayalam  -> mal_Mlym
    Hindi      -> hin_Deva
    Telugu     -> tel_Telu
    Kannada    -> kan_Knda
    Bengali    -> ben_Beng
    Marathi    -> mar_Deva
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_LANG = "eng_Latn"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GlossaryTranslator:
    def __init__(self, model_name: str = MODEL_NAME, adapter_path: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(DEVICE)

        # Load a LoRA domain-adapter if you fine-tuned one with domain_finetune.py
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).to(DEVICE)

        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    def translate_batch(self, texts: list[str], tgt_lang: str, batch_size: int = 16) -> list[str]:
        """tgt_lang e.g. 'tam_Taml' for Tamil, 'mal_Mlym' for Malayalam."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            batch = self.ip.preprocess_batch(chunk, src_lang=SRC_LANG, tgt_lang=tgt_lang)
            inputs = self.tokenizer(
                batch, truncation=True, padding="longest",
                return_tensors="pt", max_length=256,
            ).to(DEVICE)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(self.ip.postprocess_batch(decoded, lang=tgt_lang))

        return results

    def translate_glossary(self, glossary: list[dict], tgt_lang: str) -> list[dict]:
        """glossary: [{'term': ..., 'definition': ...}, ...]"""
        terms = [g["term"] for g in glossary]
        defs = [g["definition"] for g in glossary]

        translated_terms = self.translate_batch(terms, tgt_lang)
        translated_defs = self.translate_batch(defs, tgt_lang)

        out = []
        for src, term_t, def_t in zip(glossary, translated_terms, translated_defs):
            out.append({
                "term_en": src["term"],
                "definition_en": src["definition"],
                "term_translated": term_t,
                "definition_translated": def_t,
            })
        return out


if __name__ == "__main__":
    # Quick smoke test
    translator = GlossaryTranslator()
    sample = [
        {"term": "Latency", "definition": "The delay before a transfer of data begins."},
        {"term": "Throughput", "definition": "The amount of data processed in a given time."},
    ]
    for lang_name, code in [("Tamil", "tam_Taml"), ("Malayalam", "mal_Mlym")]:
        print(f"\n--- {lang_name} ---")
        for row in translator.translate_glossary(sample, code):
            print(row)
