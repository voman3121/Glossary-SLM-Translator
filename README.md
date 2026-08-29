# Glossary Translator: PDF (English) → SLM → Tamil / Malayalam

Translates a domain glossary PDF into a native Indic language using a
**Small Language Model (SLM)**: [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
(distilled, ~200M parameters), which is purpose-trained for English↔Indic
translation and outperforms generic LLMs on Tamil/Malayalam, especially
for terminology.

## Why IndicTrans2 instead of training an SLM from scratch?

Training a translation model from scratch needs millions of parallel
sentence pairs and heavy compute — impractical for a single glossary.
The practical, standard approach (and what's implemented here) is:

1. **Start from an existing SLM** already trained for your language pair.
2. **Optionally fine-tune it (LoRA)** on your domain's specific terms so
   it learns your preferred/technical translations rather than generic ones.

This gets you production-quality translation with a few hundred domain
examples instead of millions.

## Pipeline

```
glossary.pdf
    │  (glossary_extractor.py — pdfplumber + regex parsing)
    ▼
glossary_en.json      [{term, definition}, ...]
    │  (slm_translator.py — IndicTrans2 inference)
    ▼
bilingual_glossary.csv / .pdf
```

Optional domain adaptation loop:

```
draft translations → native speaker corrects → domain_pairs.csv
    │  (domain_finetune.py — LoRA fine-tune)
    ▼
lora_<lang>_adapter/   (reused by slm_translator.py --adapter)
```

## Setup

```bash
pip install -r requirements.txt
```

You'll also need a Unicode font for PDF output (system/default PDF fonts
don't render Tamil/Malayalam glyphs):
- Tamil: [Noto Sans Tamil](https://fonts.google.com/noto/specimen/Noto+Sans+Tamil)
- Malayalam: [Noto Sans Malayalam](https://fonts.google.com/noto/specimen/Noto+Sans+Malayalam)

## Usage

### 1. Basic run (extract + translate + CSV/PDF)

```bash
python pipeline.py path/to/glossary.pdf tam_Taml ./output --font NotoSansTamil-Regular.ttf
```

For Malayalam:
```bash
python pipeline.py path/to/glossary.pdf mal_Mlym ./output --font NotoSansMalayalam-Regular.ttf
```

### 2. (Optional) Improve accuracy with domain fine-tuning

a. Run the pipeline once to get draft translations in `bilingual_glossary.csv`.
b. Have a native speaker correct the `term_translated` / `definition_translated`
   columns.
c. Reformat corrections into a 2-column CSV: `english,translated`.
d. Fine-tune:
   ```bash
   python domain_finetune.py domain_pairs.csv tam_Taml ./lora_tamil_adapter
   ```
e. Re-run the pipeline using the adapter:
   ```bash
   python pipeline.py glossary.pdf tam_Taml ./output --adapter ./lora_tamil_adapter
   ```

## Files

| File | Purpose |
|---|---|
| `glossary_extractor.py` | Parses term/definition pairs out of the PDF |
| `slm_translator.py` | Loads IndicTrans2 and translates text |
| `domain_finetune.py` | LoRA fine-tunes the SLM on your corrected domain pairs |
| `build_output.py` | Writes CSV and a Unicode PDF |
| `pipeline.py` | Ties everything together (the script you actually run) |

## Notes

- **Supported languages**: any of IndicTrans2's 22 target languages — just
  change the FLORES-200 code (`tam_Taml`, `mal_Mlym`, `hin_Deva`, `tel_Telu`,
  `kan_Knda`, `ben_Beng`, `mar_Deva`, etc.) listed in `slm_translator.py`.
- **Non-Indic languages**: swap `MODEL_NAME` in `slm_translator.py` for
  `facebook/nllb-200-distilled-600M` (supports 200 languages) — the
  `IndicProcessor` pre/post-processing is IndicTrans2-specific, so remove
  those calls and pass text to the tokenizer directly for NLLB.
- **PDF parsing is layout-dependent.** If `glossary_extractor.py` extracts
  0 entries, open a few pages of your PDF, note the exact term/definition
  formatting, and adjust the regex patterns at the top of that file.
- Runs on CPU but is much faster on GPU (`DEVICE` auto-detects CUDA).
