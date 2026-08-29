"""
pipeline.py
-----------
End-to-end: English glossary PDF -> SLM translation -> bilingual output.

Usage:
    python pipeline.py glossary.pdf tam_Taml output/ [--adapter ./lora_tamil_adapter] [--font NotoSansTamil-Regular.ttf]

tgt_lang codes:
    tam_Taml -> Tamil
    mal_Mlym -> Malayalam
    (see slm_translator.py for more FLORES-200 codes)
"""

import argparse
import json
import os

from glossary_extractor import extract_glossary
from slm_translator import GlossaryTranslator
from build_output import write_csv, write_pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the English glossary PDF")
    parser.add_argument("tgt_lang", help="Target language code, e.g. tam_Taml or mal_Mlym")
    parser.add_argument("out_dir", help="Directory to write outputs into")
    parser.add_argument("--adapter", default=None, help="Path to a LoRA domain adapter (optional)")
    parser.add_argument("--font", default=None, help="Path to a Unicode .ttf font for PDF output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("1/3  Extracting glossary from PDF...")
    glossary = extract_glossary(args.pdf_path)
    print(f"     -> {len(glossary)} entries found")
    with open(os.path.join(args.out_dir, "glossary_en.json"), "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    if not glossary:
        print("No entries extracted -- check the PDF layout / patterns in glossary_extractor.py")
        return

    print("2/3  Translating with the SLM (IndicTrans2)...")
    translator = GlossaryTranslator(adapter_path=args.adapter)
    translated = translator.translate_glossary(glossary, args.tgt_lang)

    print("3/3  Writing outputs...")
    write_csv(translated, os.path.join(args.out_dir, "bilingual_glossary.csv"))

    if args.font:
        write_pdf(translated, os.path.join(args.out_dir, "bilingual_glossary.pdf"), args.font)
    else:
        print("     (skipped PDF output -- pass --font path/to/NotoSans<Lang>.ttf to generate one)")

    print("\nDone.")


if __name__ == "__main__":
    main()
