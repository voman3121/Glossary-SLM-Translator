"""
domain_finetune.py
-------------------
Optional step: LoRA fine-tunes the IndicTrans2 SLM on YOUR domain glossary,
so it learns the specific/technical translations you want instead of
generic ones (e.g. a legal, medical, or engineering glossary often has
terms a general model gets wrong).

Requires a parallel file: a CSV with columns "english,translated"
where "translated" is the correct Tamil/Malayalam translation for that
term or definition. This is typically produced by:
    1. Running slm_translator.py once to get draft translations
    2. Having a native speaker correct the drafts
    3. Saving the corrected pairs as domain_pairs.csv

Usage:
    python domain_finetune.py domain_pairs.csv tam_Taml ./lora_tamil_adapter
"""

import sys
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_LANG = "eng_Latn"


def load_pairs(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    assert {"english", "translated"}.issubset(df.columns), \
        "CSV must have 'english' and 'translated' columns"
    return Dataset.from_pandas(df[["english", "translated"]])


def build_preprocess_fn(tokenizer, ip, tgt_lang):
    def preprocess(batch):
        src = ip.preprocess_batch(batch["english"], src_lang=SRC_LANG, tgt_lang=tgt_lang)
        tgt = ip.preprocess_batch(batch["translated"], src_lang=tgt_lang, tgt_lang=tgt_lang)

        model_inputs = tokenizer(src, truncation=True, max_length=256, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(tgt, truncation=True, max_length=256, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess


def main(csv_path: str, tgt_lang: str, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ip = IndicProcessor(inference=False)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # attention projections
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_pairs(csv_path)
    preprocess_fn = build_preprocess_fn(tokenizer, ip, tgt_lang)
    tokenized = dataset.map(preprocess_fn, batched=True, remove_columns=dataset.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA domain adapter saved to {output_dir}")
    print(f"Load it later with: GlossaryTranslator(adapter_path='{output_dir}')")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python domain_finetune.py domain_pairs.csv <tgt_lang_code> <output_dir>")
        print("Example: python domain_finetune.py domain_pairs.csv tam_Taml ./lora_tamil_adapter")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
