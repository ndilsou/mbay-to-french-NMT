"""Train NLLB-200-distilled-600M (extended) for Mbay translation.

Loads the tokenizer-extended model from disk, tokenizes the unified parquet
corpus using NLLB's native src_lang/tgt_lang mechanism, and runs full
fine-tuning with chrF++ as primary metric.

Usage:
    uv run python -m mbay_nmt.train_nllb.train \\
        --model-dir models/nllb-600m-mbay-extended \\
        --data datasets/processed/mbay-corpus-v1.parquet \\
        --output-dir models/nllb-600m-mbay-finetuned
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
)

from mbay_nmt.tokenizer.nllb_extend import PIPELINE_TO_NLLB

PROJECT_NAME = "mbay-nmt-nllb"

chrf_metric = evaluate.load("chrf")
bleu_metric = evaluate.load("sacrebleu")


@dataclass
class Args:
    model_dir: str = field(metadata={"help": "Path to extended NLLB model dir"})
    data: str = field(metadata={"help": "Path to unified parquet corpus"})
    output_dir: str = field(default="models/nllb-600m-mbay-finetuned")
    epochs: int = field(default=5)
    batch_size: int = field(default=8)
    grad_accum: int = field(default=4)
    lr: float = field(default=3e-5)
    warmup_ratio: float = field(default=0.05)
    max_length: int = field(default=128)
    seed: int = field(default=42)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    eval_steps: int = field(default=500)
    save_steps: int = field(default=500)
    early_stopping_patience: int = field(default=5)
    wandb_project: str = field(default=PROJECT_NAME)
    use_wandb: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)


def load_corpus_as_hf_dataset(parquet_path: str, split: str = "train") -> Dataset:
    df = pd.read_parquet(parquet_path)
    df = df[df["split"] == split].reset_index(drop=True)
    return Dataset.from_pandas(df)


def tokenize_fn(
    examples: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict:
    """Tokenize a batch using NLLB's native language-code mechanism."""
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    src_langs = examples["source_lang"]
    tgt_langs = examples["target_lang"]
    src_texts = examples["source_text"]
    tgt_texts = examples["target_text"]

    for src_lang, tgt_lang, src_text, tgt_text in zip(src_langs, tgt_langs, src_texts, tgt_texts):
        src_nllb = PIPELINE_TO_NLLB.get(src_lang, src_lang)
        tgt_nllb = PIPELINE_TO_NLLB.get(tgt_lang, tgt_lang)

        # Encode source with source language prefix.
        tokenizer.src_lang = src_nllb
        model_inputs = tokenizer(
            src_text,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        # Encode target: set src_lang to target (NLLB uses src_lang for
        # prefix on both source and target sides).
        tokenizer.src_lang = tgt_nllb
        labels = tokenizer(
            tgt_text,
            max_length=max_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]

        all_input_ids.append(model_inputs["input_ids"])
        all_attention_mask.append(model_inputs["attention_mask"])
        all_labels.append(model_inputs["labels"])

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def compute_metrics(tokenizer: AutoTokenizer, eval_preds) -> dict:
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [ref.strip() for ref in decoded_labels]

    chrf_result = chrf_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        word_order=2,  # chrF++
    )
    bleu_result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
    )

    return {
        "chrf++": round(chrf_result["score"], 2),
        "bleu": round(bleu_result["score"], 2),
        "gen_len": round(np.mean([len(p.split()) for p in decoded_preds]), 1),
    }


def main():
    parser = HfArgumentParser(Args)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load and tokenize data.
    train_ds = load_corpus_as_hf_dataset(args.data, "train")
    val_ds = load_corpus_as_hf_dataset(args.data, "val")

    tok_fn = partial(tokenize_fn, tokenizer=tokenizer, max_length=args.max_length)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok_fn, batched=True, remove_columns=val_ds.column_names)

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    report_to = "wandb" if args.use_wandb else "none"
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf++",
        greater_is_better=True,
        logging_steps=50,
        report_to=report_to,
        run_name=f"nllb-mbay-{args.epochs}ep",
        predict_with_generate=True,
        generation_max_length=args.max_length,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, tokenizer),
    )
    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    )

    trainer.train()

    # Save final model.
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if args.use_lora:
        from peft import AutoPeftModelForSeq2SeqLM

        model.save_pretrained(args.output_dir)
        del model
        torch.cuda.empty_cache()
        model = AutoPeftModelForSeq2SeqLM.from_pretrained(
            args.output_dir, torch_dtype=torch.float16
        )
        model = model.merge_and_unload()
        model.save_pretrained(final_dir, safe_serialization=True)
    else:
        trainer.save_model(final_dir)

    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
