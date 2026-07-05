"""Train NLLB-200-distilled-600M (extended) for Mbay translation.

Loads the tokenizer-extended model from disk, tokenizes the unified parquet
corpus using NLLB's native src_lang/tgt_lang mechanism, and runs full
fine-tuning with eval loss for checkpoint selection plus lightweight
per-direction forced-BOS generation diagnostics.

Usage:
    uv run python -m mbay_nmt.train_nllb.train \\
        --model-dir models/nllb-600m-mbay-extended \\
        --data datasets/processed/mbay-corpus-v1.parquet \\
        --output-dir models/nllb-600m-mbay-finetuned
"""

from __future__ import annotations

import logging
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

os.environ.setdefault("HF_EVALUATE_CACHE", "/tmp/hf-evaluate-cache")
os.environ.setdefault("HF_METRICS_CACHE", "/tmp/hf-metrics-cache")

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
    TrainerCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    HfArgumentParser,
    set_seed,
)

from mbay_nmt.tokenizer.nllb_extend import PIPELINE_TO_NLLB

PROJECT_NAME = "mbay-nmt-nllb"
MBAY_EVAL_DIRECTIONS = {("mbay", "eng"), ("mbay", "fra"), ("eng", "mbay"), ("fra", "mbay")}
LOGGER = logging.getLogger(__name__)


chrf_metric = evaluate.load("chrf")
bleu_metric = evaluate.load("sacrebleu")


class PerDirectionChrfCallback(TrainerCallback):
    """Run small forced-BOS generation evals per Mbay direction."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        raw_val_df: pd.DataFrame,
        sample_size_per_direction: int = 400,
        batch_size: int = 32,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        eval_every: int = 1,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.eval_every = max(1, eval_every)
        self.eval_count = 0
        self.samples = self._sample_by_direction(raw_val_df, sample_size_per_direction, seed)

    @staticmethod
    def _sample_by_direction(df: pd.DataFrame, sample_size: int, seed: int) -> dict[tuple[str, str], pd.DataFrame]:
        samples: dict[tuple[str, str], pd.DataFrame] = {}
        for direction in sorted(MBAY_EVAL_DIRECTIONS):
            src_lang, tgt_lang = direction
            direction_df = df[(df["source_lang"] == src_lang) & (df["target_lang"] == tgt_lang)]
            if direction_df.empty:
                continue

            parts = []
            for _, domain_df in direction_df.groupby("domain", dropna=False):
                n_domain = max(1, round(sample_size * len(domain_df) / len(direction_df)))
                parts.append(domain_df.sample(n=min(len(domain_df), n_domain), random_state=seed))
            sampled = pd.concat(parts)
            if len(sampled) > sample_size:
                sampled = sampled.sample(n=sample_size, random_state=seed)
            samples[direction] = sampled.reset_index(drop=True)
        return samples

    def _generate_direction(
        self,
        model: AutoModelForSeq2SeqLM,
        rows: pd.DataFrame,
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        src_nllb = PIPELINE_TO_NLLB[src_lang]
        tgt_nllb = PIPELINE_TO_NLLB[tgt_lang]
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb)
        if forced_bos_token_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Target language token {tgt_nllb!r} resolves to UNK")

        device = next(model.parameters()).device
        predictions: list[str] = []
        source_max_length = self.tokenizer.model_max_length
        if source_max_length > 4096:
            source_max_length = 1024

        for start in range(0, len(rows), self.batch_size):
            batch = rows.iloc[start : start + self.batch_size]
            self.tokenizer.src_lang = src_nllb
            inputs = self.tokenizer(
                batch["source_text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=source_max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
            predictions.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [prediction.strip() for prediction in predictions]

    def on_evaluate(self, args, state, control, **kwargs):
        self.eval_count += 1
        if self.eval_count % self.eval_every != 0:
            return

        model = kwargs.get("model")
        if model is None:
            LOGGER.warning("Skipping chrF eval: callback did not receive a model")
            return

        was_training = model.training
        old_src_lang = getattr(self.tokenizer, "src_lang", None)
        try:
            model.eval()
            results: dict[str, float] = {}
            sample_rows = []
            chrf_scores = []
            bleu_scores = []

            with torch.no_grad():
                for src_lang, tgt_lang in sorted(self.samples):
                    rows = self.samples[(src_lang, tgt_lang)]
                    predictions = [
                        unicodedata.normalize("NFC", prediction)
                        for prediction in self._generate_direction(model, rows, src_lang, tgt_lang)
                    ]
                    references = [
                        unicodedata.normalize("NFC", str(ref).strip()) for ref in rows["target_text"].tolist()
                    ]

                    chrf = chrf_metric.compute(
                        predictions=predictions,
                        references=references,
                        word_order=2,
                    )["score"]
                    bleu = bleu_metric.compute(
                        predictions=predictions,
                        references=[[ref] for ref in references],
                    )["score"]
                    key = f"{src_lang}-{tgt_lang}"
                    results[f"eval/chrf_{key}"] = chrf
                    results[f"eval/bleu_{key}"] = bleu
                    chrf_scores.append(chrf)
                    bleu_scores.append(bleu)
                    print(f"eval/chrf_{key}: {chrf:.2f} | eval/bleu_{key}: {bleu:.2f} | n={len(rows)}")

                    for row, prediction, reference in zip(rows.itertuples(index=False), predictions, references):
                        if len(sample_rows) >= 15:
                            break
                        sample_rows.append(
                            [
                                key,
                                getattr(row, "source_text"),
                                reference,
                                prediction,
                            ]
                        )

            if chrf_scores:
                results["eval/chrf_macro"] = float(np.mean(chrf_scores))
                results["eval/bleu_macro"] = float(np.mean(bleu_scores))
                print(
                    f"eval/chrf_macro: {results['eval/chrf_macro']:.2f} | "
                    f"eval/bleu_macro: {results['eval/bleu_macro']:.2f}"
                )

            metrics = kwargs.get("metrics")
            if metrics is not None:
                metrics.update(results)

            report_targets = args.report_to if isinstance(args.report_to, (list, tuple, set)) else [args.report_to]
            if "wandb" in report_targets:
                try:
                    import wandb

                    if wandb.run is not None:
                        log_payload = dict(results)
                        if sample_rows:
                            log_payload["eval/predictions"] = wandb.Table(
                                columns=["direction", "source", "reference", "prediction"],
                                data=sample_rows,
                            )
                        wandb.log(log_payload, step=state.global_step)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to log chrF eval to wandb: %s", exc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Per-direction chrF eval failed; continuing training: %s", exc, exc_info=True)
        finally:
            if old_src_lang is not None:
                self.tokenizer.src_lang = old_src_lang
            if was_training:
                model.train()


def shift_labels_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


class MbaySeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with decoder inputs prepared for M2M/NLLB label smoothing."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_smoother is not None and "labels" in inputs and "decoder_input_ids" not in inputs:
            decoder_start_token_id = model.config.decoder_start_token_id
            pad_token_id = model.config.pad_token_id
            if decoder_start_token_id is None or pad_token_id is None:
                raise ValueError("Model config must define decoder_start_token_id and pad_token_id")
            inputs["decoder_input_ids"] = shift_labels_right(
                inputs["labels"],
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
            )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


@dataclass
class Args:
    model_dir: str = field(metadata={"help": "Path to extended NLLB model dir"})
    data: str = field(metadata={"help": "Path to unified parquet corpus"})
    output_dir: str = field(default="models/nllb-600m-mbay-finetuned")
    epochs: int = field(default=6)
    batch_size: int = field(default=16)
    grad_accum: int = field(default=4)
    lr: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.05)
    max_length: int = field(default=128)
    seed: int = field(default=42)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    eval_steps: int = field(default=500)
    save_steps: int = field(default=500)
    early_stopping_patience: int = field(default=5)
    label_smoothing: float = field(default=0.1)
    chrf_eval_every: int = field(default=1)
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = HfArgumentParser(Args)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()

    set_seed(args.seed)

    # Auto-detect device: CUDA > MPS > CPU
    use_mps = (not torch.cuda.is_available()
               and torch.backends.mps.is_available())
    if not torch.cuda.is_available() and args.bf16:
        # MPS and CPU do not support Trainer bf16 the way CUDA does; use fp32.
        print("CUDA not available: switching bf16→fp32")
        args.bf16 = False

    if use_mps:
        # MPS doesn't support bf16; use fp32 (fp16 has accuracy issues on M-series)
        if args.bf16:
            print("MPS detected: switching bf16→fp32 (MPS lacks bf16 support)")
            args.bf16 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    if use_mps:
        model = model.to("mps")
        print(f"Model on MPS ({torch.backends.mps.is_available()})")

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
    raw_val_df = pd.read_parquet(args.data)
    raw_val_df = raw_val_df[raw_val_df["split"] == "val"].reset_index(drop=True)
    train_ds = load_corpus_as_hf_dataset(args.data, "train")
    val_ds = Dataset.from_pandas(raw_val_df)

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
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=report_to,
        run_name=f"nllb-mbay-{args.epochs}ep",
        predict_with_generate=False,
        group_by_length=True,
        # MPS: Trainer auto-detects mps device when CUDA absent
        use_cpu=False,
    )

    trainer = MbaySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    )
    trainer.add_callback(
        PerDirectionChrfCallback(
            tokenizer=tokenizer,
            raw_val_df=raw_val_df,
            eval_every=args.chrf_eval_every,
        )
    )

    # Resume from checkpoint if one exists (e.g. spot interruption restart).
    ckpt_dirs = sorted(
        Path(args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    resume_from = str(ckpt_dirs[-1]) if ckpt_dirs else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

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
