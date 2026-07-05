"""Fine-tune ByT5 to restore Keegan diacritics from bible-style Mbay text.

Pairs come from `mbay_nmt.tone_restore.prep_byt5_pairs` (input = simulated
1980-bible orthography, target = toned Keegan orthography). Byte-level ByT5
needs no tokenizer work and is robust to the orthography's combining marks.
Checkpoint selection follows eval_loss, matching the NMT runs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    set_seed,
)

DEFAULT_MODEL_ID = "google/byt5-small"
DEFAULT_PAIRS = Path("datasets/processed/tone-restore-pairs.parquet")
MAX_LEN = 512  # bytes; covers >99% of dictionary sentences


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_splits(pairs_path: Path, limit: int | None = None) -> dict[str, Dataset]:
    df = pd.read_parquet(pairs_path)
    splits: dict[str, Dataset] = {}
    for split in ("train", "val", "test"):
        rows = df[df["split"].eq(split)][["input", "target"]]
        if limit is not None:
            rows = rows.head(limit)
        splits[split] = Dataset.from_pandas(rows.reset_index(drop=True))
    return splits


def tokenize_fn(tokenizer, max_len: int = MAX_LEN):
    def fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        model_inputs = tokenizer(batch["input"], max_length=max_len, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return fn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=Path("models/tone-restore-byt5"))
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap rows per split (smoke test)")
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--wandb", action="store_true", help="log to WandB (project mbay-nmt)")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    splits = load_splits(args.pairs, limit=args.limit)
    tokenized = {
        name: ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=["input", "target"])
        for name, ds in splits.items()
    }
    print({name: len(ds) for name, ds in tokenized.items()})

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to=["wandb"] if (args.wandb or os.environ.get("WANDB_API_KEY")) else [],
        run_name="tone-restore-byt5",
        seed=args.seed,
        use_cpu=device == "cpu",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
    tokenizer.save_pretrained(str(args.output_dir / "best"))

    metrics = trainer.evaluate(tokenized["val"])
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
