"""Evaluate a trained tone restorer: word accuracy + diacritic error rate.

Also restores a sample of real bible verses for eyeballing (no gold labels
exist for those yet; see notes/MBAY_GRAMMAR.md §4 on the gold eval set).
"""

from __future__ import annotations

import argparse
import unicodedata as ud
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mbay_nmt.tone_restore.lexicon_restore import lexical_tokens
from mbay_nmt.tone_restore.skeleton import skeleton

DEFAULT_PAIRS = Path("datasets/processed/tone-restore-pairs.parquet")
DEFAULT_CORPUS = Path("datasets/processed/mbay-corpus-v2.parquet")


def restore_batch(model, tokenizer, texts: list[str], device: str, max_len: int = 512) -> list[str]:
    out: list[str] = []
    for i in range(0, len(texts), 16):
        chunk = texts[i : i + 16]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_len, num_beams=1)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def word_metrics(predictions: list[str], targets: list[str]) -> dict[str, float]:
    total = correct = skeleton_ok = 0
    for pred, target in zip(predictions, targets, strict=True):
        pred_tokens = lexical_tokens(ud.normalize("NFC", pred))
        target_tokens = lexical_tokens(ud.normalize("NFC", target))
        for pt, tt in zip(pred_tokens, target_tokens, strict=False):
            total += 1
            correct += pt == tt
            skeleton_ok += skeleton(pt) == skeleton(tt)
    return {
        "word_accuracy": correct / total if total else 0.0,
        "skeleton_preserved": skeleton_ok / total if total else 0.0,
        "words": total,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bible-sample", type=int, default=20)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()

    pairs = pd.read_parquet(args.pairs)
    rows = pairs[pairs["split"].eq(args.split)]
    if args.limit:
        rows = rows.head(args.limit)
    predictions = restore_batch(model, tokenizer, rows["input"].tolist(), device)
    metrics = word_metrics(predictions, rows["target"].tolist())
    print(f"{args.split} ({len(rows)} rows): {metrics}")

    if args.bible_sample:
        df = pd.read_parquet(args.corpus)
        bible = df[df["source_lang"].eq("mbay") & df["domain"].eq("bible")]
        sample = bible.sample(n=args.bible_sample, random_state=7)["source_text"].astype(str).tolist()
        restored = restore_batch(model, tokenizer, sample, device)
        print("\nReal bible restorations (no gold):")
        for before, after in zip(sample, restored, strict=True):
            print(f"- in:  {before}")
            print(f"  out: {after}")


if __name__ == "__main__":
    main()
