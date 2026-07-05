"""Validate the full conversion pipeline and calibrate gate thresholds.

Runs the exact convert.py path (chunking, generation, dual gates) on the
held-out test split of the tone-restore pairs — where targets are known —
then sweeps (roundtrip_sim, confidence) thresholds and reports keep-rate vs
kept-set word accuracy. The chosen operating point transfers to the real
bible conversion because inputs are simulated with the same rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mbay_nmt.train_tone_restore.convert import (
    chunk_verse,
    restore_chunks,
    roundtrip_similarity,
)
from mbay_nmt.train_tone_restore.evaluate import word_metrics

DEFAULT_PAIRS = Path("datasets/processed/tone-restore-pairs.parquet")


def run_pipeline(model, tokenizer, inputs: list[str], device: str, batch_size: int) -> pd.DataFrame:
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in inputs:
        chunks = chunk_verse(text)
        spans.append((len(flat), len(flat) + len(chunks)))
        flat.extend(chunks)

    restored_chunks, conf = restore_chunks(model, tokenizer, flat, device, batch_size)
    rows = []
    for (start, end), source in zip(spans, inputs, strict=True):
        restored = "".join(restored_chunks[start:end])
        rows.append(
            {
                "input": source,
                "restored": restored,
                "confidence": min(conf[start:end]),
                "roundtrip_sim": roundtrip_similarity(source, restored),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()

    pairs = pd.read_parquet(args.pairs)
    rows = pairs[pairs["split"].eq(args.split)].reset_index(drop=True)
    result = run_pipeline(model, tokenizer, rows["input"].tolist(), device, args.batch_size)
    result["target"] = rows["target"]

    overall = word_metrics(result["restored"].tolist(), result["target"].tolist())
    print(f"pipeline on {args.split} ({len(result)} rows): {overall}")

    print("\nGate sweep (keep-rate vs kept-set word accuracy):")
    print(f"{'rt_sim>=':>9} {'conf>=':>7} {'keep%':>7} {'acc%':>7} {'words':>7}")
    for rt in (0.0, 0.90, 0.95, 0.98, 1.0):
        for cf in (0.0, 0.90, 0.95):
            kept = result[(result["roundtrip_sim"] >= rt) & (result["confidence"] >= cf)]
            if not len(kept):
                continue
            m = word_metrics(kept["restored"].tolist(), kept["target"].tolist())
            print(
                f"{rt:>9} {cf:>7} {len(kept)/len(result):>7.1%} "
                f"{m['word_accuracy']:>7.1%} {m['words']:>7}"
            )


if __name__ == "__main__":
    main()
