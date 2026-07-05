"""Convert the bible corpus to Keegan diacritics with confidence gating.

For each unique bible verse:
- chunk at punctuation (long verses repetition-loop under greedy decode)
- restore each chunk with the ByT5 model, tracking mean token log-prob
- round-trip check: re-simulate bible style from the restoration
  (`keegan_to_bible`) and compare folded skeletons against the input

Output parquet columns: provenance, bible_text, restored_text, confidence
(exp mean logprob), roundtrip_sim (0-1), n_chunks. Downstream consumers gate
on both scores (see notes/MBAY_GRAMMAR.md §4).
"""

from __future__ import annotations

import argparse
import difflib
import re
import unicodedata as ud
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from mbay_nmt.tone_restore.bible_style import keegan_to_bible
from mbay_nmt.tone_restore.skeleton import skeleton

DEFAULT_CORPUS = Path("datasets/processed/mbay-corpus-v2.parquet")
DEFAULT_OUTPUT = Path("datasets/processed/bible-restored.parquet")

CHUNK_SPLIT_RE = re.compile(r"([.:;!?]+\s+|,\s+)")
MAX_CHUNK_CHARS = 120


def chunk_verse(text: str) -> list[str]:
    """Split at punctuation, greedily merging back up to MAX_CHUNK_CHARS."""
    parts = [p for p in CHUNK_SPLIT_RE.split(text) if p]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks or [text]


def fold_key(text: str) -> str:
    sk = skeleton(ud.normalize("NFC", text), hyphens="strip")
    sk = re.sub(r"[^\w]", "", sk, flags=re.UNICODE)
    sk = re.sub(r"ny", "y", sk)
    return re.sub(r"(.)\1+", r"\1", sk)


def roundtrip_similarity(bible_input: str, restored: str) -> float:
    return difflib.SequenceMatcher(
        None, fold_key(bible_input), fold_key(keegan_to_bible(restored))
    ).ratio()


@torch.no_grad()
def restore_chunks(
    model, tokenizer, chunks: list[str], device: str, batch_size: int = 32
) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    confidences: list[float] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=512,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        texts.extend(tokenizer.batch_decode(out.sequences, skip_special_tokens=True))
        scores = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )
        mask = out.sequences[:, 1:] != tokenizer.pad_token_id
        mean_logprob = (scores * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        confidences.extend(mean_logprob.exp().tolist())
    return texts, confidences


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
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

    df = pd.read_parquet(args.corpus)
    bible = df[df["source_lang"].eq("mbay") & df["domain"].eq("bible")]
    verses = (
        bible[["provenance", "source_text"]]
        .drop_duplicates(subset="source_text")
        .reset_index(drop=True)
    )
    if args.limit:
        verses = verses.head(args.limit)
    print(f"{len(verses):,} unique verses on {device}")

    records: list[dict] = []
    flat_chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in verses["source_text"].astype(str):
        chunks = chunk_verse(text)
        spans.append((len(flat_chunks), len(flat_chunks) + len(chunks)))
        flat_chunks.extend(chunks)

    restored_chunks, chunk_conf = restore_chunks(
        model, tokenizer, flat_chunks, device, batch_size=args.batch_size
    )

    for (start, end), (_, row) in zip(spans, verses.iterrows(), strict=True):
        restored = "".join(restored_chunks[start:end])
        conf = min(chunk_conf[start:end])
        records.append(
            {
                "provenance": row["provenance"],
                "bible_text": row["source_text"],
                "restored_text": restored,
                "confidence": conf,
                "roundtrip_sim": roundtrip_similarity(row["source_text"], restored),
                "n_chunks": end - start,
            }
        )
        if len(records) % 2000 == 0:
            print(f"  {len(records):,} verses done")

    out = pd.DataFrame.from_records(records)
    out.to_parquet(args.output, index=False)
    print(f"Wrote {len(out):,} rows to {args.output}")
    print(out[["confidence", "roundtrip_sim"]].describe().to_string())


if __name__ == "__main__":
    main()
