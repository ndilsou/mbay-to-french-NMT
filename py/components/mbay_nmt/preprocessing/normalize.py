"""Merge all parallel-corpus sources, NFC normalize, deduplicate, and assign
train/val/test splits.

Input: parquet files from each source (keegan sister-langs, mbay PG/CSV, bible if accepted).
Output: `mbay-corpus-v1.parquet` — the unified training corpus.

Dedup: by (source_lang, target_lang, normalized source_text, normalized target_text).
Split: stratified by (source_lang, domain) into 90/5/5 train/val/test.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _nfc(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", " ".join(text.split()))


def merge_and_normalize(parquet_paths: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_parquet(p) for p in parquet_paths if p.exists()]
    if not dfs:
        raise ValueError("No parquet files found")
    df = pd.concat(dfs, ignore_index=True)

    df["source_text"] = df["source_text"].apply(_nfc)
    df["target_text"] = df["target_text"].apply(_nfc)

    # Drop rows with empty text
    df = df[(df["source_text"].str.len() > 0) & (df["target_text"].str.len() > 0)].copy()

    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(
        subset=["source_lang", "target_lang", "source_text", "target_text"],
        keep="first",
    ).copy()
    after = len(df)
    if before != after:
        print(f"Dedup: {before:,} -> {after:,} ({before - after:,} removed)")
    return df


def assign_splits(
    df: pd.DataFrame,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified split by (source_lang, domain). Falls back to random for tiny strata."""
    df = df.copy()
    df["split"] = ""
    df["_strata"] = df["source_lang"] + ":" + df["domain"]

    # Strata with fewer than 10 rows go into train directly (can't stratify).
    small_mask = df.groupby("_strata")["_strata"].transform("count") < 10
    df.loc[small_mask, "split"] = "train"

    rest = df[df["split"] != "train"].copy()
    if len(rest) == 0:
        df.drop(columns="_strata", inplace=True)
        return df

    hold_frac = val_frac + test_frac
    train_idx, hold_idx = train_test_split(
        rest.index, test_size=hold_frac, random_state=seed, stratify=rest["_strata"]
    )
    val_idx, test_idx = train_test_split(
        hold_idx,
        test_size=test_frac / hold_frac,
        random_state=seed,
        stratify=rest.loc[hold_idx, "_strata"],
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    df.drop(columns="_strata", inplace=True)
    return df


def build_corpus(
    parquet_paths: list[Path],
    output_path: Path,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    df = merge_and_normalize(parquet_paths)
    df = dedup(df)
    df = assign_splits(df, val_frac=val_frac, test_frac=test_frac, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path, help="Parquet files to merge")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = build_corpus(args.inputs, args.output, args.val_frac, args.test_frac, args.seed)

    print(f"\nWrote {len(df):,} rows to {args.output}")
    print(f"\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nBy source_lang:")
    print(df.groupby("source_lang").size().to_string())
    print(f"\nBy source_lang + domain + split:")
    print(df.groupby(["source_lang", "domain", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
