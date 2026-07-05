"""Build tone-restoration pairs for a future ByT5 restorer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mbay_nmt.tone_restore.lexicon_restore import DEFAULT_CORPUS, DICT_DOMAINS
from mbay_nmt.tone_restore.skeleton import dict_to_bible_style, skeleton

DEFAULT_OUTPUT = Path("datasets/processed/tone-restore-pairs.parquet")


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[
        df["source_lang"].eq("mbay")
        & df["domain"].isin(DICT_DOMAINS)
        & df["split"].eq("train")
    ].copy()
    rows["target"] = rows["source_text"].astype(str)
    rows["input"] = rows["target"].map(dict_to_bible_style)
    rows = rows[rows.apply(lambda row: skeleton(row["input"]) == skeleton(row["target"]), axis=1)]
    return rows[["input", "target", "split"]].drop_duplicates().reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.corpus)
    pairs = build_pairs(df)
    pairs.to_parquet(args.output, index=False)
    print(f"Wrote {len(pairs):,} tone-restoration pairs to {args.output}")
    print(pairs["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

