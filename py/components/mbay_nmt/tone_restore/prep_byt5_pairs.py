"""Build tone-restoration pairs for a ByT5 restorer.

Targets = toned Keegan-orthography text (dictionary domains + ASR manifest).
Inputs = simulated 1980-bible orthography via `bible_style.keegan_to_bible`,
plus stochastic variants so the model sees the conventions that vary in the
real bible text. Train/val/test follow the corpus split; ASR-manifest-only
sentences (no split) go to train after dedup against val/test targets.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

from mbay_nmt.tone_restore.bible_style import keegan_to_bible, variants
from mbay_nmt.tone_restore.lexicon_restore import DICT_DOMAINS
from mbay_nmt.tone_restore.skeleton import TONE_MARKS

DEFAULT_CORPUS = Path("datasets/processed/mbay-corpus-v2.parquet")
DEFAULT_ASR_MANIFEST = Path("datasets/processed/asr-manifest.parquet")
DEFAULT_OUTPUT = Path("datasets/processed/tone-restore-pairs.parquet")

_TONE_RE = re.compile("[" + "".join(TONE_MARKS) + "]")

# rows with these letters (or a combining circumflex) are font-decode residue
_ILLEGAL_ROW_RE = re.compile(r"[cfqvxzΓŋ╬̂]", re.IGNORECASE)


def is_clean_target(text: str) -> bool:
    import unicodedata as ud

    return not _ILLEGAL_ROW_RE.search(ud.normalize("NFD", text))


def has_tone(text: str) -> bool:
    import unicodedata as ud

    return _TONE_RE.search(ud.normalize("NFD", text)) is not None


def corpus_targets(corpus_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(corpus_path)
    rows = df[df["source_lang"].eq("mbay") & df["domain"].isin(DICT_DOMAINS)]
    out = rows[["source_text", "split"]].rename(columns={"source_text": "target"})
    return out.dropna().astype({"target": str})


def asr_targets(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        return pd.DataFrame(columns=["target", "split"])
    m = pd.read_parquet(manifest_path)
    rows = m[m["language"].eq("mbay")]["text"].dropna().astype(str)
    return pd.DataFrame({"target": rows, "split": "train"})


def build_pairs(
    targets: pd.DataFrame, n_variants: int = 1, seed: int = 42
) -> pd.DataFrame:
    rng = random.Random(seed)
    # Keegan 1997-era ə == post-2012 ɨ; bible + modern dict use ɨ
    targets = targets.assign(target=targets["target"].str.replace("ə", "ɨ"))
    # dedup: keep first occurrence; corpus rows come first so their split wins
    targets = targets.drop_duplicates(subset="target", keep="first")
    targets = targets[targets["target"].map(has_tone)]
    targets = targets[targets["target"].map(is_clean_target)]

    records: list[dict[str, str]] = []
    for target, split in targets.itertuples(index=False):
        records.append(
            {"input": keegan_to_bible(target), "target": target, "split": split}
        )
        if split == "train" and n_variants:
            for variant in variants(target, rng, n=n_variants):
                records.append({"input": variant, "target": target, "split": split})

    pairs = pd.DataFrame.from_records(records)
    pairs = pairs[pairs["input"].str.strip().astype(bool)]
    return pairs.drop_duplicates().reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--asr-manifest", type=Path, default=DEFAULT_ASR_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    targets = pd.concat(
        [corpus_targets(args.corpus), asr_targets(args.asr_manifest)],
        ignore_index=True,
    )
    pairs = build_pairs(targets, n_variants=args.variants, seed=args.seed)
    pairs.to_parquet(args.output, index=False)
    print(f"Wrote {len(pairs):,} tone-restoration pairs to {args.output}")
    print(pairs["split"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
