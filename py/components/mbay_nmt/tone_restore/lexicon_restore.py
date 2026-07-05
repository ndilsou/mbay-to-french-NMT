"""Lexicon baseline for restoring dictionary-style Mbay tone marks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import unicodedata as ud

import pandas as pd

from mbay_nmt.tone_restore.skeleton import dict_to_bible_style, skeleton

DICT_DOMAINS = {"example", "lexicon", "expression"}
DEFAULT_CORPUS = Path("datasets/processed/mbay-corpus-v2.parquet")
DEFAULT_SAMPLE = Path("datasets/processed/tone-restore-bible-sample.parquet")


@dataclass(frozen=True)
class LexiconEntry:
    forms: Counter[str]

    @property
    def total(self) -> int:
        return self.forms.total()

    @property
    def most_common_form(self) -> str:
        return self.forms.most_common(1)[0][0]

    @property
    def is_unambiguous(self) -> bool:
        return len(self.forms) == 1

    @property
    def dominant_ratio(self) -> float:
        return self.forms.most_common(1)[0][1] / self.total

    @property
    def is_restorable(self) -> bool:
        return self.is_unambiguous or self.dominant_ratio > 0.80


class LexiconRestorer:
    def __init__(self, entries: dict[str, LexiconEntry]):
        self.entries = entries

    @classmethod
    def from_texts(cls, texts: list[str] | pd.Series) -> "LexiconRestorer":
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for text in texts:
            for token in lexical_tokens(str(text)):
                key = skeleton(token)
                if key:
                    counts[key][token] += 1
        return cls({key: LexiconEntry(forms) for key, forms in counts.items()})

    @classmethod
    def from_corpus(cls, corpus_path: Path = DEFAULT_CORPUS) -> "LexiconRestorer":
        df = pd.read_parquet(corpus_path)
        train = dictionary_mbay_rows(df)
        train = train[train["split"].eq("train")]
        return cls.from_texts(train["source_text"])

    def restore_token(self, token: str) -> str:
        key = skeleton(token)
        entry = self.entries.get(key)
        if entry is None or not entry.is_restorable:
            return token
        return preserve_first_char_case(entry.most_common_form, token)

    def restore(self, text: str) -> str:
        parts: list[str] = []
        current: list[str] = []
        current_is_word: bool | None = None

        for ch in text:
            is_word = is_lexical_char(ch)
            if current and is_word != current_is_word:
                part = "".join(current)
                parts.append(self.restore_token(part) if current_is_word else part)
                current = []
            current.append(ch)
            current_is_word = is_word

        if current:
            part = "".join(current)
            parts.append(self.restore_token(part) if current_is_word else part)
        return "".join(parts)


def is_lexical_char(ch: str) -> bool:
    category = ud.category(ch)
    return category[0] in {"L", "M"}


def lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for ch in text:
        if is_lexical_char(ch):
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def preserve_first_char_case(restored: str, original: str) -> str:
    if not original or not original[0].isupper():
        return restored
    for index, ch in enumerate(restored):
        if ud.category(ch).startswith("L"):
            return restored[:index] + ch.upper() + restored[index + 1 :]
    return restored


def dictionary_mbay_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["source_lang"].eq("mbay") & df["domain"].isin(DICT_DOMAINS)].copy()


def self_test(df: pd.DataFrame, restorer: LexiconRestorer) -> dict[str, dict[str, float | int]]:
    val = dictionary_mbay_rows(df)
    val = val[val["split"].eq("val")]

    buckets: dict[str, Counter[str]] = {
        "overall": Counter(),
        "in-lexicon unambiguous": Counter(),
        "in-lexicon ambiguous": Counter(),
        "OOV": Counter(),
    }

    for target_text in val["source_text"].dropna().astype(str):
        input_text = dict_to_bible_style(target_text)
        restored_text = restorer.restore(input_text)

        input_tokens = lexical_tokens(input_text)
        restored_tokens = lexical_tokens(restored_text)
        target_tokens = lexical_tokens(target_text)

        for input_token, restored_token, target_token in zip(input_tokens, restored_tokens, target_tokens, strict=False):
            key = skeleton(input_token)
            entry = restorer.entries.get(key)
            if entry is None:
                bucket = "OOV"
            elif entry.is_unambiguous:
                bucket = "in-lexicon unambiguous"
            else:
                bucket = "in-lexicon ambiguous"

            is_correct = restored_token == target_token
            buckets["overall"]["total"] += 1
            buckets[bucket]["total"] += 1
            if is_correct:
                buckets["overall"]["correct"] += 1
                buckets[bucket]["correct"] += 1

    return {name: summarize_accuracy(counts) for name, counts in buckets.items()}


def summarize_accuracy(counts: Counter[str]) -> dict[str, float | int]:
    total = counts["total"]
    correct = counts["correct"]
    accuracy = correct / total if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def print_self_test(summary: dict[str, dict[str, float | int]]) -> None:
    print("Lexicon restoration self-test on dictionary validation split")
    for bucket, stats in summary.items():
        print(
            f"- {bucket}: {stats['accuracy']:.2%} "
            f"({stats['correct']}/{stats['total']} lexical tokens)"
        )


def write_bible_sample(df: pd.DataFrame, restorer: LexiconRestorer, output_path: Path) -> pd.DataFrame:
    bible = df[df["source_lang"].eq("mbay") & df["domain"].eq("bible")]
    sample = bible.sample(n=min(200, len(bible)), random_state=42).copy()
    sample["before"] = sample["source_text"]
    sample["after"] = sample["before"].map(restorer.restore)
    sample[["split", "provenance", "before", "after"]].to_parquet(output_path, index=False)
    return sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.corpus)
    train = dictionary_mbay_rows(df)
    train = train[train["split"].eq("train")]
    restorer = LexiconRestorer.from_texts(train["source_text"])

    print(f"Built train lexicon with {len(restorer.entries):,} skeleton keys")
    print_self_test(self_test(df, restorer))

    sample = write_bible_sample(df, restorer, args.sample_output)
    print(f"\nWrote bible sample: {args.sample_output}")
    print("\nSample restorations")
    for row in sample[["before", "after"]].head(10).itertuples(index=False):
        print(f"- before: {row.before}")
        print(f"  after:  {row.after}")


if __name__ == "__main__":
    main()

