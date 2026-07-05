"""Build the v3 Mbay NMT corpus from the verified v2 parquet.

Usage:
    uv run python -m mbay_nmt.data_sources.build_v3 \
        --input datasets/processed/mbay-corpus-v2.parquet \
        --output datasets/processed/mbay-corpus-v3.parquet
"""

from __future__ import annotations

import argparse
import itertools
import re
import unicodedata
from pathlib import Path

import pandas as pd

DICTIONARY_DOMAINS = {"example", "lexicon", "expression"}
BIBLE_PROVENANCE_RE = re.compile(r"^bible:MlL:([^.]+)\.\d+\.\d+$")
V31_OT_NARRATIVE_BOOKS = {
    "GEN",
    "EXO",
    "JOS",
    "JDG",
    "RUT",
    "1SA",
    "2SA",
    "1KI",
    "2KI",
    "1CH",
    "2CH",
    "EZR",
    "NEH",
    "EST",
}
V31_NT_EPISTLE_BOOKS = {
    "ROM",
    "1CO",
    "2CO",
    "GAL",
    "EPH",
    "PHP",
    "PHI",
    "COL",
    "1TH",
    "2TH",
    "1TI",
    "2TI",
    "TIT",
    "PHM",
    "HEB",
    "JAS",
    "JAM",
    "1PE",
    "2PE",
    "1JN",
    "2JN",
    "3JN",
    "JUD",
}
BIBLE_REF_RE = re.compile(
    r"(?:[1-3]\s?)?[A-Z][a-zà-ü]+\.?\s\d+[:.]\d+(?:\s*[-–]\s*\d+)?\s*;?",
    flags=re.UNICODE,
)
VERSE_REF_RE = re.compile(r"\b\d+:\d+(?:[-–]\d+)?\b")
SHORT_ABBREV = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ]{0,3}\.?"
REF_MARKER = r"(?:(?:[Ss]ee\s+)?[Cc]h\.|[Vv]oir|#)"
REF_CORE = rf"(?:(?:{REF_MARKER})\s*)?(?:{SHORT_ABBREV}\s+)?\d+:\d+(?:[-–]\d+)?"
MARKED_REF_CORE = rf"(?:{REF_MARKER})\s*(?:{SHORT_ABBREV}\s+)?\d+:\d+(?:[-–]\d+)?"
MARKED_REF_CLUSTER_RE = re.compile(
    rf"(?<!\w)(?:\d+\s+)*(?:{MARKED_REF_CORE})(?:\s+(?:{REF_CORE}|\d+))*",
    flags=re.UNICODE,
)
REF_CLUSTER_RE = re.compile(
    rf"(?<!\w)(?:\d+\s+)*(?:{REF_CORE})(?:\s+(?:{REF_CORE}|\d+))*",
    flags=re.UNICODE,
)
BRACKETED_RE = re.compile(r"\[[^\]]*\]")
ORPHAN_BRACKET_RE = re.compile(r"[\[\]]")
PUNCT_TOKEN_RE = re.compile(r"(?:^|\s)[;:,.]+\s*(?=$|\s)")


def nfc_strip(text: object) -> str:
    """NFC-normalize string-ish text and strip outer whitespace."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).strip()


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def clean_bible_target(text: object) -> str:
    """Remove inline bible cross references and punctuation left behind."""
    cleaned = nfc_strip(text)
    cleaned = BRACKETED_RE.sub(" ", cleaned)
    cleaned = MARKED_REF_CLUSTER_RE.sub(" ", cleaned)
    cleaned = BIBLE_REF_RE.sub(" ", cleaned)
    cleaned = REF_CLUSTER_RE.sub(" ", cleaned)
    cleaned = VERSE_REF_RE.sub(" ", cleaned)
    cleaned = ORPHAN_BRACKET_RE.sub(" ", cleaned)
    cleaned = collapse_whitespace(cleaned)
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = PUNCT_TOKEN_RE.sub(" ", cleaned)
        cleaned = collapse_whitespace(cleaned)
    return cleaned.strip(" ;:,")


def strip_combining_marks(text: str) -> str:
    """Remove tone/diacritic combining marks while preserving base letters."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def bible_book_from_provenance(provenance: object) -> str | None:
    if not isinstance(provenance, str):
        return None
    match = BIBLE_PROVENANCE_RE.match(provenance)
    return match.group(1) if match else None


def _bible_book_counts(df: pd.DataFrame) -> pd.Series:
    bible_books = df.loc[df["domain"] == "bible", "provenance"].map(bible_book_from_provenance).dropna()
    return bible_books.value_counts().sort_index()


def _choose_v31_book_pair(book_counts: pd.Series, *, excluded: set[str]) -> tuple[str, str]:
    available = [book for book in book_counts.index if book not in excluded]
    if len(available) < 2:
        raise ValueError("Need at least two bible books to select a v3.1 holdout split")

    mixed_genre_pairs = [
        pair
        for pair in itertools.combinations(available, 2)
        if (
            (pair[0] in V31_OT_NARRATIVE_BOOKS and pair[1] in V31_NT_EPISTLE_BOOKS)
            or (pair[1] in V31_OT_NARRATIVE_BOOKS and pair[0] in V31_NT_EPISTLE_BOOKS)
        )
    ]
    candidate_pairs = mixed_genre_pairs or list(itertools.combinations(available, 2))
    total = int(book_counts.sum())
    target = total * 0.04
    lower = total * 0.03
    upper = total * 0.05

    def objective(pair: tuple[str, str]) -> tuple[float, float, tuple[str, str]]:
        rows = int(book_counts.loc[list(pair)].sum())
        outside_band = 0.0 if lower <= rows <= upper else min(abs(rows - lower), abs(rows - upper))
        return (outside_band, abs(rows - target), tuple(sorted(pair)))

    return tuple(sorted(min(candidate_pairs, key=objective)))


def choose_v31_bible_holdout_books(df: pd.DataFrame) -> tuple[tuple[str, str], tuple[str, str], pd.Series]:
    book_counts = _bible_book_counts(df)
    val_books = _choose_v31_book_pair(book_counts, excluded=set())
    test_books = _choose_v31_book_pair(book_counts, excluded=set(val_books))
    return val_books, test_books, book_counts


def reassign_v31_bible_splits(
    df: pd.DataFrame,
    *,
    val_books: tuple[str, str] | None = None,
    test_books: tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = df.copy()
    bible_book = out["provenance"].map(bible_book_from_provenance)
    if val_books is None or test_books is None:
        selected_val_books, selected_test_books, book_counts = choose_v31_bible_holdout_books(out)
        val_books = selected_val_books if val_books is None else val_books
        test_books = selected_test_books if test_books is None else test_books
    else:
        book_counts = _bible_book_counts(out)

    overlap = set(val_books) & set(test_books)
    if overlap:
        raise ValueError(f"v3.1 bible val/test books overlap: {sorted(overlap)}")

    bible_mask = out["domain"] == "bible"
    out.loc[bible_mask, "split"] = "train"
    out.loc[bible_mask & bible_book.isin(val_books), "split"] = "val"
    out.loc[bible_mask & bible_book.isin(test_books), "split"] = "test"

    val_rows = int(bible_book.isin(val_books).sum())
    test_rows = int(bible_book.isin(test_books).sum())
    info = {
        "val_books": tuple(val_books),
        "test_books": tuple(test_books),
        "val_rows": val_rows,
        "test_rows": test_rows,
        "book_counts": {book: int(book_counts.loc[book]) for book in book_counts.index},
    }
    return out, info


def apply_v31_source_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bible_mask = out["domain"] == "bible"
    untagged_bible_mask = bible_mask & ~out["source_text"].str.startswith("<bible> ", na=False)
    out.loc[untagged_bible_mask, "source_text"] = "<bible> " + out.loc[untagged_bible_mask, "source_text"]
    return out


def _downsample_v31_lexicon_train(df: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, int]:
    lexicon_train_mask = (df["split"] == "train") & (df["domain"] == "lexicon")
    lexicon_train = df.loc[lexicon_train_mask]
    if lexicon_train.empty:
        return df, 0
    kept_indices = set(lexicon_train.sample(frac=0.5, random_state=seed).index)
    drop_mask = lexicon_train_mask & ~df.index.isin(kept_indices)
    dropped = int(drop_mask.sum())
    return df.loc[~drop_mask].reset_index(drop=True), dropped


def _format_book_counts(books: tuple[str, str], book_counts: dict[str, int]) -> str:
    return ", ".join(f"{book}={book_counts.get(book, 0)}" for book in books)


def _stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["direction", "domain", "split", "rows"])
    stats_df = df.assign(direction=df["source_lang"] + "->" + df["target_lang"])
    return (
        stats_df.groupby(["direction", "domain", "split"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["direction", "domain", "split"])
    )


def _print_stats_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    stats = _stats(df)
    if stats.empty:
        print("(no rows)")
        return
    print(stats.to_string(index=False))


def _dedupe_prefer_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key_cols = ["source_lang", "target_lang", "source_text", "target_text"]
    work = df.copy()
    work["_orig_order"] = range(len(work))
    work["_split_priority"] = (work["split"] == "train").astype(int)
    work = work.sort_values(key_cols + ["_split_priority", "_orig_order"], kind="mergesort")
    deduped = work.drop_duplicates(key_cols, keep="first")
    dropped = len(work) - len(deduped)
    deduped = deduped.sort_values("_orig_order", kind="mergesort").drop(
        columns=["_orig_order", "_split_priority"]
    )
    return deduped.reset_index(drop=True), dropped


def _assert_no_train_eval_leakage(df: pd.DataFrame) -> None:
    train_pairs = set(
        map(tuple, df.loc[df["split"] == "train", ["source_text", "target_text"]].itertuples(index=False, name=None))
    )
    eval_pairs = set(
        map(tuple, df.loc[df["split"].isin(["val", "test"]), ["source_text", "target_text"]].itertuples(index=False, name=None))
    )
    leaked = train_pairs & eval_pairs
    if leaked:
        examples = list(leaked)[:5]
        raise AssertionError(
            f"{len(leaked)} val/test source-target pairs also appear in train; examples={examples!r}"
        )


def _drop_train_eval_pair_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    eval_pairs = set(
        map(tuple, df.loc[df["split"].isin(["val", "test"]), ["source_text", "target_text"]].itertuples(index=False, name=None))
    )
    train_pair_mask = df["split"].eq("train") & df[["source_text", "target_text"]].apply(tuple, axis=1).isin(eval_pairs)
    dropped = int(train_pair_mask.sum())
    if dropped:
        df = df.loc[~train_pair_mask].reset_index(drop=True)
    return df, dropped


def build_v3(
    input_path: str | Path,
    *,
    no_notone_aug: bool = False,
    v31: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_parquet(input_path)
    required_cols = {
        "source_lang",
        "target_lang",
        "source_text",
        "target_text",
        "domain",
        "provenance",
        "split",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input corpus is missing columns: {sorted(missing)}")

    df = df.copy()
    counts: dict[str, int] = {
        "dropped_bible_short_target": 0,
        "dropped_bible_remaining_ref": 0,
        "dropped_bible_length_ratio": 0,
        "dropped_empty_after_hygiene": 0,
        "dropped_exact_dupes": 0,
        "dropped_train_eval_pair_leakage": 0,
        "added_flipped_dictionary": 0,
        "added_notone_train": 0,
    }

    bible_mask = df["domain"] == "bible"
    df.loc[bible_mask, "target_text"] = df.loc[bible_mask, "target_text"].map(clean_bible_target)
    remaining_ref_mask = bible_mask & df["target_text"].str.contains(VERSE_REF_RE, na=False)
    counts["dropped_bible_remaining_ref"] = int(remaining_ref_mask.sum())
    df = df.loc[~remaining_ref_mask].reset_index(drop=True)

    bible_mask = df["domain"] == "bible"
    short_bible_mask = bible_mask & (df["target_text"].str.len() < 2)
    counts["dropped_bible_short_target"] = int(short_bible_mask.sum())
    df = df.loc[~short_bible_mask].reset_index(drop=True)

    bible_mask = df["domain"] == "bible"
    source_len = df["source_text"].map(lambda value: len(nfc_strip(value)))
    target_len = df["target_text"].map(lambda value: len(nfc_strip(value)))
    ratio = source_len / target_len.clip(lower=1)
    bad_ratio_mask = bible_mask & ((ratio < 0.25) | (ratio > 4.0))
    counts["dropped_bible_length_ratio"] = int(bad_ratio_mask.sum())
    df = df.loc[~bad_ratio_mask].reset_index(drop=True)

    df["source_text"] = df["source_text"].map(nfc_strip)
    df["target_text"] = df["target_text"].map(nfc_strip)
    empty_mask = (df["source_text"].str.len() == 0) | (df["target_text"].str.len() == 0)
    counts["dropped_empty_after_hygiene"] = int(empty_mask.sum())
    df = df.loc[~empty_mask].reset_index(drop=True)

    # Dictionary export uses ____ for the omitted headword in example rows;
    # unrepaired occurrences would teach the model to emit literal underscores.
    placeholder_mask = df["source_text"].str.contains("____", regex=False) | df[
        "target_text"
    ].str.contains("____", regex=False)
    counts["dropped_headword_placeholder"] = int(placeholder_mask.sum())
    df = df.loc[~placeholder_mask].reset_index(drop=True)

    df, counts["dropped_exact_dupes"] = _dedupe_prefer_eval(df)

    if v31:
        df, holdout_info = reassign_v31_bible_splits(df)
        counts["dropped_v31_lexicon_train_downsample"] = 0
        counts["v31_bible_val_rows"] = int(holdout_info["val_rows"])
        counts["v31_bible_test_rows"] = int(holdout_info["test_rows"])
        print(
            "v3.1 bible holdout val books: "
            f"{_format_book_counts(holdout_info['val_books'], holdout_info['book_counts'])} "
            f"(rows={holdout_info['val_rows']})"
        )
        print(
            "v3.1 bible holdout test books: "
            f"{_format_book_counts(holdout_info['test_books'], holdout_info['book_counts'])} "
            f"(rows={holdout_info['test_rows']})"
        )
        df, counts["dropped_v31_lexicon_train_downsample"] = _downsample_v31_lexicon_train(df, seed=seed)

    flip_mask = (df["source_lang"] == "mbay") & df["domain"].isin(DICTIONARY_DOMAINS)
    flipped = df.loc[flip_mask].copy()
    if not flipped.empty:
        flipped[["source_lang", "target_lang"]] = flipped[["target_lang", "source_lang"]]
        flipped[["source_text", "target_text"]] = flipped[["target_text", "source_text"]]
        flipped["provenance"] = "flip:" + flipped["provenance"].astype(str)
        counts["added_flipped_dictionary"] = len(flipped)
        df = pd.concat([df, flipped], ignore_index=True)

    if not no_notone_aug:
        notone_mask = (
            (df["split"] == "train")
            & (df["source_lang"] == "mbay")
            & df["domain"].isin(DICTIONARY_DOMAINS)
        )
        notone = df.loc[notone_mask].copy()
        if not notone.empty:
            notone["source_text"] = notone["source_text"].map(strip_combining_marks)
            notone = notone[notone["source_text"] != df.loc[notone_mask, "source_text"].to_numpy()]
            if not notone.empty:
                notone["provenance"] = "notone:" + notone["provenance"].astype(str)
                counts["added_notone_train"] = len(notone)
                df = pd.concat([df, notone], ignore_index=True)

    if v31:
        df = apply_v31_source_tags(df)

    df, counts["dropped_train_eval_pair_leakage"] = _drop_train_eval_pair_leakage(df)
    _assert_no_train_eval_leakage(df)
    return df.reset_index(drop=True), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input v2 parquet path")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--no-notone-aug", action="store_true", help="Disable tone-stripped train augmentation")
    parser.add_argument("--v31", action="store_true", help="Build corpus v3.1 with book-level bible holdout and tags")
    parser.add_argument("--seed", type=int, default=42, help="Seed for v3.1 deterministic sampling")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(
        args.output
        or ("datasets/processed/mbay-corpus-v3.1.parquet" if args.v31 else "datasets/processed/mbay-corpus-v3.parquet")
    )

    before = pd.read_parquet(input_path)
    _print_stats_table("Rows before", before)

    after, counts = build_v3(input_path, no_notone_aug=args.no_notone_aug, v31=args.v31, seed=args.seed)

    print("\nFilter/addition counts")
    for name, count in counts.items():
        print(f"{name}: {count}")

    _print_stats_table("Rows after", after)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    after.to_parquet(output_path, index=False)
    print(f"\nSaved {len(after):,} rows to {output_path}")


if __name__ == "__main__":
    main()
