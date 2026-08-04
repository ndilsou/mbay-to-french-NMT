"""Build the v3 Mbay NMT corpus from the verified v2 parquet.

Usage:
    uv run python -m mbay_nmt.data_sources.build_v3 \
        --input datasets/processed/mbay-corpus-v2.parquet \
        --output datasets/processed/mbay-corpus-v3.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import string
import unicodedata
from pathlib import Path

import pandas as pd

DICTIONARY_DOMAINS = {"example", "lexicon", "expression"}
BIBLE_PROVENANCE_RE = re.compile(r"^bible:MlL:([^.]+)\.\d+\.\d+$")
RESTORED_BIBLE_COLUMNS = {
    "provenance",
    "bible_text",
    "restored_text",
    "confidence",
    "roundtrip_sim",
    "n_chunks",
    "second_text",
    "agreement",
    "word_agreement",
}
CORPUS_COLUMNS = [
    "source_lang",
    "target_lang",
    "source_text",
    "target_text",
    "domain",
    "provenance",
    "split",
]
V32_SOURCE_ROUNDTRIP_MIN = 0.90
V32_FLIP_WORD_AGREEMENT_MIN = 0.90
V32_FLIP_ROUNDTRIP_MIN = 0.95
V32_BASIC_ALLOWED_CHARS = set(string.digits + string.punctuation + string.whitespace)
V34_FLIP_WORD_AGREEMENT_MIN = 0.75
V35_LEX_TAG = "<lex> "
V35_BT_TAG = "<bt> "
V35_COPY_TAG = "<copy> "
V35_HINT_MARKER = " <hint> "
V35_HINT_PROB_NUM = 128  # out of 256: p=0.5 on first digest byte
V35_HINT_MAX = 2
V35_BT_MIN_SOURCE_WORDS = 3
V35_BT_MIN_UNIQUE_SOURCE_WORDS = 3
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
BIBLE_APPARATUS_RE = re.compile(r"\b(?:for\s+ver\.?|see\s+ver\.?|see\s+ch\.?)\b", flags=re.IGNORECASE)
BIBLE_INITIAL_OR_PUNCT_INT_RUN_RE = re.compile(r"(?:^|[.!?;:,]\s*)\b\d+\s+\d+\b")
EXAMPLE_LIT_SUFFIX_RE = re.compile(r"\s*\[[Ll]it:? [^\]]*\]\s*$")
EXAMPLE_EQUALS_SUFFIX_RE = re.compile(r"\s*\(=[^)]*\)\s*$")
REPLAY_URL_RE = re.compile(r"http", flags=re.IGNORECASE)
WORD_TOKEN_RE = re.compile(r"\S+")
SENTENCE_FINAL_RE = re.compile(r"[.!?…][\"')\]]*$")


PUNCT_NORM = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'",  # curly/modifier apostrophes -> ASCII
    "“": '"', "”": '"',
    "«": '"', "»": '"',
    "–": "-", "—": "-",
    " ": " ",
})


def nfc_strip(text: object) -> str:
    """NFC-normalize string-ish text, fold typographic punctuation, strip whitespace.

    The NLLB tokenizer maps U+2019 (curly apostrophe) to <unk>; the French
    bible uses it everywhere, which silently deleted apostrophes from training
    labels. Fold to ASCII before tokenization ever sees the text.
    """
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text).translate(PUNCT_NORM).strip()


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


def bible_target_is_mostly_digits_punct(text: object) -> bool:
    target = nfc_strip(text)
    if len(target) >= 15 or not target:
        return False
    digit_punct_space = sum(ch.isdigit() or ch in string.punctuation or ch.isspace() for ch in target)
    return digit_punct_space / len(target) >= 0.6


def bible_target_lacks_final_punctuation(text: object) -> bool:
    target = nfc_strip(text).rstrip()
    if not target:
        return False
    return SENTENCE_FINAL_RE.search(target) is None


def clean_example_expression_target(text: object) -> str:
    """Strip trailing dictionary gloss suffixes from example/expression targets."""
    cleaned = nfc_strip(text)
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = EXAMPLE_LIT_SUFFIX_RE.sub("", cleaned)
        cleaned = EXAMPLE_EQUALS_SUFFIX_RE.sub("", cleaned)
        cleaned = cleaned.strip()
    return cleaned


def _tokenize_for_repetition(text: object) -> list[str]:
    tokens = []
    for token in WORD_TOKEN_RE.findall(nfc_strip(text).lower()):
        stripped = token.strip(string.punctuation)
        if stripped:
            tokens.append(stripped)
    return tokens


def restored_text_has_repetition_failure(text: object) -> bool:
    tokens = _tokenize_for_repetition(text)
    if len(tokens) > 6:
        max_token_count = max(tokens.count(token) for token in set(tokens))
        if max_token_count / len(tokens) > 0.4:
            return True
    if len(tokens) < 6:
        return False
    four_gram_counts: dict[tuple[str, ...], int] = {}
    for index in range(len(tokens) - 3):
        gram = tuple(tokens[index : index + 4])
        four_gram_counts[gram] = four_gram_counts.get(gram, 0) + 1
        if four_gram_counts[gram] >= 3:
            return True
    return False


def _word_count(text: object) -> int:
    return len(WORD_TOKEN_RE.findall(nfc_strip(text)))


def _replay_drop_mask(df: pd.DataFrame) -> pd.Series:
    source = df["source_text"].map(nfc_strip)
    target = df["target_text"].map(nfc_strip)
    source_len = source.str.len()
    target_len = target.str.len()
    ratio = source_len / target_len.clip(lower=1)
    return (
        source.str.contains("¶", regex=False)
        | target.str.contains("¶", regex=False)
        | source.str.contains("BAR", regex=False)
        | target.str.contains("BAR", regex=False)
        | source.str.contains(REPLAY_URL_RE, na=False)
        | target.str.contains(REPLAY_URL_RE, na=False)
        | source.str.startswith("-")
        | target.str.startswith("-")
        | (ratio < 0.5)
        | (ratio > 2.0)
        | (source.map(_word_count) < 3)
        | (target.map(_word_count) < 3)
    )


def filter_replay(df: pd.DataFrame) -> pd.DataFrame:
    """Drop replay rows with document artifacts or weak sentence-pair shape."""
    out = df.copy()
    out["source_text"] = out["source_text"].map(nfc_strip)
    out["target_text"] = out["target_text"].map(nfc_strip)
    return out.loc[~_replay_drop_mask(out)].reset_index(drop=True)


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


def _load_extra_train_rows(extra_path: str | Path, *, label: str, required: bool = True) -> pd.DataFrame | None:
    path = Path(extra_path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{label} parquet not found: {path}")
        print(f"WARNING: {label} parquet not found at {path}; skipping.")
        return None

    extra = pd.read_parquet(path)
    required_cols = set(CORPUS_COLUMNS)
    missing = required_cols - set(extra.columns)
    if missing:
        raise ValueError(f"{label} parquet is missing columns: {sorted(missing)}")
    non_train = extra[extra["split"] != "train"]
    if not non_train.empty:
        examples = non_train[["source_lang", "target_lang", "split", "provenance"]].head(5).to_dict("records")
        raise AssertionError(f"{label} rows must be train-only; examples={examples!r}")
    extra = extra.loc[:, CORPUS_COLUMNS].copy()
    extra["source_text"] = extra["source_text"].map(nfc_strip)
    extra["target_text"] = extra["target_text"].map(nfc_strip)
    return extra.reset_index(drop=True)


def apply_v35_lexicon_flip_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Tag flipped-lexicon sources so the headword register is opt-in at inference."""
    out = df.copy()
    mask = (
        (out["domain"] == "lexicon")
        & (out["target_lang"] == "mbay")
        & out["provenance"].astype(str).str.startswith("flip:")
        & ~out["source_text"].str.startswith(V35_LEX_TAG, na=False)
    )
    out.loc[mask, "source_text"] = V35_LEX_TAG + out.loc[mask, "source_text"]
    return out, int(mask.sum())


_V35_WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*", re.UNICODE)


def load_v35_bt_rows(bt_path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Tagged back-translation + copied-monolingual rows from a bt-generated parquet.

    Targets are real Keegan Mbay text; only sources are model-generated, so gates
    apply to the source side (the silver-text gates guard model-generated targets).
    """
    bt = pd.read_parquet(bt_path)
    missing = {"mbay", "fra", "eng"} - set(bt.columns)
    if missing:
        raise ValueError(f"bt parquet is missing columns: {sorted(missing)}")

    counts = {
        "added_bt_fra": 0,
        "added_bt_eng": 0,
        "added_bt_copy": 0,
        "dropped_bt_source_short": 0,
        "dropped_bt_source_degenerate": 0,
        "dropped_bt_source_dupe": 0,
    }
    mbay_texts = [nfc_strip(t) for t in bt["mbay"]]
    rows: list[tuple[str, ...]] = []
    seen_sources: set[tuple[str, str]] = set()
    for lang in ("fra", "eng"):
        for mb, raw_source in zip(mbay_texts, bt[lang]):
            source = nfc_strip(raw_source)
            words = source.split()
            if len(words) < V35_BT_MIN_SOURCE_WORDS:
                counts["dropped_bt_source_short"] += 1
                continue
            if len({w.lower() for w in words}) < V35_BT_MIN_UNIQUE_SOURCE_WORDS:
                counts["dropped_bt_source_degenerate"] += 1
                continue
            key = (lang, source.lower())
            if key in seen_sources:
                counts["dropped_bt_source_dupe"] += 1
                continue
            seen_sources.add(key)
            rows.append((lang, "mbay", V35_BT_TAG + source, mb, "bt", "bt:asr-keegan", "train"))
            counts[f"added_bt_{lang}"] += 1
    for mb in dict.fromkeys(mbay_texts):
        rows.append(("mbay", "mbay", V35_COPY_TAG + mb, mb, "copy", "copy:asr-keegan", "train"))
        counts["added_bt_copy"] += 1
    return pd.DataFrame(rows, columns=CORPUS_COLUMNS), counts


def _v35_hint_lookup(df: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    lex_flips = df.loc[
        (df["domain"] == "lexicon")
        & (df["target_lang"] == "mbay")
        & df["provenance"].astype(str).str.startswith("flip:")
    ]
    lookup: dict[tuple[str, str], list[str]] = {}
    for source_lang, gloss, headword in zip(
        lex_flips["source_lang"], lex_flips["source_text"], lex_flips["target_text"]
    ):
        gloss = gloss.removeprefix(V35_LEX_TAG)
        words = _V35_WORD_RE.findall(gloss.lower())
        if len(words) > 3:
            continue
        for word in words:
            if len(word) > 3:
                lookup.setdefault((source_lang, word), []).append(headword)
    return lookup


def apply_v35_hints(df: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, int]:
    """Append verified dictionary hints to sentence-register X->mbay train sources.

    A hint is only attached when the headword actually occurs in the target, so
    the model learns hints are trustworthy; p=0.5 dropout keeps the unhinted
    path trained for inputs where inference-time lookup finds nothing.
    """
    lookup = _v35_hint_lookup(df)
    out = df.copy()
    mask = (
        (out["split"] == "train")
        & (out["target_lang"] == "mbay")
        & out["domain"].isin({"example", "expression"})
        & out["provenance"].astype(str).str.startswith("flip:")
    )
    hinted = 0
    for idx in out.index[mask]:
        source = out.at[idx, "source_text"]
        target = out.at[idx, "target_text"]
        digest = hashlib.sha256(f"{seed}:{source}:{target}".encode()).digest()
        if digest[0] >= V35_HINT_PROB_NUM:
            continue
        source_lang = out.at[idx, "source_lang"]
        hints: list[tuple[str, str]] = []
        used_headwords: set[str] = set()
        for word in _V35_WORD_RE.findall(source.lower()):
            if len(word) <= 3:
                continue
            for headword in lookup.get((source_lang, word), []):
                if headword in used_headwords or headword not in target or headword == target:
                    continue
                hints.append((word, headword))
                used_headwords.add(headword)
                break
            if len(hints) >= V35_HINT_MAX:
                break
        if hints:
            out.at[idx, "source_text"] = (
                source + V35_HINT_MARKER + " ; ".join(f"{w} = {hw}" for w, hw in hints)
            )
            hinted += 1
    return out, hinted


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_restored_bible_ready(restored_bible_path: str | Path) -> None:
    restored_path = Path(restored_bible_path)
    if not restored_path.exists():
        raise FileNotFoundError(
            f"Restored bible parquet not found: {restored_path}. "
            "v3.2 requires the READY contract in notes/BIBLE_RESTORED_HANDOFF.md "
            "and py/datasets/processed/bible-restored.parquet."
        )

    ready_path = restored_path.with_suffix(".READY.json")
    if not ready_path.exists():
        print(
            f"WARNING: {ready_path} is absent; continuing via manual override. "
            "See notes/BIBLE_RESTORED_HANDOFF.md READY contract."
        )
        return

    with ready_path.open("r", encoding="utf-8") as handle:
        ready = json.load(handle)
    if ready.get("status") != "ready":
        raise ValueError(f"{ready_path} status is not ready: {ready.get('status')!r}")

    expected_sha = ready.get("sha256")
    actual_sha = _sha256_file(restored_path)
    if expected_sha != actual_sha:
        raise ValueError(
            f"{ready_path} sha256 mismatch for {restored_path}: "
            f"expected {expected_sha!r}, got {actual_sha!r}"
        )


def _load_restored_bible(restored_bible_path: str | Path) -> pd.DataFrame:
    restored = pd.read_parquet(restored_bible_path)
    missing = RESTORED_BIBLE_COLUMNS - set(restored.columns)
    if missing:
        raise ValueError(f"Restored bible parquet is missing columns: {sorted(missing)}")

    restored = restored.copy()
    restored["provenance"] = restored["provenance"].map(nfc_strip)
    restored["restored_text"] = restored["restored_text"].map(nfc_strip)
    restored = restored.drop_duplicates("provenance", keep="first")
    return restored


def build_v32_restored_mbay_char_whitelist(df: pd.DataFrame) -> set[str]:
    """Allow restored Mbay glyphs seen in original train dictionary Mbay text."""
    whitelist = set(V32_BASIC_ALLOWED_CHARS)
    train_dictionary = (df["split"] == "train") & df["domain"].isin(DICTIONARY_DOMAINS)
    for lang_col, text_col in [("source_lang", "source_text"), ("target_lang", "target_text")]:
        mbay_text = df.loc[train_dictionary & (df[lang_col] == "mbay"), text_col]
        for text in mbay_text:
            whitelist.update(nfc_strip(text))
    return whitelist


def restored_text_disallowed_chars(text: object, whitelist: set[str]) -> set[str]:
    return {ch for ch in nfc_strip(text) if ch not in whitelist}


def apply_v32_restored_bible(
    df: pd.DataFrame,
    restored_bible_path: str | Path,
    *,
    flip_word_agreement_min: float = V32_FLIP_WORD_AGREEMENT_MIN,
    flip_roundtrip_min: float = V32_FLIP_ROUNDTRIP_MIN,
) -> tuple[pd.DataFrame, dict[str, int]]:
    restored = _load_restored_bible(restored_bible_path)
    restored_by_provenance = restored.set_index("provenance")
    restored_mbay_whitelist = build_v32_restored_mbay_char_whitelist(df)

    train_bible_mask = (df["split"] == "train") & (df["domain"] == "bible")
    train_bible = df.loc[train_bible_mask].copy()
    joined = train_bible.join(
        restored_by_provenance[["restored_text", "roundtrip_sim", "word_agreement"]],
        on="provenance",
        how="left",
    )

    has_restored = joined["restored_text"].map(lambda value: isinstance(value, str) and len(value) > 0)
    source_gate = has_restored & (joined["roundtrip_sim"] >= V32_SOURCE_ROUNDTRIP_MIN)
    flip_gate = (
        has_restored
        & (joined["word_agreement"] >= flip_word_agreement_min)
        & (joined["roundtrip_sim"] >= flip_roundtrip_min)
    )
    whitelist_safe = joined["restored_text"].map(
        lambda value: not restored_text_disallowed_chars(value, restored_mbay_whitelist)
    )
    repetition_safe = ~joined["restored_text"].map(restored_text_has_repetition_failure)
    source_whitelist_drop = source_gate & ~whitelist_safe
    flip_whitelist_drop = flip_gate & ~whitelist_safe
    source_repetition_drop = source_gate & whitelist_safe & ~repetition_safe
    flip_repetition_drop = flip_gate & whitelist_safe & ~repetition_safe
    source_add_gate = source_gate & whitelist_safe & repetition_safe
    flip_add_gate = flip_gate & whitelist_safe & repetition_safe

    train_bible_verses = train_bible["provenance"].nunique()
    restored_verses = joined.loc[has_restored, "provenance"].nunique()
    source_gate_verses = joined.loc[source_gate, "provenance"].nunique()
    flip_gate_verses = joined.loc[flip_gate, "provenance"].nunique()
    counts = {
        "v32_train_bible_verses": int(train_bible_verses),
        "v32_restored_verses_joined": int(restored_verses),
        "v32_source_gate_verses_in": int(source_gate_verses),
        "v32_source_gate_verses_out": int(restored_verses - source_gate_verses),
        "v32_flip_gate_verses_in": int(flip_gate_verses),
        "v32_flip_gate_verses_out": int(restored_verses - flip_gate_verses),
        "dropped_v32_restored_source_non_whitelist": int(source_whitelist_drop.sum()),
        "dropped_v32_restored_flip_non_whitelist": int(flip_whitelist_drop.sum()),
        "dropped_v32_restored_source_repetition": int(source_repetition_drop.sum()),
        "dropped_v32_restored_flip_repetition": int(flip_repetition_drop.sum()),
        "added_v32_restored_source": 0,
        "added_v32_restored_flip": 0,
    }

    additions: list[pd.DataFrame] = []
    source_additions = joined.loc[source_add_gate, df.columns].copy()
    if not source_additions.empty:
        source_additions["source_text"] = joined.loc[source_add_gate, "restored_text"].to_numpy()
        source_additions["provenance"] = "restored:" + source_additions["provenance"].astype(str)
        counts["added_v32_restored_source"] = len(source_additions)
        additions.append(source_additions)

    flip_additions = joined.loc[flip_add_gate, df.columns].copy()
    if not flip_additions.empty:
        flip_additions[["source_lang", "target_lang"]] = flip_additions[["target_lang", "source_lang"]].to_numpy()
        flip_additions["source_text"] = joined.loc[flip_add_gate, "target_text"].to_numpy()
        flip_additions["target_text"] = joined.loc[flip_add_gate, "restored_text"].to_numpy()
        flip_additions["provenance"] = "restored-flip:" + flip_additions["provenance"].astype(str)
        counts["added_v32_restored_flip"] = len(flip_additions)
        additions.append(flip_additions)

    if additions:
        df = pd.concat([df, *additions], ignore_index=True)
    return df, counts


def build_v3(
    input_path: str | Path,
    *,
    no_notone_aug: bool = False,
    v31: bool = False,
    v32: bool = False,
    v34: bool = False,
    v35: bool = False,
    restored_bible_path: str | Path = "datasets/processed/bible-restored.parquet",
    v32_flip_wa: float | None = None,
    v32_flip_rt: float = V32_FLIP_ROUNDTRIP_MIN,
    codeswitch_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    bt_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    # v3.5 keeps the v3.4 recipe but reverts the full-lexicon flip upweight
    # (register-bleed finding), so it deliberately does NOT set v34.
    v32 = v32 or v34 or v35
    v31 = v31 or v32
    if v32_flip_wa is None:
        v32_flip_wa = V34_FLIP_WORD_AGREEMENT_MIN if (v34 or v35) else V32_FLIP_WORD_AGREEMENT_MIN
    if v32:
        verify_restored_bible_ready(restored_bible_path)

    df = pd.read_parquet(input_path)
    required_cols = set(CORPUS_COLUMNS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input corpus is missing columns: {sorted(missing)}")

    df = df.copy()
    counts: dict[str, int] = {
        "dropped_bible_short_target": 0,
        "dropped_bible_remaining_ref": 0,
        "dropped_bible_apparatus_remnant": 0,
        "dropped_bible_initial_integer_run": 0,
        "dropped_bible_mostly_digits_punct": 0,
        "dropped_bible_truncated_target": 0,
        "bible_truncation_threshold_chars": 30,
        "dropped_bible_length_ratio": 0,
        "cleaned_example_expression_gloss_suffix": 0,
        "dropped_empty_after_hygiene": 0,
        "dropped_exact_dupes": 0,
        "dropped_train_eval_pair_leakage": 0,
        "added_flipped_dictionary": 0,
        "added_v34_full_flipped_train_lexicon": 0,
        "added_codeswitch": 0,
        "added_replay": 0,
        "dropped_replay_filter": 0,
        "added_notone_train": 0,
    }

    bible_mask = df["domain"] == "bible"
    df.loc[bible_mask, "target_text"] = df.loc[bible_mask, "target_text"].map(clean_bible_target)
    remaining_ref_mask = bible_mask & df["target_text"].str.contains(VERSE_REF_RE, na=False)
    counts["dropped_bible_remaining_ref"] = int(remaining_ref_mask.sum())
    df = df.loc[~remaining_ref_mask].reset_index(drop=True)

    bible_mask = df["domain"] == "bible"
    apparatus_mask = bible_mask & df["target_text"].str.contains(BIBLE_APPARATUS_RE, na=False)
    counts["dropped_bible_apparatus_remnant"] = int(apparatus_mask.sum())
    df = df.loc[~apparatus_mask].reset_index(drop=True)

    bible_mask = df["domain"] == "bible"
    integer_run_mask = bible_mask & df["target_text"].str.contains(BIBLE_INITIAL_OR_PUNCT_INT_RUN_RE, na=False)
    counts["dropped_bible_initial_integer_run"] = int(integer_run_mask.sum())
    df = df.loc[~integer_run_mask].reset_index(drop=True)

    bible_mask = df["domain"] == "bible"
    mostly_digits_punct_mask = bible_mask & df["target_text"].map(bible_target_is_mostly_digits_punct)
    counts["dropped_bible_mostly_digits_punct"] = int(mostly_digits_punct_mask.sum())
    df = df.loc[~mostly_digits_punct_mask].reset_index(drop=True)

    bible_mask = df["domain"] == "bible"
    target_len = df["target_text"].map(lambda value: len(nfc_strip(value)))
    lacks_final_punct = df["target_text"].map(bible_target_lacks_final_punctuation)
    truncation_30_mask = bible_mask & (target_len < 30) & lacks_final_punct
    bible_rows = int(bible_mask.sum())
    truncation_threshold = 20 if bible_rows and int(truncation_30_mask.sum()) / bible_rows > 0.03 else 30
    counts["bible_truncation_threshold_chars"] = truncation_threshold
    truncated_mask = bible_mask & (target_len < truncation_threshold) & lacks_final_punct
    counts["dropped_bible_truncated_target"] = int(truncated_mask.sum())
    df = df.loc[~truncated_mask].reset_index(drop=True)

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

    example_expression_mask = df["domain"].isin({"example", "expression"})
    original_example_expression_targets = df.loc[example_expression_mask, "target_text"].map(nfc_strip)
    df.loc[example_expression_mask, "target_text"] = df.loc[
        example_expression_mask, "target_text"
    ].map(clean_example_expression_target)
    cleaned_example_expression_targets = df.loc[example_expression_mask, "target_text"].map(nfc_strip)
    counts["cleaned_example_expression_gloss_suffix"] = int(
        (original_example_expression_targets != cleaned_example_expression_targets).sum()
    )

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

    v34_full_train_mbay_lexicon = pd.DataFrame(columns=df.columns)
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
        if v34:
            full_lexicon_mask = (
                (df["split"] == "train")
                & (df["domain"] == "lexicon")
                & (df["source_lang"] == "mbay")
            )
            v34_full_train_mbay_lexicon = df.loc[full_lexicon_mask].copy()
        df, counts["dropped_v31_lexicon_train_downsample"] = _downsample_v31_lexicon_train(df, seed=seed)

    if v32:
        df, v32_counts = apply_v32_restored_bible(
            df,
            restored_bible_path,
            flip_word_agreement_min=v32_flip_wa,
            flip_roundtrip_min=v32_flip_rt,
        )
        counts.update(v32_counts)

    if v34 and codeswitch_path is not None:
        codeswitch = _load_extra_train_rows(codeswitch_path, label="codeswitch", required=True)
        if codeswitch is not None:
            counts["added_codeswitch"] = len(codeswitch)
            df = pd.concat([df, codeswitch], ignore_index=True)

    if (v34 or v35) and replay_path is not None:
        replay = _load_extra_train_rows(replay_path, label="replay", required=False)
        if replay is not None:
            replay_before = len(replay)
            replay = filter_replay(replay)
            counts["dropped_replay_filter"] = replay_before - len(replay)
            counts["added_replay"] = len(replay)
            df = pd.concat([df, replay], ignore_index=True)

    flip_mask = (df["source_lang"] == "mbay") & df["domain"].isin(DICTIONARY_DOMAINS)
    if v34 and not v34_full_train_mbay_lexicon.empty:
        current_flip_source = df.loc[
            flip_mask & ~((df["split"] == "train") & (df["domain"] == "lexicon"))
        ].copy()
        flipped_source = pd.concat([current_flip_source, v34_full_train_mbay_lexicon], ignore_index=True)
        counts["added_v34_full_flipped_train_lexicon"] = len(v34_full_train_mbay_lexicon)
    else:
        flipped_source = df.loc[flip_mask].copy()
    flipped = flipped_source.copy()
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

    if v35:
        df, counts["tagged_v35_lexicon_flips"] = apply_v35_lexicon_flip_tags(df)
        if bt_path is not None:
            bt_rows, bt_counts = load_v35_bt_rows(bt_path)
            counts.update(bt_counts)
            df = pd.concat([df, bt_rows], ignore_index=True)

    if v31:
        df = apply_v31_source_tags(df)

    if v35:
        df, counts["hinted_v35_example_rows"] = apply_v35_hints(df, seed=seed)

    df, counts["dropped_train_eval_pair_leakage"] = _drop_train_eval_pair_leakage(df)
    _assert_no_train_eval_leakage(df)
    return df.reset_index(drop=True), counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input v2 parquet path")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--no-notone-aug", action="store_true", help="Disable tone-stripped train augmentation")
    parser.add_argument("--v31", action="store_true", help="Build corpus v3.1 with book-level bible holdout and tags")
    parser.add_argument("--v32", action="store_true", help="Build corpus v3.2 with restored bible train additions")
    parser.add_argument("--v34", action="store_true", help="Build corpus v3.4 with restored bible and DDA codeswitch rows")
    parser.add_argument(
        "--v35",
        action="store_true",
        help="Build corpus v3.5: v3.4 hygiene + <lex>-tagged 1x lexicon flips + tagged BT/copy rows + dictionary hints",
    )
    parser.add_argument(
        "--bt",
        default=None,
        help="bt-generated parquet (columns mbay/fra/eng) for --v35 tagged back-translation rows",
    )
    parser.add_argument(
        "--restored-bible",
        default="datasets/processed/bible-restored.parquet",
        help="Restored bible parquet for --v32/--v34",
    )
    parser.add_argument(
        "--codeswitch",
        default=None,
        help="Codeswitch parquet to append for --v34; rows must be train-only",
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Optional replay parquet to append for --v34; skipped with a warning if absent",
    )
    parser.add_argument("--v32-flip-wa", type=float, default=None)
    parser.add_argument("--v32-flip-rt", type=float, default=V32_FLIP_ROUNDTRIP_MIN)
    parser.add_argument("--seed", type=int, default=42, help="Seed for v3.1 deterministic sampling")
    args = parser.parse_args()

    input_path = Path(args.input)
    default_output = "datasets/processed/mbay-corpus-v3.parquet"
    if args.v35:
        default_output = "datasets/processed/mbay-corpus-v3.5.parquet"
    elif args.v34:
        default_output = "datasets/processed/mbay-corpus-v3.4.parquet"
    elif args.v32:
        default_output = "datasets/processed/mbay-corpus-v3.2.parquet"
    elif args.v31:
        default_output = "datasets/processed/mbay-corpus-v3.1.parquet"
    output_path = Path(
        args.output
        or default_output
    )

    before = pd.read_parquet(input_path)
    _print_stats_table("Rows before", before)

    after, counts = build_v3(
        input_path,
        no_notone_aug=args.no_notone_aug,
        v31=args.v31,
        v32=args.v32,
        v34=args.v34,
        v35=args.v35,
        restored_bible_path=args.restored_bible,
        v32_flip_wa=args.v32_flip_wa,
        v32_flip_rt=args.v32_flip_rt,
        codeswitch_path=args.codeswitch,
        replay_path=args.replay,
        bt_path=args.bt,
        seed=args.seed,
    )

    print("\nFilter/addition counts")
    for name, count in counts.items():
        print(f"{name}: {count}")

    _print_stats_table("Rows after", after)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    after.to_parquet(output_path, index=False)
    print(f"\nSaved {len(after):,} rows to {output_path}")


if __name__ == "__main__":
    main()
