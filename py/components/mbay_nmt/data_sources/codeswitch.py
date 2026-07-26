"""Dictionary codeswitch augmentation for X->Mbay lexical grounding.

Usage:
    uv run python -m mbay_nmt.data_sources.codeswitch \
        --input datasets/processed/mbay-corpus-v2.parquet \
        --output datasets/processed/mbay-corpus-v3.4-codeswitch.parquet
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mbay_nmt.data_sources.build_v3 import nfc_strip

SUPPORTED_TARGET_LANGS = {"eng", "fra"}
TOKEN_CHARS = r"\w\u0300-\u036f\u1dc0-\u1dff'’ʼ-"
TOKEN_RE = re.compile(rf"[{TOKEN_CHARS}]+", flags=re.UNICODE)
BRACKETED_CONTEXT_RE = re.compile(r"\s*[\(\[\{][^\)\]\}]*[\)\]\}]")
TRAILING_PUNCT_RE = re.compile(r"[\s\.;:,!?]+$")

EN_NOUN_STOPWORDS = {
    "a",
    "an",
    "and",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "with",
}
FR_STOPWORDS = {
    "au",
    "aux",
    "d",
    "de",
    "des",
    "du",
    "en",
    "et",
    "l",
    "la",
    "le",
    "les",
    "un",
    "une",
}
FR_AMBIGUOUS_INFINITIVES = {"cher", "fer", "hier", "hiver", "mer", "ver"}


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    lang: str
    gloss: str
    mbay: str
    pos: str


@dataclass(frozen=True, slots=True)
class Anchor:
    lang: str
    mbay_text: str
    x_text: str
    provenance: str
    entry: LexiconEntry


def clean_gloss(text: object) -> str:
    cleaned = nfc_strip(text).lower()
    cleaned = BRACKETED_CONTEXT_RE.sub(" ", cleaned)
    cleaned = TRAILING_PUNCT_RE.sub("", cleaned)
    return " ".join(cleaned.strip(" \"'`«»“”").split())


def _is_single_token(text: str) -> bool:
    if not TOKEN_RE.fullmatch(text):
        return False
    return any(ch.isalpha() for ch in text) and "_" not in text


def _word_count(text: str) -> int:
    return len(text.split())


def guess_pos(lang: str, gloss: str) -> str | None:
    words = gloss.split()
    if not words or len(words) > 3:
        return None

    if lang == "eng":
        if gloss.startswith("to ") and len(words) >= 2 and all(_is_single_token(word) for word in words[1:]):
            return "verb"
        if len(words) == 1 and _is_single_token(words[0]) and words[0] not in EN_NOUN_STOPWORDS:
            return "noun"
        return None

    if lang == "fra":
        if len(words) != 1 or not _is_single_token(words[0]) or words[0] in FR_STOPWORDS:
            return None
        word = words[0]
        if (
            len(word) > 4
            and word not in FR_AMBIGUOUS_INFINITIVES
            and (word.endswith("er") or word.endswith("ir") or word.endswith("re"))
        ):
            return "verb"
        return "noun"

    return None


def _whole_phrase_re(phrase: str, *, ignore_case: bool = False) -> re.Pattern[str]:
    escaped_parts = [re.escape(part) for part in phrase.split()]
    body = r"\s+".join(escaped_parts)
    flags = re.UNICODE | (re.IGNORECASE if ignore_case else 0)
    return re.compile(rf"(?<![{TOKEN_CHARS}]){body}(?![{TOKEN_CHARS}])", flags=flags)


def contains_whole_phrase(text: object, phrase: str, *, ignore_case: bool = False) -> bool:
    return bool(_whole_phrase_re(phrase, ignore_case=ignore_case).search(nfc_strip(text)))


def replace_first_whole_phrase(text: object, old: str, new: str, *, ignore_case: bool = False) -> str | None:
    source = nfc_strip(text)
    replaced, count = _whole_phrase_re(old, ignore_case=ignore_case).subn(new, source, count=1)
    return replaced if count == 1 else None


def build_word_maps(df: pd.DataFrame) -> dict[str, dict[str, list[LexiconEntry]]]:
    """Build fra/eng gloss -> Mbay entry maps from short lexicon rows."""
    maps: dict[str, dict[str, list[LexiconEntry]]] = {lang: defaultdict(list) for lang in SUPPORTED_TARGET_LANGS}
    seen: set[tuple[str, str, str, str]] = set()

    lexicon = df[
        (df["domain"] == "lexicon")
        & (df["source_lang"] == "mbay")
        & df["target_lang"].isin(SUPPORTED_TARGET_LANGS)
    ]
    for row in lexicon.itertuples(index=False):
        mbay = nfc_strip(row.source_text)
        gloss = clean_gloss(row.target_text)
        lang = str(row.target_lang)
        if not mbay or not gloss or _word_count(gloss) > 3 or not _is_single_token(mbay):
            continue
        pos = guess_pos(lang, gloss)
        if pos is None:
            continue
        key = (lang, gloss, mbay, pos)
        if key in seen:
            continue
        seen.add(key)
        maps[lang][gloss].append(LexiconEntry(lang=lang, gloss=gloss, mbay=mbay, pos=pos))

    return {lang: dict(gloss_map) for lang, gloss_map in maps.items()}


def _entries_by_mbay(
    word_maps: dict[str, dict[str, list[LexiconEntry]]],
) -> dict[str, dict[str, list[LexiconEntry]]]:
    by_mbay: dict[str, dict[str, list[LexiconEntry]]] = {lang: defaultdict(list) for lang in SUPPORTED_TARGET_LANGS}
    for lang, gloss_map in word_maps.items():
        for entries in gloss_map.values():
            for entry in entries:
                by_mbay[lang][entry.mbay].append(entry)
    return {lang: dict(entries) for lang, entries in by_mbay.items()}


def _entries_by_pos(
    word_maps: dict[str, dict[str, list[LexiconEntry]]],
) -> dict[str, dict[str, list[LexiconEntry]]]:
    by_pos: dict[str, dict[str, list[LexiconEntry]]] = {lang: defaultdict(list) for lang in SUPPORTED_TARGET_LANGS}
    for lang, gloss_map in word_maps.items():
        for entries in gloss_map.values():
            for entry in entries:
                by_pos[lang][entry.pos].append(entry)
    return {lang: dict(entries) for lang, entries in by_pos.items()}


def find_anchors(df: pd.DataFrame, word_maps: dict[str, dict[str, list[LexiconEntry]]]) -> list[Anchor]:
    by_mbay = _entries_by_mbay(word_maps)
    examples = df[
        (df["domain"] == "example")
        & (df["source_lang"] == "mbay")
        & df["target_lang"].isin(SUPPORTED_TARGET_LANGS)
    ]
    anchors: list[Anchor] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for row in examples.itertuples(index=False):
        lang = str(row.target_lang)
        mbay_text = nfc_strip(row.source_text)
        x_text = nfc_strip(row.target_text)
        if "____" in mbay_text or "____" in x_text:
            continue
        source_tokens = {match.group(0) for match in TOKEN_RE.finditer(mbay_text)}
        for token in source_tokens:
            for entry in by_mbay.get(lang, {}).get(token, []):
                if not contains_whole_phrase(x_text, entry.gloss, ignore_case=True):
                    continue
                key = (lang, mbay_text, x_text, str(row.provenance), entry.mbay)
                if key in seen:
                    continue
                seen.add(key)
                anchors.append(
                    Anchor(
                        lang=lang,
                        mbay_text=mbay_text,
                        x_text=x_text,
                        provenance=str(row.provenance),
                        entry=entry,
                    )
                )
    return anchors


def generate_codeswitch_rows(
    df: pd.DataFrame,
    *,
    max_rows: int = 40_000,
    seed: int = 42,
    k: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = random.Random(seed)
    word_maps = build_word_maps(df)
    anchors = find_anchors(df, word_maps)
    by_pos = _entries_by_pos(word_maps)
    rng.shuffle(anchors)

    rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    attempted = 0

    for anchor in anchors:
        if len(rows) + 2 > max_rows:
            break
        candidates = [
            entry
            for entry in by_pos.get(anchor.lang, {}).get(anchor.entry.pos, [])
            if entry.mbay != anchor.entry.mbay and entry.gloss != anchor.entry.gloss
        ]
        rng.shuffle(candidates)
        emitted_for_anchor = 0
        for candidate in candidates:
            if emitted_for_anchor >= k or len(rows) + 2 > max_rows:
                break
            attempted += 1
            mbay_text = replace_first_whole_phrase(anchor.mbay_text, anchor.entry.mbay, candidate.mbay)
            x_text = replace_first_whole_phrase(anchor.x_text, anchor.entry.gloss, candidate.gloss, ignore_case=True)
            if mbay_text is None or x_text is None:
                continue
            mbay_text = nfc_strip(mbay_text)
            x_text = nfc_strip(x_text)
            provenance = "dda:" + anchor.provenance
            bidirectional = [
                {
                    "source_lang": anchor.lang,
                    "target_lang": "mbay",
                    "source_text": x_text,
                    "target_text": mbay_text,
                    "domain": "codeswitch",
                    "provenance": provenance,
                    "split": "train",
                },
                {
                    "source_lang": "mbay",
                    "target_lang": anchor.lang,
                    "source_text": mbay_text,
                    "target_text": x_text,
                    "domain": "codeswitch",
                    "provenance": provenance,
                    "split": "train",
                },
            ]
            keys = [
                (str(row["source_lang"]), str(row["target_lang"]), str(row["source_text"]), str(row["target_text"]))
                for row in bidirectional
            ]
            if any(key in seen_pairs for key in keys):
                continue
            rows.extend(bidirectional)
            seen_pairs.update(keys)
            emitted_for_anchor += 1

    out = pd.DataFrame(
        rows,
        columns=["source_lang", "target_lang", "source_text", "target_text", "domain", "provenance", "split"],
    )
    direction_counts = Counter(
        f"{row.source_lang}->{row.target_lang}" for row in out.itertuples(index=False)
    )
    stats = {
        "word_map_sizes": {lang: sum(len(entries) for entries in gloss_map.values()) for lang, gloss_map in word_maps.items()},
        "word_map_glosses": {lang: len(gloss_map) for lang, gloss_map in word_maps.items()},
        "anchors": len(anchors),
        "substitutions_attempted": attempted,
        "rows_emitted": len(out),
        "rows_by_direction": dict(sorted(direction_counts.items())),
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input v2 parquet path")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max-rows", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    out, stats = generate_codeswitch_rows(df, max_rows=args.max_rows, seed=args.seed)

    print("Codeswitch DDA stats")
    print(f"word-map entries: {stats['word_map_sizes']}")
    print(f"word-map glosses: {stats['word_map_glosses']}")
    print(f"anchored sentences found: {stats['anchors']}")
    print(f"substitutions attempted: {stats['substitutions_attempted']}")
    print(f"rows emitted: {stats['rows_emitted']}")
    print(f"rows emitted per direction/lang: {stats['rows_by_direction']}")

    print("\nRandom sample pairs")
    if out.empty:
        print("(no rows)")
    else:
        sample = out.sample(n=min(10, len(out)), random_state=args.seed)
        for row in sample.itertuples(index=False):
            print(f"{row.source_lang}->{row.target_lang} | {row.source_text} ||| {row.target_text}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"\nSaved {len(out):,} rows to {output_path}")


if __name__ == "__main__":
    main()
