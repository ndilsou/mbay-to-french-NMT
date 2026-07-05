"""Orthography bridge report for bible-style and dictionary-style Mbay."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import unicodedata as ud

import pandas as pd

from mbay_nmt.tone_restore.lexicon_restore import DICT_DOMAINS
from mbay_nmt.tone_restore.skeleton import NASAL_TILDE_BELOW, TONE_MARKS, dict_to_bible_style, strip_tones, skeleton

DEFAULT_CORPUS = Path("datasets/processed/mbay-corpus-v2.parquet")
TOKEN_SPLIT_RE = re.compile(r"[\s\-\u2010-\u2015]+")
NASAL_ABOVE = set("ãẽĩõũÃẼĨÕŨ")


def split_tokens(text: str) -> list[str]:
    return [part.lower() for part in TOKEN_SPLIT_RE.split(text) if part]


def mbay_texts(df: pd.DataFrame, domains: set[str] | None = None) -> pd.Series:
    rows = df[df["source_lang"].eq("mbay")]
    if domains is not None:
        rows = rows[rows["domain"].isin(domains)]
    return rows["source_text"].dropna().astype(str)


def char_counts(texts: pd.Series) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(ud.normalize("NFC", text))
    return counts


def char_group(ch: str) -> str:
    decomposed = ud.normalize("NFD", ch)
    if ch.isupper() or any(part.isupper() for part in decomposed):
        return "case"
    if ch in NASAL_ABOVE or NASAL_TILDE_BELOW in decomposed or "\u0303" in decomposed:
        return "nasal convention"
    if ch in TONE_MARKS or any(part in TONE_MARKS for part in decomposed):
        return "tone marks"
    return "unexplained"


def describe_char(ch: str) -> str:
    names = " + ".join(ud.name(part, "UNKNOWN") for part in ud.normalize("NFD", ch))
    printable = ch if ch != " " else "SPACE"
    return f"{printable!r} U+{ord(ch):04X} {names}"


def print_inventory_diff(bible_counts: Counter[str], dict_counts: Counter[str]) -> None:
    print("Character inventory diff (NFC)")
    for title, left, right in [
        ("Bible chars not in dictionary", bible_counts, dict_counts),
        ("Dictionary chars not in bible", dict_counts, bible_counts),
    ]:
        print(f"\n{title}")
        grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for ch in sorted(set(left) - set(right), key=lambda c: (char_group(c), c)):
            grouped[char_group(ch)].append((ch, left[ch]))
        for group in ["tone marks", "nasal convention", "case", "unexplained"]:
            items = grouped.get(group, [])
            print(f"  {group}: {len(items)}")
            for ch, count in items:
                print(f"    {describe_char(ch)}: {count:,}")


def dictionary_skeleton_lexicon(dict_tokens: list[str]) -> dict[str, set[str]]:
    lexicon: dict[str, set[str]] = defaultdict(set)
    for token in dict_tokens:
        key = skeleton(token)
        if key:
            lexicon[key].add(token)
    return lexicon


def weighted_bridge_stats(bible_tokens: Counter[str], lexicon: dict[str, set[str]]) -> dict[str, int | float]:
    total = bible_tokens.total()
    matched = 0
    ambiguity = Counter()
    for token, count in bible_tokens.items():
        forms = lexicon.get(skeleton(token), set())
        if not forms:
            continue
        matched += count
        form_count = len(forms)
        if form_count == 1:
            ambiguity["1"] += count
        elif form_count == 2:
            ambiguity["2"] += count
        else:
            ambiguity["3+"] += count

    return {
        "total": total,
        "matched": matched,
        "oov": total - matched,
        "unambiguous": ambiguity["1"],
        "ambiguous_2": ambiguity["2"],
        "ambiguous_3plus": ambiguity["3+"],
    }


def pct(part: int, total: int) -> float:
    return part / total if total else 0.0


def print_bridge_stats(
    bible_tokens: Counter[str],
    dict_tokens: Counter[str],
    lexicon: dict[str, set[str]],
) -> dict[str, int | float]:
    stats = weighted_bridge_stats(bible_tokens, lexicon)
    total = int(stats["total"])
    matched = int(stats["matched"])
    unambiguous = int(stats["unambiguous"])
    ambiguous = int(stats["ambiguous_2"]) + int(stats["ambiguous_3plus"])
    oov = int(stats["oov"])

    print("\nToken-level bridge stats")
    print(f"- unique dictionary Mbay tokens: {len(dict_tokens):,}")
    print(f"- unique bible Mbay tokens: {len(bible_tokens):,}")
    print(f"- bible token occurrences with dictionary skeleton match: {pct(matched, total):.2%} ({matched:,}/{total:,})")

    if matched:
        print("- matched skeleton ambiguity, weighted by bible occurrence:")
        print(f"  - exactly 1 toned form: {pct(unambiguous, matched):.2%} ({unambiguous:,}/{matched:,})")
        print(f"  - exactly 2 toned forms: {pct(int(stats['ambiguous_2']), matched):.2%} ({int(stats['ambiguous_2']):,}/{matched:,})")
        print(f"  - 3+ toned forms: {pct(int(stats['ambiguous_3plus']), matched):.2%} ({int(stats['ambiguous_3plus']):,}/{matched:,})")

    converted_dict_tokens = {dict_to_bible_style(token).lower() for token in dict_tokens}
    detoned_no_nasal_conversion = {strip_tones(token).lower() for token in dict_tokens}
    dict_skeletons = set(lexicon)
    exact_converted = sum(count for token, count in bible_tokens.items() if token in converted_dict_tokens)
    exact_detoned = sum(count for token, count in bible_tokens.items() if token in detoned_no_nasal_conversion)
    print(
        "- bible token occurrences exactly matching dict_to_bible_style(dictionary token): "
        f"{pct(exact_converted, total):.2%} ({exact_converted:,}/{total:,})"
    )
    print(
        "- bible token occurrences exactly matching strip_tones(dictionary token) without nasal conversion: "
        f"{pct(exact_detoned, total):.2%} ({exact_detoned:,}/{total:,})"
    )
    print(f"- nasal-conversion exact-match gain: {pct(exact_converted - exact_detoned, total):.2%}")
    print_nasal_validation(bible_tokens, converted_dict_tokens, detoned_no_nasal_conversion, dict_skeletons)

    print("\nVerdict")
    print(
        "- estimated restorable unambiguously by pure lexicon lookup: "
        f"{pct(unambiguous, total):.2%} ({unambiguous:,}/{total:,})"
    )
    print(f"- ambiguous by pure lexicon lookup: {pct(ambiguous, total):.2%} ({ambiguous:,}/{total:,})")
    print(f"- OOV by pure lexicon lookup: {pct(oov, total):.2%} ({oov:,}/{total:,})")

    stats["exact_converted"] = exact_converted
    stats["exact_detoned"] = exact_detoned
    return stats


def print_nasal_validation(
    bible_tokens: Counter[str],
    converted_dict_tokens: set[str],
    detoned_no_nasal_conversion: set[str],
    dict_skeletons: set[str],
) -> None:
    print("\nNasal-mapping validation")
    for label, chars in [("any nasal vowel", "ãẽĩõũ"), ("ã", "ã"), ("ẽ", "ẽ"), ("ĩ", "ĩ"), ("õ", "õ"), ("ũ", "ũ")]:
        subset = {token: count for token, count in bible_tokens.items() if any(ch in token for ch in chars)}
        total = sum(subset.values())
        if not total:
            continue
        converted = sum(count for token, count in subset.items() if token in converted_dict_tokens)
        detoned = sum(count for token, count in subset.items() if token in detoned_no_nasal_conversion)
        skeleton_matches = sum(count for token, count in subset.items() if skeleton(token) in dict_skeletons)
        print(
            f"- {label}: dict_to_bible_style exact {pct(converted, total):.2%} "
            f"({converted:,}/{total:,}); strip_tones-only exact {pct(detoned, total):.2%} "
            f"({detoned:,}/{total:,}); skeleton match {pct(skeleton_matches, total):.2%}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.corpus)
    bible_texts = mbay_texts(df, {"bible"})
    dict_texts = mbay_texts(df, DICT_DOMAINS)

    print(f"Loaded {len(bible_texts):,} bible Mbay rows and {len(dict_texts):,} dictionary-domain Mbay rows")
    print_inventory_diff(char_counts(bible_texts), char_counts(dict_texts))

    bible_token_counts = Counter(token for text in bible_texts for token in split_tokens(text))
    dict_token_counts = Counter(token for text in dict_texts for token in split_tokens(text))
    lexicon = dictionary_skeleton_lexicon(list(dict_token_counts))
    print_bridge_stats(bible_token_counts, dict_token_counts, lexicon)


if __name__ == "__main__":
    main()
