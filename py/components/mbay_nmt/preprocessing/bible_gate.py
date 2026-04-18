"""Bible orthography compatibility gate.

Compares the character inventory of scraped Mbay bible text against Keegan's
Mbay dictionary corpus. Classifies mismatches as trivial (auto-fixable) or
fatal (reject). Computes lexical overlap on a sample of verses.

Usage:
    uv run python -m mbay_nmt.preprocessing.bible_gate \\
        --bible datasets/processed/bible-mbay-test.parquet \\
        --keegan datasets/processed/mbay-pg-export.parquet \\
        --report datasets/processed/bible-compatibility-report.md
"""

from __future__ import annotations

import argparse
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import regex

# Known trivial substitutions: bible convention → Keegan convention.
# MlL (2015) uses tilde-above for nasalization; Keegan uses tilde-below.
TRIVIAL_SUBS: dict[str, str] = {
    "\u00e3": "a\u0330",  # ã → a̰ (a + combining tilde below)
    "\u00f5": "o\u0330",  # õ → o̰
    "\u0129": "i\u0330",  # ĩ → ḭ
    "\u0169": "u\u0330",  # ũ → ṵ
    "\u1ebd": "e\u0330",  # ẽ → ḛ
}

# Fatal mismatches: chars that indicate incompatible orthography.
# If the bible uses these AND Keegan doesn't (or vice versa), reject.
FATAL_CHARS = {
    "\u00fe",  # þ (thorn) — legacy encoding, not modern Mbay
    "\u00f0",  # ð (eth) — legacy encoding
    "\u00f8",  # ø — legacy encoding
}


def _strip_combining(text: str) -> str:
    """Strip all combining marks (tone, nasalization) to get base-form skeleton."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", regex.sub(r"\p{M}", "", nfd))


def apply_trivial_subs(text: str) -> str:
    for src, tgt in TRIVIAL_SUBS.items():
        text = text.replace(src, tgt)
    return unicodedata.normalize("NFC", text)


def extract_codepoints(texts: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for text in texts:
        for ch in unicodedata.normalize("NFD", text):
            counts[ord(ch)] += 1
    return counts


def word_types(texts: list[str]) -> set[str]:
    words: set[str] = set()
    for text in texts:
        for w in text.lower().split():
            w = w.strip(".,;:!?()[]{}\"'«»-–—")
            if w:
                words.add(w)
    return words


def run_gate(
    bible_df: pd.DataFrame,
    keegan_df: pd.DataFrame,
    sample_size: int = 200,
) -> dict:
    """Run the compatibility gate. Returns a report dict."""
    # Extract Mbay-only source texts.
    bible_mbay = bible_df[bible_df["source_lang"] == "mbay"]["source_text"].tolist()
    keegan_mbay = keegan_df[keegan_df["source_lang"] == "mbay"]["source_text"].tolist()

    if not bible_mbay:
        return {"decision": "reject", "reason": "no Mbay text in bible data"}

    # Step 1: Character inventory diff.
    bible_cp = extract_codepoints(bible_mbay)
    keegan_cp = extract_codepoints(keegan_mbay)
    bible_only = set(bible_cp) - set(keegan_cp)
    keegan_only = set(keegan_cp) - set(bible_cp)

    # Classify mismatches.
    trivial_fixes = {}
    fatal_mismatches = []

    for cp in bible_only:
        ch = chr(cp)
        if ch in TRIVIAL_SUBS:
            trivial_fixes[ch] = TRIVIAL_SUBS[ch]
        elif ch in FATAL_CHARS:
            fatal_mismatches.append((ch, unicodedata.name(ch, f"U+{cp:04X}"), bible_cp[cp]))
        elif unicodedata.category(ch).startswith("M"):
            # Combining mark not in Keegan — check if trivial or fatal.
            name = unicodedata.name(ch, f"U+{cp:04X}")
            if bible_cp[cp] < 10:
                trivial_fixes[ch] = ""  # rare, drop
            else:
                fatal_mismatches.append((ch, name, bible_cp[cp]))

    # Step 2: Fatal check — reject if any fatal char is >1% of word-types.
    bible_words = word_types(bible_mbay)
    fatal_word_pct = 0
    for ch, name, count in fatal_mismatches:
        words_with = sum(1 for w in bible_words if ch in w)
        pct = words_with / max(len(bible_words), 1) * 100
        fatal_word_pct = max(fatal_word_pct, pct)

    # Step 3: Lexical overlap after trivial normalization.
    normalized_bible = [apply_trivial_subs(t) for t in bible_mbay[:sample_size]]
    bible_wt = word_types(normalized_bible)
    keegan_wt = word_types(keegan_mbay)

    overlap = bible_wt & keegan_wt
    overlap_pct = len(overlap) / max(len(bible_wt), 1) * 100

    # Step 3b: Base-form overlap (strip all combining marks = tone-neutral).
    # Catches the case where orthographies match at base level but differ
    # in tone-marking density (e.g., bible under-marks tone vs dictionary).
    bible_base = {_strip_combining(w) for w in bible_wt}
    keegan_base = {_strip_combining(w) for w in keegan_wt}
    base_overlap = bible_base & keegan_base
    base_overlap_pct = len(base_overlap) / max(len(bible_base), 1) * 100

    # Decision: use base-form overlap as the primary metric if strict is low
    # but base-form suggests compatible orthography (tone-marking gap only).
    if fatal_word_pct > 1.0:
        decision = "reject"
        weight = 0.0
        reason = f"Fatal chars in {fatal_word_pct:.1f}% of word-types"
    elif overlap_pct >= 80:
        decision = "accept"
        weight = 1.0
        reason = f"Strict lexical overlap {overlap_pct:.1f}% ≥ 80%"
    elif overlap_pct >= 60:
        decision = "accept_low_weight"
        weight = 0.3
        reason = f"Strict lexical overlap {overlap_pct:.1f}% (60-80% range)"
    elif base_overlap_pct >= 15:
        # Strict overlap is low but base-forms match — tone-marking gap.
        decision = "accept_low_weight"
        weight = 0.3
        reason = (f"Strict overlap {overlap_pct:.1f}% but base-form overlap "
                  f"{base_overlap_pct:.1f}% — tone-marking density gap, not orthographic incompatibility")
    else:
        decision = "reject"
        weight = 0.0
        reason = f"Strict overlap {overlap_pct:.1f}%, base-form {base_overlap_pct:.1f}% — incompatible"

    return {
        "decision": decision,
        "weight": weight,
        "reason": reason,
        "bible_verses": len(bible_mbay),
        "keegan_entries": len(keegan_mbay),
        "bible_word_types": len(bible_wt),
        "keegan_word_types": len(keegan_wt),
        "overlap_word_types": len(overlap),
        "overlap_pct": round(overlap_pct, 1),
        "base_overlap_word_types": len(base_overlap),
        "base_overlap_pct": round(base_overlap_pct, 1),
        "trivial_fixes": {k: repr(v) for k, v in trivial_fixes.items()},
        "fatal_mismatches": [(ch, name, count) for ch, name, count in fatal_mismatches],
        "bible_only_codepoints": len(bible_only),
        "keegan_only_codepoints": len(keegan_only),
    }


def write_report(report: dict, path: Path) -> None:
    lines = [
        "# Bible Orthography Compatibility Report\n",
        f"## Decision: **{report['decision'].upper()}** (weight={report['weight']})\n",
        f"**Reason:** {report['reason']}\n",
        f"## Corpus Stats\n",
        f"- Bible Mbay verses: {report['bible_verses']:,}",
        f"- Keegan Mbay entries: {report['keegan_entries']:,}",
        f"- Bible word-types (after normalization): {report['bible_word_types']:,}",
        f"- Keegan word-types: {report['keegan_word_types']:,}",
        f"- Overlapping word-types (strict): {report['overlap_word_types']:,} ({report['overlap_pct']}%)",
        f"- Overlapping word-types (base-form, tone-stripped): {report.get('base_overlap_word_types', 'N/A')} ({report.get('base_overlap_pct', 'N/A')}%)\n",
        f"## Character Analysis\n",
        f"- Bible-only codepoints: {report['bible_only_codepoints']}",
        f"- Keegan-only codepoints: {report['keegan_only_codepoints']}\n",
        "### Trivial Fixes (auto-applied)\n",
    ]
    for src, tgt in report["trivial_fixes"].items():
        lines.append(f"- `{repr(src)}` → `{tgt}`")
    lines.append("\n### Fatal Mismatches\n")
    if report["fatal_mismatches"]:
        for ch, name, count in report["fatal_mismatches"]:
            lines.append(f"- `{repr(ch)}` ({name}): {count:,} occurrences")
    else:
        lines.append("None.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Report written to {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bible", type=Path, required=True)
    p.add_argument("--keegan", type=Path, required=True)
    p.add_argument("--report", type=Path, default=Path("datasets/processed/bible-compatibility-report.md"))
    args = p.parse_args()

    bible_df = pd.read_parquet(args.bible)
    keegan_df = pd.read_parquet(args.keegan)

    report = run_gate(bible_df, keegan_df)
    write_report(report, args.report)

    print(f"\nDecision: {report['decision']} (weight={report['weight']})")
    print(f"Reason: {report['reason']}")


if __name__ == "__main__":
    main()
