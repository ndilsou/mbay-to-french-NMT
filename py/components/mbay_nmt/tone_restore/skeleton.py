"""Orthography folding helpers for Mbay tone restoration."""

from __future__ import annotations

import re
import unicodedata as ud

TONE_MARKS = {"\u0300", "\u0301", "\u0304"}
NASAL_TILDE_BELOW = "\u0330"

HYPHEN_RE = re.compile(r"[-\u2010-\u2015]+")

_NASAL_BELOW_TO_ABOVE = {
    "a": "ã",
    "e": "ẽ",
    "i": "ĩ",
    "o": "õ",
    "u": "ũ",
    "A": "Ã",
    "E": "Ẽ",
    "I": "Ĩ",
    "O": "Õ",
    "U": "Ũ",
}


def strip_tones(text: str) -> str:
    """Remove only grave, acute, and macron tone marks, preserving nasalization."""
    decomposed = ud.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if ch not in TONE_MARKS)
    return ud.normalize("NFC", stripped)


def skeleton(text: str, hyphens: str = "strip") -> str:
    """Fold text to an unmarked comparison key.

    Args:
        text: Input text.
        hyphens: ``"strip"`` removes hyphens; ``"space"`` replaces hyphen runs
            with spaces and collapses whitespace.
    """
    if hyphens not in {"strip", "space"}:
        msg = f"hyphens must be 'strip' or 'space', got {hyphens!r}"
        raise ValueError(msg)

    hyphen_replacement = "" if hyphens == "strip" else " "
    text = HYPHEN_RE.sub(hyphen_replacement, text)

    decomposed = ud.normalize("NFD", text)
    unmarked = "".join(ch for ch in decomposed if ud.category(ch) != "Mn")
    folded = ud.normalize("NFC", unmarked).lower()

    if hyphens == "space":
        folded = " ".join(folded.split())
    return folded


def dict_to_bible_style(text: str) -> str:
    """Convert dictionary-style Mbay orthography to the bible-style hypothesis.

    Tone marks are stripped, and nasal vowels marked with COMBINING TILDE BELOW
    are converted to bible-style precomposed tilde-above vowels.
    """
    text = strip_tones(text)
    decomposed = ud.normalize("NFD", text)
    out: list[str] = []
    i = 0
    while i < len(decomposed):
        ch = decomposed[i]
        if i + 1 < len(decomposed) and decomposed[i + 1] == NASAL_TILDE_BELOW:
            out.append(_NASAL_BELOW_TO_ABOVE.get(ch, ch))
            i += 2
            continue
        out.append(ch)
        i += 1
    return ud.normalize("NFC", "".join(out))

