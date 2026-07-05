"""Simulate 1980 Bible Mbaï orthography from Keegan dictionary orthography.

Rule inventory derived 2026-07-05 (see notes/MBAY_GRAMMAR.md §2):
tones stripped; nasal tilde below -> precomposed tilde above; nasalized y -> ny;
long (doubled) vowels collapsed; hyphenated morphemes fused solid; postposition
tɨ́ cliticized to its host; word-final ɨ dropped (after fusion); single-letter
prefixes joined with apostrophe (m-ā -> m'a, ń-tèń -> 'nten); Cyrillic у (from
the dictionary's ӯ encoding) mapped to Latin y.

`keegan_to_bible` is deterministic; `variants` adds stochastic alternates
(e.g. final -ah for word-final long aa: bāa -> bah) so a trained restorer sees
both attested styles.
"""

from __future__ import annotations

import random
import re
import unicodedata as ud

from mbay_nmt.tone_restore.skeleton import HYPHEN_RE, NASAL_TILDE_BELOW, TONE_MARKS

_NASAL_TO_ABOVE = {
    "a": "ã",
    "e": "ẽ",
    "i": "ĩ",
    "o": "õ",
    "u": "ũ",
    "ɨ": "ɨ",  # no precomposed nasal barred-i in bible inventory; nasality lost
    "A": "Ã",
    "E": "Ẽ",
    "I": "Ĩ",
    "O": "Õ",
    "U": "Ũ",
}

_CYRILLIC_Y = {"у": "y", "У": "Y"}

VOWELS = "aeiouɔɨãẽĩõũÃẼĨÕŨAEIOUƆƗ"

# single-letter morphemes Keegan joins with a hyphen; bible uses an apostrophe
_PREFIX_APOS_RE = re.compile(r"\b([mjnMJN])-(?=[" + VOWELS + "])")
_SYLLABIC_N_RE = re.compile(r"(?<![\w'])([nmNM])-(?=[a-zA-Zɓɗɨɔ])")


# caron appears on the instrumental clitic -ň ("with it"); bible writes plain n
_STRIP_MARKS = TONE_MARKS | {"̌"}


def _strip_tones_nfd(text: str) -> str:
    return "".join(ch for ch in ud.normalize("NFD", text) if ch not in _STRIP_MARKS)


def _nasals_above(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if i + 1 < len(text) and text[i + 1] == NASAL_TILDE_BELOW:
            out.append(_NASAL_TO_ABOVE.get(ch, ch))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _nasal_y_to_ny(text: str) -> str:
    """y followed by nasal vowel(s) -> ny + oral vowels (yá̰a̰ -> nyaa)."""

    def repl(m: re.Match[str]) -> str:
        y, vowels = m.group(1), m.group(2)
        ny = "Ny" if y.isupper() else "ny"
        return ny + vowels.replace(NASAL_TILDE_BELOW, "")

    return re.sub(
        r"([yY])((?:[aeiouAEIOU]" + NASAL_TILDE_BELOW + r")+)",
        repl,
        text,
    )


def _collapse_long_vowels(text: str) -> str:
    """Identical adjacent vowels (same nasality, tones already gone) -> one."""
    return re.sub(r"([" + VOWELS + r"])(\1)+", r"\1", text, flags=re.IGNORECASE)


def _fuse_locative(text: str) -> str:
    """Free postposition tɨ́ (already detoned to 'tɨ') cliticizes to its host."""
    return re.sub(r"(\w) tɨ\b", r"\1tɨ", text)


_FINAL_BARRED_I_RE = re.compile(r"([" + VOWELS + r"][\w']*?)ɨ\b")


def _drop_final_barred_i(text: str) -> str:
    """Drop word-final ɨ, but never strip a word's only vowel (kɨ́ stays kɨ)."""
    return _FINAL_BARRED_I_RE.sub(r"\1", text)


def keegan_to_bible(text: str) -> str:
    """Deterministic Keegan-orthography -> bible-style conversion."""
    text = ud.normalize("NFC", text)
    text = "".join(_CYRILLIC_Y.get(ch, ch) for ch in ud.normalize("NFD", text))
    text = _strip_tones_nfd(text)
    text = _nasal_y_to_ny(text)
    text = _nasals_above(text)
    text = _collapse_long_vowels(text)
    text = _fuse_locative(text)
    text = _PREFIX_APOS_RE.sub(r"\1'", text)
    text = HYPHEN_RE.sub("", text)
    text = _drop_final_barred_i(text)
    return ud.normalize("NFC", text)


def _collapse_with_optional_ah(text: str, rng: random.Random, p_ah: float) -> str:
    """Collapse long vowels; words ending in long aa may take bible-style -ah."""

    def per_word(m: re.Match[str]) -> str:
        word = m.group(0)
        had_final_long_a = re.search(r"aa\b", word, flags=re.IGNORECASE) is not None
        word = _collapse_long_vowels(word)
        if had_final_long_a and rng.random() < p_ah:
            word = re.sub(r"(a)\b", r"\1h", word, count=1, flags=re.IGNORECASE)
        return word

    return re.sub(r"[\w'-]+", per_word, text)


def variants(text: str, rng: random.Random, n: int = 1) -> list[str]:
    """Stochastic bible-style alternates for training-data augmentation.

    Toggles the conventions that vary in the real bible text: final -ah for
    word-final a from a collapsed long vowel, keeping the locative unfused,
    and keeping word-final ɨ.
    """
    base = ud.normalize("NFC", text)
    out: list[str] = []
    for _ in range(n):
        t = "".join(_CYRILLIC_Y.get(ch, ch) for ch in ud.normalize("NFD", base))
        t = _strip_tones_nfd(t)
        t = _nasal_y_to_ny(t)
        t = _nasals_above(t)
        t = _collapse_with_optional_ah(t, rng, p_ah=0.3)
        if rng.random() < 0.9:
            t = _fuse_locative(t)
        t = _PREFIX_APOS_RE.sub(r"\1'", t)
        t = HYPHEN_RE.sub("", t)
        if rng.random() < 0.9:
            t = _drop_final_barred_i(t)
        out.append(ud.normalize("NFC", t))
    return out
