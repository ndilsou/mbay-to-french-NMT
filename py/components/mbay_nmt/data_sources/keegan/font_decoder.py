"""SIL legacy font decoder for Keegan Mbay's Sara-Bagirmi XML dumps.

Mapping derived verbatim from `FontMappingsForDbaseTables.pdf` (bundled with
`SaraBagirmiXmlXsd.zip`). Source data originated in dBase files where the
string column was a single-byte encoding pointing into a custom SIL font.
When the dBase tables were converted to XML, each single byte was lifted
directly into a Unicode codepoint (so byte 0xC6 became U+00C6, `Æ`). The
result is mojibake for any byte that had a custom font glyph.

This module maps the raw codepoint back to the intended Unicode sequence.
Output is NFC where possible; sequences with multiple combining marks
(e.g., macron + nasal-below) remain as base + combining-mark sequences.
"""

from __future__ import annotations

import unicodedata

# Raw byte value -> intended Unicode sequence.
# Transcribed from FontMappingsForDbaseTables.pdf, pp.1-5.
#
# IPA/tonal sources:
# - Grave/acute/macron diacritics on vowels (tone marking)
# - Under-dot/tilde = creaky/nasalized voice
# - ɓ ɗ = voiced implosives (byte 223, 247)
# - ɨ ɘ = high-central vowel
# - ə = mid-central schwa
# - ɛ ɔ = open e / open o
# - ŋ ɲ = velar / palatal nasals
LEGACY_MAP: dict[int, str] = {
    # --- Vowels: a ---
    228: "\u0101",                # ā  (a + macron)
    224: "\u00e0",                # à  (identity)
    225: "\u00e1",                # á  (identity)
    226: "\u00e2",                # â  (identity)
    227: "\u0103",                # ă
    183: "\u00c1",                # Á
    192: "\u00c0",                # À  (identity)
    193: "\u00c1",                # Á  (identity)
    195: "\u0100",                # Ā
    188: "a\u0300\u0330",         # à̰  (a + grave + combining tilde below)
    170: "a\u0301\u0330",         # á̰
    186: "a\u0304\u0330",         # ā̰
    189: "a\u0330",               # a̰
    # --- Vowels: e / ɛ ---
    232: "\u00e8",                # è  (identity)
    233: "\u00e9",                # é  (identity)
    235: "\u0113",                # ē
    168: "\u025b\u0301",          # ɛ́
    153: "\u025b\u0300",          # ɛ̀
    138: "\u025b\u0304",          # ɛ̄
    155: "\u025b",                # ɛ
    254: "e\u0304\u0330",         # ḛ̄
    159: "e\u0300\u0330",         # ḛ̀
    207: "e\u0301\u0330",         # ḛ́
    245: "e\u0330",               # ḛ
    # --- Vowels: ə (schwa) ---
    198: "\u0259\u0300",          # ə̀
    230: "\u0259\u0304",          # ə̄
    201: "\u0259\u0301",          # ə́
    194: "\u0259",                # ə
    # --- Vowels: ɘ (close-mid central) ---
    240: "\u0258\u0304",          # ɘ̄
    164: "\u0258\u0330",          # ɘ̰
    220: "\u0258\u0300\u0330",    # ɘ̰̀
    208: "\u0258\u0301\u0330",    # ɘ̰́
    # --- Vowels: i ---
    236: "\u00ec",                # ì  (identity)
    237: "\u00ed",                # í  (identity)
    239: "\u012b",                # ī
    204: "\u00cc",                # Ì  (identity)
    206: "\u012a",                # Ī
    161: "i\u0300\u0330",         # ḭ̀
    172: "i\u0304\u0330",         # ḭ̄
    131: "i\u0301\u0330",         # ḭ́
    191: "i\u0330",               # ḭ
    # --- Vowels: ɨ (close central) ---
    135: "\u0268\u0301",          # ɨ́
    213: "\u0268\u0300",          # ɨ̀
    184: "\u0268\u0304",          # ɨ̄
    139: "\u0268",                # ɨ
    # --- Vowels: o ---
    242: "\u00f2",                # ò  (identity)
    243: "\u00f3",                # ó  (identity)
    246: "\u014d",                # ō
    210: "\u00d2",                # Ò  (identity)
    211: "\u00d3",                # Ó  (identity)
    212: "\u014c",                # Ō
    197: "o\u0300\u0330",         # ò̰
    199: "o\u0301\u0330",         # ó̰
    231: "o\u0304\u0330",         # ō̰
    162: "o\u0330",               # o̰
    # --- Vowels: ɔ (open-mid back) ---
    238: "\u0254\u0300",          # ɔ̀
    251: "\u0254\u0304",          # ɔ̄
    244: "\u0254\u0301",          # ɔ́
    234: "\u0254",                # ɔ
    140: "\u0186\u0304",          # Ɔ̄
    148: "\u0186\u0300",          # Ɔ̀
    # --- Vowels: u ---
    249: "\u00f9",                # ù  (identity)
    250: "\u00fa",                # ú  (identity)
    252: "\u016b",                # ū
    215: "\u016a",                # Ū
    217: "\u00d9",                # Ù  (identity)
    218: "\u00da",                # Ú  (identity)
    248: "u\u0301\u0330",         # ṵ́
    147: "u\u0300\u0330",         # ṵ̀
    166: "u\u0304\u0330",         # ṵ̄
    190: "u\u0330",               # ṵ
    # --- Consonants ---
    142: "\u010d",                # č
    154: "\u0161",                # š
    158: "\u017e",                # ž
    223: "\u0253",                # ɓ  (voiced bilabial implosive)
    247: "\u0257",                # ɗ  (voiced alveolar implosive)
    149: "l\u0301",               # ĺ
    165: "l\u0300",               # l̀
    163: "l\u0304",               # l̄
    171: "m\u0304",               # m̄
    175: "m\u0300",               # m̀
    187: "m\u0301",               # ḿ
    178: "\u0148",                # ň
    241: "n\u0304",               # n̄
    181: "n\u0300",               # ǹ
    209: "n\u0301",               # ń
    205: "\u014b\u0300",          # ŋ̀
    182: "\u014b\u0304",          # ŋ̄
    214: "\u014b\u0301",          # ŋ́
    167: "\u014b",                # ŋ
    133: "\u0272",                # ɲ
    174: "r\u0304",               # r̄
    169: "r\u0301",               # ŕ
    177: "r\u0300",               # r̀
    128: "w\u0301",               # ẃ
    219: "w\u0304",               # w̄
    222: "w\u0300",               # ẁ
    216: "y\u0300",               # ỳ
    253: "\u00fd",                # ý  (identity)
    255: "y\u0304",               # ȳ
}


# Source codepoint (as it appears in the XML, post CP1252 round-trip) -> target.
# The dBase→XML conversion decoded bytes using CP1252, so byte 0x95 became
# U+2022 (•), byte 0x8C became U+0152 (Œ), etc. Bytes outside 0x80-0x9F
# retain their Latin-1 codepoint (byte 0xC6 → U+00C6 Æ).
def _build_codepoint_map() -> dict[int, str]:
    out: dict[int, str] = {}
    for byte_val, target in LEGACY_MAP.items():
        # CP1252 defines all bytes 0x00-0xFF, so decode always succeeds.
        src_char = bytes([byte_val]).decode("cp1252")
        out[ord(src_char)] = target
    return out


_CODEPOINT_MAP: dict[int, str] = _build_codepoint_map()


def decode_sil_legacy(text: str, *, nfc: bool = True) -> str:
    """Decode a string produced by the dBase→XML lift that CP1252-decoded the raw bytes.

    For each character, if its codepoint matches an entry in the legacy
    mapping table (translated through CP1252), replace with the target
    Unicode sequence. Other characters pass through. The result is optionally
    NFC-normalized (default on).
    """
    if not text:
        return text
    out_parts: list[str] = []
    for ch in text:
        mapped = _CODEPOINT_MAP.get(ord(ch))
        out_parts.append(mapped if mapped is not None else ch)
    out = "".join(out_parts)
    if nfc:
        out = unicodedata.normalize("NFC", out)
    return out
