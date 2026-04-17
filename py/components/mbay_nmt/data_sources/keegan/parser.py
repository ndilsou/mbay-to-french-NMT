"""Parse Keegan's Sara-Bagirmi XML dump into unified parallel-corpus rows.

Input: the unzipped contents of `SaraBagirmiXmlXsd.zip`. Each non-Mbay
language is represented by four XML files:

    {Lang}.xml       - <Entry> or <Table> rows with ENTRY (headword), ENTRYCODE, SOUNDFILE
    {Lang}Trans.xml  - <Table> rows with ENTRYCODE + TRANSLATE (French, usually) + CATEGORY
    {Lang}Samp.xml   - <Table> rows with ENTRYCODE + SAMPLESENT + TRANS_SENT
    {Lang}Expr.xml   - <Table> rows with ENTRYCODE + IDIOM_EXP + TRANS_EXP + SAMPLESENT + TRANS_SENT

Mbay.xml is excluded because it contains only headwords/sound files with no
translations — Mbay data is sourced from mbay-dict's Postgres/CSV.

Output schema (one row per parallel pair):
    source_lang: ISO-ish language code (e.g., 'sar', 'ngambay', 'gulay', ...)
    target_lang: 'fra' (French) for all Keegan sister data
    source_text: text in the Sara-Bagirmi language (legacy-font decoded, NFC)
    target_text: French translation (NFC)
    domain:      'lexicon' | 'example' | 'expression'
    provenance:  f"keegan:{lang_xml_name}:{entrycode}[:{trancode}]"
    split:       blank here, assigned downstream

Legacy-font decoding is applied to the source-language text only. French
text was already correct Unicode in the XML.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Iterator

from lxml import etree

from ..schema import Row
from .font_decoder import decode_sil_legacy

# Language XML stem -> our short code for use in source_lang.
# All "sister" languages (Mbay-adjacent Sara-Bagirmi). Mbay intentionally absent.
LANGUAGES: dict[str, str] = {
    "Beboto": "beboto",
    "Bediondo": "bediondo",
    "Daba": "daba",
    "Gor": "gor",
    "Gulay": "gulay",
    "Kaba": "kaba",
    "KabaNa": "kabana",
    "Laka": "laka",
    "Mango": "mango",
    "Nar": "nar",
    "Ngam": "ngam",
    "Ngambay": "ngambay",
    "Sar": "sar",
}


def _iter_rows(xml_path: Path) -> Iterator[etree._Element]:
    """Yield each <Entry> or <Table> row element from a Keegan XML file."""
    if not xml_path.exists():
        print(f"WARN: missing XML file {xml_path}", file=sys.stderr)
        return
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    for child in root:
        if child.tag in ("Table", "Entry"):
            yield child


def _field(row: etree._Element, name: str) -> str | None:
    el = row.find(name)
    if el is None or el.text is None:
        return None
    return el.text.strip()


def _norm_french(text: str) -> str:
    """French text is already Unicode; just NFC + whitespace collapse."""
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _norm_source(text: str) -> str:
    """Sara-Bagirmi source text: legacy-decode, NFC, whitespace collapse."""
    decoded = decode_sil_legacy(text, nfc=True)
    return " ".join(decoded.split())


def parse_language(
    xml_dir: Path, lang_xml_name: str, lang_code: str
) -> Iterator[Row]:
    """Yield parallel rows for a single sister language."""
    entries_path = xml_dir / f"{lang_xml_name}.xml"
    trans_path = xml_dir / f"{lang_xml_name}Trans.xml"
    samp_path = xml_dir / f"{lang_xml_name}Samp.xml"
    expr_path = xml_dir / f"{lang_xml_name}Expr.xml"

    # Index headwords by ENTRYCODE (raw code as string, stripped).
    # A single ENTRYCODE may have multiple <Entry> rows in {Lang}.xml (variants);
    # we prefer the USEWORD=true row when present, else the first.
    entries: dict[str, str] = {}
    for row in _iter_rows(entries_path):
        code = _field(row, "ENTRYCODE")
        word = _field(row, "ENTRY")
        if not code or not word:
            continue
        use_word = (_field(row, "USEWORD") or "").lower() == "true"
        if code not in entries or use_word:
            entries[code] = _norm_source(word)

    # --- Lexicon pairs: headword -> French translation ---
    # One ENTRYCODE may have multiple translations (different TRANCODE). Emit
    # each as its own row so the model sees polysemy.
    for row in _iter_rows(trans_path):
        code = _field(row, "ENTRYCODE")
        trans = _field(row, "TRANSLATE")
        trancode = _field(row, "TRANCODE") or "01"
        if not code or not trans:
            continue
        src = entries.get(code)
        if not src:
            continue
        fr = _norm_french(trans)
        if not fr:
            continue
        yield Row(
            source_lang=lang_code,
            target_lang="fra",
            source_text=src,
            target_text=fr,
            domain="lexicon",
            provenance=f"keegan:{lang_xml_name}:{code}:{trancode}",
        )

    # --- Example pairs: sample sentence in language -> French translation ---
    for row in _iter_rows(samp_path):
        code = _field(row, "ENTRYCODE") or ""
        trancode = _field(row, "TRANCODE") or "01"
        src_sent = _field(row, "SAMPLESENT")
        tgt_sent = _field(row, "TRANS_SENT")
        if not src_sent or not tgt_sent:
            continue
        yield Row(
            source_lang=lang_code,
            target_lang="fra",
            source_text=_norm_source(src_sent),
            target_text=_norm_french(tgt_sent),
            domain="example",
            provenance=f"keegan:{lang_xml_name}Samp:{code}:{trancode}",
        )

    # --- Expression pairs: idiom + (optional) sample sentence ---
    for row in _iter_rows(expr_path):
        code = _field(row, "ENTRYCODE") or ""
        idiom = _field(row, "IDIOM_EXP")
        tgt_idiom = _field(row, "TRANS_EXP")
        if idiom and tgt_idiom:
            yield Row(
                source_lang=lang_code,
                target_lang="fra",
                source_text=_norm_source(idiom),
                target_text=_norm_french(tgt_idiom),
                domain="expression",
                provenance=f"keegan:{lang_xml_name}Expr:{code}:idiom",
            )
        src_sent = _field(row, "SAMPLESENT")
        tgt_sent = _field(row, "TRANS_SENT")
        if src_sent and tgt_sent:
            yield Row(
                source_lang=lang_code,
                target_lang="fra",
                source_text=_norm_source(src_sent),
                target_text=_norm_french(tgt_sent),
                domain="example",
                provenance=f"keegan:{lang_xml_name}Expr:{code}:sent",
            )


def parse_all_languages(xml_dir: Path) -> Iterator[Row]:
    for xml_name, lang_code in LANGUAGES.items():
        yield from parse_language(xml_dir, xml_name, lang_code)
