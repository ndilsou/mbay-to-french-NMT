"""Scrape Mbay bible from bible.com (YouVersion) and pair with English/French.

Bible.com serves SSR HTML with verse spans tagged `data-usfm="BOOK.CH.VERSE"`.
We scrape the MlL version (ID 3108, 2015 revision — uses modern Unicode orthography)
and align verse-by-verse with English (ESV-like, ID 59) and French (LSG, ID 93).

Usage:
    uv run python -m mbay_nmt.data_sources.bible \\
        --output datasets/processed/bible-mbay-parallel.parquet \\
        [--books GEN,EXO,MAT,MRK,LUK,JHN] \\
        [--all-books]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import httpx
import pandas as pd
from bs4 import BeautifulSoup

MBY_VERSION = 3108   # MlL — Maktub Ta Kumkajir (2015)
ENG_VERSION = 59     # ESV
FRA_VERSION = 93     # LSG

BOOKS_OT = [
    "GEN", "EXO", "LEV", "NUM", "DEU", "JOS", "JDG", "RUT",
    "1SA", "2SA", "1KI", "2KI", "1CH", "2CH", "EZR", "NEH",
    "EST", "JOB", "PSA", "PRO", "ECC", "SNG", "ISA", "JER",
    "LAM", "EZK", "DAN", "HOS", "JOL", "AMO", "OBA", "JON",
    "MIC", "NAM", "HAB", "ZEP", "HAG", "ZEC", "MAL",
]
BOOKS_NT = [
    "MAT", "MRK", "LUK", "JHN", "ACT", "ROM", "1CO", "2CO",
    "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI",
    "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN",
    "3JN", "JUD", "REV",
]
ALL_BOOKS = BOOKS_OT + BOOKS_NT

VERSION_ABBREVS = {MBY_VERSION: "MlL", ENG_VERSION: "ESV", FRA_VERSION: "LSG"}

RATE_LIMIT_SECONDS = 1.5  # polite: ~40 req/min


def _fetch_chapter(
    client: httpx.Client, version_id: int, book: str, chapter: int
) -> dict[str, str]:
    """Fetch a single chapter and return {verse_ref: text}."""
    abbrev = VERSION_ABBREVS.get(version_id, str(version_id))
    url = f"https://www.bible.com/bible/{version_id}/{book}.{chapter}.{abbrev}"
    try:
        resp = client.get(url, follow_redirects=True)
        if resp.status_code != 200:
            return {}
    except httpx.HTTPError as e:
        print(f"WARN: failed to fetch {url}: {e}", file=sys.stderr)
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    verses: dict[str, str] = {}

    for span in soup.find_all("span", attrs={"data-usfm": True}):
        usfm = span["data-usfm"]
        # Filter to actual verses (BOOK.CH.V), skip chapter-level divs.
        if usfm.count(".") < 2:
            continue
        text = span.get_text(separator=" ", strip=True)
        # Strip leading verse number.
        text = re.sub(r"^\d+\s*", "", text).strip()
        # Strip cross-reference annotations (e.g. "# Job 38:4-7 ; Ps. 33:6 ;")
        text = re.sub(r"#[^#]*?(?=[A-Z])", "", text).strip()
        # Strip footnote markers like "1:10 Or Land ; also verses 11, 12"
        text = re.sub(r"\d+:\d+\s+Or\s+[^.;]*[.;]?\s*", "", text).strip()
        if text:
            verses.setdefault(usfm, text)

    return verses


def _detect_chapter_count(client: httpx.Client, version_id: int, book: str) -> int:
    """Binary-search for the last chapter in a book."""
    lo, hi = 1, 150
    while lo < hi:
        mid = (lo + hi + 1) // 2
        abbrev = VERSION_ABBREVS.get(version_id, str(version_id))
        url = f"https://www.bible.com/bible/{version_id}/{book}.{mid}.{abbrev}"
        try:
            resp = client.get(url, follow_redirects=True)
            if resp.status_code == 200 and f"{book}.{mid}" in resp.text:
                lo = mid
            else:
                hi = mid - 1
        except httpx.HTTPError:
            hi = mid - 1
        time.sleep(0.3)
    return lo


def scrape_book(
    client: httpx.Client,
    book: str,
    chapter_count: int | None = None,
) -> list[dict]:
    """Scrape all three versions of a book, align by verse, return parallel rows."""
    if chapter_count is None:
        chapter_count = _detect_chapter_count(client, MBY_VERSION, book)
        print(f"  {book}: detected {chapter_count} chapters")

    rows = []
    for ch in range(1, chapter_count + 1):
        mbay_verses = _fetch_chapter(client, MBY_VERSION, book, ch)
        time.sleep(RATE_LIMIT_SECONDS)
        eng_verses = _fetch_chapter(client, ENG_VERSION, book, ch)
        time.sleep(RATE_LIMIT_SECONDS)
        fra_verses = _fetch_chapter(client, FRA_VERSION, book, ch)
        time.sleep(RATE_LIMIT_SECONDS)

        for ref, mbay_text in mbay_verses.items():
            mbay_nfc = unicodedata.normalize("NFC", mbay_text)
            eng_text = eng_verses.get(ref, "")
            fra_text = fra_verses.get(ref, "")
            if eng_text:
                rows.append({
                    "source_lang": "mbay",
                    "target_lang": "eng",
                    "source_text": mbay_nfc,
                    "target_text": unicodedata.normalize("NFC", eng_text),
                    "domain": "bible",
                    "provenance": f"bible:MlL:{ref}",
                    "split": "",
                })
            if fra_text:
                rows.append({
                    "source_lang": "mbay",
                    "target_lang": "fra",
                    "source_text": mbay_nfc,
                    "target_text": unicodedata.normalize("NFC", fra_text),
                    "domain": "bible",
                    "provenance": f"bible:MlL:{ref}",
                    "split": "",
                })

        print(f"    {book}.{ch}: {len(mbay_verses)} Mbay verses, "
              f"{sum(1 for r in mbay_verses if r in eng_verses)} eng aligned, "
              f"{sum(1 for r in mbay_verses if r in fra_verses)} fra aligned")

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--books", type=str, default=None, help="Comma-sep book codes, e.g. GEN,MAT")
    p.add_argument("--all-books", action="store_true")
    args = p.parse_args()

    books = ALL_BOOKS if args.all_books else (args.books.split(",") if args.books else ["GEN", "MAT"])

    with httpx.Client(
        headers={"User-Agent": "mbay-nmt-research/0.1 (noncommercial academic)"},
        timeout=30,
    ) as client:
        all_rows = []
        for book in books:
            print(f"Scraping {book}...")
            rows = scrape_book(client, book)
            all_rows.extend(rows)
            print(f"  {book} total: {len(rows)} parallel pairs")

    if not all_rows:
        print("No verses scraped!")
        return

    df = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df):,} rows to {args.output}")
    print(df.groupby(["target_lang"]).size().to_string())


if __name__ == "__main__":
    main()
