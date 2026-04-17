"""Convert the existing mbay-translations-review.csv to the unified parallel-corpus schema.

The CSV was produced in 2023 from MongoDB and contains clean Unicode text
(not legacy-encoded). Columns: id, entry_id, type, mbay, french, english, edited.

Output: rows in unified schema for both Mbay→French and Mbay→English directions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import Row, nfc


def parse_mbay_csv(csv_path: Path) -> Iterator[Row]:
    """Yield parallel rows from the Mbay translations CSV.

    Emits two rows per record: Mbay→French and Mbay→English (both useful
    for bidirectional training). Skips rows with empty text.
    """
    df = pd.read_csv(csv_path)
    type_to_domain = {"entry": "lexicon", "example": "example", "expression": "expression"}

    for row in df.itertuples():
        mbay = nfc(row.mbay) if pd.notna(row.mbay) else ""
        french = nfc(row.french) if pd.notna(row.french) else ""
        english = nfc(row.english) if pd.notna(row.english) else ""
        domain = type_to_domain.get(getattr(row, "type", ""), "lexicon")
        prov = f"mbay-csv:{getattr(row, 'id', '')}"

        if mbay and french:
            yield Row("mbay", "fra", mbay, french, domain, prov)
        if mbay and english:
            yield Row("mbay", "eng", mbay, english, domain, prov)
