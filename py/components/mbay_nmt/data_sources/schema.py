"""Shared schema and utilities for data source modules."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(slots=True)
class Row:
    source_lang: str
    target_lang: str
    source_text: str
    target_text: str
    domain: str
    provenance: str


def nfc(text: str | None) -> str:
    """NFC-normalize and collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", " ".join(text.split()))
