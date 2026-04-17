"""CLI: parse Keegan's Sara-Bagirmi XML dump into a parquet file.

Usage:
    uv run python -m mbay_nmt.data_sources.keegan \\
        --xml-dir datasets/raw/keegan/xml \\
        --output datasets/processed/keegan-sister-langs.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from pathlib import Path

import pandas as pd

from .parser import parse_all_languages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = list(parse_all_languages(args.xml_dir))
    if not rows:
        raise SystemExit(f"No rows parsed from {args.xml_dir}")

    df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    df["split"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    counts = Counter((r.source_lang, r.domain) for r in rows)
    print(f"\nWrote {len(rows):,} rows to {args.output}\n")
    print(f"{'lang':<10} {'domain':<12} {'count':>8}")
    print("-" * 32)
    for (lang, domain), n in sorted(counts.items()):
        print(f"{lang:<10} {domain:<12} {n:>8,}")


if __name__ == "__main__":
    main()
