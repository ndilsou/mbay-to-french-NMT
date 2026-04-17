"""One-shot Postgres export for delta check against CSV.

Connects to mbay-dict's Neon PG to pull entries + examples + expressions
including fields not present in the legacy CSV (grammatical notes,
expression example sentences).

Requires DATABASE_URL env var pointing to the Neon instance.

Usage:
    uv run python -m mbay_nmt.data_sources.postgres_export \\
        --output datasets/processed/mbay-pg-delta.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import psycopg

from .schema import Row, nfc


def _emit_pairs(mbay: str, eng: str, fra: str, domain: str, prov: str) -> Iterator[Row]:
    if mbay and fra:
        yield Row("mbay", "fra", mbay, fra, domain, prov)
    if mbay and eng:
        yield Row("mbay", "eng", mbay, eng, domain, prov)


def pull_from_pg(database_url: str) -> Iterator[Row]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, headword, english, french FROM entries")
            for eid, hw, eng, fra in cur.fetchall():
                yield from _emit_pairs(nfc(hw), nfc(eng), nfc(fra), "lexicon", f"pg:entries:{eid}")

            cur.execute("SELECT id, mbay, english, french FROM examples")
            for xid, mbay, eng, fra in cur.fetchall():
                yield from _emit_pairs(nfc(mbay), nfc(eng), nfc(fra), "example", f"pg:examples:{xid}")

            cur.execute("""
                SELECT id, mbay, english, french,
                       example_mbay, example_english, example_french
                FROM expressions
            """)
            for xid, mbay, eng, fra, ex_mbay, ex_eng, ex_fra in cur.fetchall():
                yield from _emit_pairs(nfc(mbay), nfc(eng), nfc(fra), "expression", f"pg:expressions:{xid}")
                yield from _emit_pairs(nfc(ex_mbay), nfc(ex_eng), nfc(ex_fra), "example", f"pg:expressions:{xid}:ex")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL"))
    args = p.parse_args()

    if not args.database_url:
        raise SystemExit("DATABASE_URL not set")

    rows = list(pull_from_pg(args.database_url))
    df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    df["split"] = ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
