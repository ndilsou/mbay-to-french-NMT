"""Build an ASR manifest by joining Keegan XML sound references to S3 audio.

Run from ``py/``:

    UV_CACHE_DIR=/tmp/uv-cache-asr UV_NO_SYNC=1 PYTHONPATH=components:bases \
      uv run --no-sync python -m mbay_nmt.asr_data.build_manifest
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import boto3
import pandas as pd
from lxml import etree

from mbay_nmt.data_sources.keegan.font_decoder import decode_sil_legacy
from mbay_nmt.data_sources.keegan.parser import LANGUAGES

DEFAULT_XML_DIR = Path("datasets/raw/keegan/xml")
DEFAULT_OUTPUT = Path("datasets/processed/asr-manifest.parquet")
DEFAULT_BUCKET = "production-mbay-dictiona-publicfilesbucket3eb487b-1jjyf8g110a7q"
DEFAULT_PREFIX = "sounds/"
DEFAULT_PROFILE = "mbay"
DEFAULT_REGION = "eu-west-1"


@dataclass(frozen=True)
class ManifestRow:
    language: str
    text: str
    kind: str
    soundfile_ref: str
    s3_key: str | None = None
    duration_hint_bytes: int | None = None


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


def _field(row: etree._Element, name: str) -> str | None:
    el = row.find(name)
    if el is None or el.text is None:
        return None
    text = " ".join(el.text.split())
    return text or None


def _norm_source(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(decode_sil_legacy(text, nfc=True).split()))


def _clean_sound_ref(soundfile_ref: str) -> str | None:
    soundfile_ref = soundfile_ref.strip()
    if not soundfile_ref or not soundfile_ref.lower().endswith(".wav"):
        return None
    return Path(soundfile_ref).name


def _row_for(language: str, text: str | None, kind: str, soundfile_ref: str | None) -> ManifestRow | None:
    if not text or not soundfile_ref:
        return None
    clean_ref = _clean_sound_ref(soundfile_ref)
    if not clean_ref:
        return None
    clean_text = _norm_source(text)
    if not clean_text or "\ufffd" in clean_text:
        return None
    return ManifestRow(language=language, text=clean_text, kind=kind, soundfile_ref=clean_ref)


def _parse_xml(path: Path) -> etree._Element:
    parser = etree.XMLParser(recover=False, remove_blank_text=False)
    return etree.parse(str(path), parser).getroot()


def _iter_mbay_rows(xml_path: Path) -> Iterator[ManifestRow]:
    for row in _parse_xml(xml_path):
        if row.tag == "Entry":
            out = _row_for("mbay", _field(row, "ENTRY"), "headword", _field(row, "SOUNDFILE"))
        elif row.tag == "Samples":
            out = _row_for("mbay", _field(row, "SAMPLESENT"), "sentence", _field(row, "SOUNDFILE"))
        elif row.tag == "Expressions":
            out = _row_for("mbay", _field(row, "IDIOM_EXP"), "expression", _field(row, "SOUNDEXPR"))
            if out is not None:
                yield out
            out = _row_for("mbay", _field(row, "SAMPLESENT"), "sentence", _field(row, "SOUNDFILE"))
        else:
            out = None
        if out is not None:
            yield out


def _language_code_from_stem(stem: str) -> str:
    for suffix in ("Expr", "Samp", "Trans"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return LANGUAGES.get(stem, stem.lower())


def _iter_sister_rows(xml_path: Path) -> Iterator[ManifestRow]:
    stem = xml_path.stem
    language = _language_code_from_stem(stem)
    if stem.endswith("Trans"):
        return

    if stem.endswith("Samp"):
        for row in _parse_xml(xml_path):
            out = _row_for(language, _field(row, "SAMPLESENT"), "sentence", _field(row, "SOUNDFILE"))
            if out is not None:
                yield out
        return

    if stem.endswith("Expr"):
        for row in _parse_xml(xml_path):
            out = _row_for(language, _field(row, "IDIOM_EXP"), "expression", _field(row, "SOUNDEXPR"))
            if out is not None:
                yield out
            out = _row_for(language, _field(row, "SAMPLESENT"), "sentence", _field(row, "SOUNDFILE"))
            if out is not None:
                yield out
        return

    for row in _parse_xml(xml_path):
        out = _row_for(language, _field(row, "ENTRY"), "headword", _field(row, "SOUNDFILE"))
        if out is not None:
            yield out


def iter_xml_rows(xml_dir: Path) -> Iterator[ManifestRow]:
    """Yield ASR text/audio reference rows from Mbay and sister-language XML files."""
    mbay_path = xml_dir / "Mbay.xml"
    if mbay_path.exists():
        yield from _iter_mbay_rows(mbay_path)
    else:
        print(f"WARN: missing {mbay_path}", file=sys.stderr)

    for xml_path in sorted(xml_dir.glob("*.xml")):
        if xml_path.name in {"Languages.xml", "Mbay.xml"}:
            continue
        if ".wav" not in xml_path.read_text(errors="ignore"):
            continue
        yield from _iter_sister_rows(xml_path)


def list_s3_audio_objects(
    bucket: str,
    prefix: str,
    *,
    profile: str,
    region: str,
) -> dict[str, S3Object]:
    """List all S3 objects under prefix once and index them by key."""
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("s3")
    paginator = client.get_paginator("list_objects_v2")

    objects: dict[str, S3Object] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".mp3"):
                objects[key] = S3Object(key=key, size=int(obj["Size"]))
    return objects


def join_s3(rows: Iterable[ManifestRow], s3_objects: dict[str, S3Object], prefix: str) -> list[ManifestRow]:
    by_basename_lower: dict[str, S3Object] = {}
    for obj in s3_objects.values():
        by_basename_lower.setdefault(Path(obj.key).name.lower(), obj)

    joined: list[ManifestRow] = []
    prefix = prefix if prefix.endswith("/") else f"{prefix}/"
    for row in rows:
        mp3_name = f"{Path(row.soundfile_ref).stem}.mp3"
        exact_key = f"{prefix}{mp3_name}"
        s3_obj = s3_objects.get(exact_key) or by_basename_lower.get(mp3_name.lower())
        if s3_obj is None:
            joined.append(row)
        else:
            joined.append(
                ManifestRow(
                    language=row.language,
                    text=row.text,
                    kind=row.kind,
                    soundfile_ref=row.soundfile_ref,
                    s3_key=s3_obj.key,
                    duration_hint_bytes=s3_obj.size,
                )
            )
    return joined


def rows_to_dataframe(rows: Iterable[ManifestRow]) -> pd.DataFrame:
    columns = ["language", "text", "kind", "soundfile_ref", "s3_key", "duration_hint_bytes"]
    return pd.DataFrame([asdict(row) for row in rows], columns=columns)


def print_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("No manifest rows.")
        return

    stats = (
        df.assign(resolved=df["s3_key"].notna())
        .groupby(["language", "kind"], dropna=False)
        .agg(refs=("soundfile_ref", "size"), resolved=("resolved", "sum"))
        .reset_index()
    )
    stats["coverage_pct"] = (stats["resolved"] / stats["refs"] * 100).round(2)
    print("\nRefs and S3 coverage by language/kind")
    print(stats.to_string(index=False))

    resolved = int(df["s3_key"].notna().sum())
    total = len(df)
    total_bytes = pd.to_numeric(df["duration_hint_bytes"], errors="coerce").fillna(0).sum()
    total_mb = float(total_bytes) / 1_000_000
    print(f"\nResolved: {resolved:,}/{total:,} ({resolved / total * 100:.2f}%)")
    print(f"Total resolved audio MB: {total_mb:,.2f}")


def build_manifest(
    xml_dir: Path,
    output: Path,
    *,
    bucket: str,
    prefix: str,
    profile: str,
    region: str,
    allow_s3_failure: bool = False,
) -> pd.DataFrame:
    rows = list(iter_xml_rows(xml_dir))
    print(f"Parsed XML rows: {len(rows):,}")
    print("XML refs by language/kind:")
    for (language, kind), count in sorted(Counter((r.language, r.kind) for r in rows).items()):
        print(f"  {language:10s} {kind:10s} {count:6d}")

    try:
        s3_objects = list_s3_audio_objects(bucket, prefix, profile=profile, region=region)
    except Exception as exc:
        if not allow_s3_failure:
            raise
        print(f"WARN: S3 listing failed; writing unresolved manifest: {exc}", file=sys.stderr)
        s3_objects = {}
    print(f"Listed S3 mp3 objects: {len(s3_objects):,}")

    joined = join_s3(rows, s3_objects, prefix)
    df = rows_to_dataframe(joined)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Wrote {output}")
    print_stats(df)
    return df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--allow-s3-failure",
        action="store_true",
        help="Write an unresolved manifest with null S3 fields if S3 listing fails.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_manifest(
        args.xml_dir,
        args.output,
        bucket=args.bucket,
        prefix=args.prefix,
        profile=args.profile,
        region=args.region,
        allow_s3_failure=args.allow_s3_failure,
    )


if __name__ == "__main__":
    main()
