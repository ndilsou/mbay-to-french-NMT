"""Download a 32-row Mbay ASR smoke set and run one tiny training epoch.

Run from py/:
    UV_CACHE_DIR=/tmp/uv-cache-asrt UV_NO_SYNC=1 PYTHONPATH=components:bases \\
      uv run --no-sync --with librosa python -m mbay_nmt.train_asr.smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import boto3
import pandas as pd

BUCKET = "production-mbay-dictiona-publicfilesbucket3eb487b-1jjyf8g110a7q"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/processed/asr-manifest.parquet"))
    parser.add_argument("--audio-dir", type=Path, default=Path("/tmp/asr-smoke-audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/asr-smoke-output"))
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--profile", default="mbay")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--model-id", default=os.environ.get("ASR_SMOKE_MODEL_ID", "facebook/mms-1b-all"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.manifest)
    subset = df[(df["language"] == "mbay") & (df["kind"] == "sentence") & df["s3_key"].notna()].head(args.rows)
    if len(subset) < args.rows:
        raise SystemExit(f"Only found {len(subset)} sentence rows with audio, wanted {args.rows}")

    session = boto3.Session(profile_name=args.profile)
    s3 = session.client("s3")
    for key in subset["s3_key"]:
        target = args.audio_dir / Path(str(key)).name
        if not target.exists():
            print(f"Downloading s3://{args.bucket}/{key} -> {target}")
            s3.download_file(args.bucket, str(key), str(target))

    smoke_manifest = args.output_dir / "asr-smoke-manifest.parquet"
    subset.to_parquet(smoke_manifest, index=False)

    cmd = [
        sys.executable,
        "-m",
        "mbay_nmt.train_asr.train",
        "--manifest",
        str(smoke_manifest),
        "--audio-dir",
        str(args.audio_dir),
        "--output-dir",
        str(args.output_dir / "train"),
        "--model-id",
        args.model_id,
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--eval-fraction",
        "0.25",
        "--max-audio-seconds",
        "12",
        "--min-audio-seconds",
        "0.1",
        "--kinds",
        "sentence",
        "--logging-steps",
        "1",
        "--eval-steps",
        "4",
        "--save-steps",
        "4",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

